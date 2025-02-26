import asyncio
import io
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
from pydantic import BaseModel
from fastapi.responses import FileResponse
import json
import os
import torch
from diffusers import StableDiffusionPipeline
import tempfile

settings = get_settings()

negative_prompts = "font++, typo++, signature, text++, watermark++, cropped, disfigured, duplicate, error, " \
                   "jpeg artifacts, low quality, lowres, mutated hands, out of frame, worst quality"

MODEL_ID = "stabilityai/stable-diffusion-2-base"
GUIDANCE_SCALE = 5


def build_pipeline_from_model_id(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return pipe


def prompt_builder(lyrics_infos, music_style):
    # Check if a sentiment is dominant
    dominant_sentiment = None
    sentiment_prompts = ""
    for sentiment in lyrics_infos["sentiments"]:
        if lyrics_infos["sentiments"][sentiment] > 0.5:
            dominant_sentiment = sentiment
            break
    if dominant_sentiment is not None and not "others":
        sentiment_prompts = f'with a {dominant_sentiment}++++ sentiment '

    prompt = (f'A {music_style["genre_top"]} album cover {sentiment_prompts}'
              f'but without any text and illustrating the following themes:')
    for i, word in enumerate(lyrics_infos["top_words"]):
        if i == 0:
            prompt += f' {word}++++++'
        elif i == 1:
            prompt += f', {word}++++'
        elif i == 2:
            prompt += f', {word}++'
        else:
            prompt += f', {word}'
    return prompt


class MyService(Service):
    """
    Album Cover Art Generation service
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: StableDiffusionPipeline
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Album Cover Art Generation",
            slug="album-cover-art-generation",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="lyrics_analysis", type=[FieldDescriptionType.APPLICATION_JSON]),
                FieldDescription(name="music_style", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            data_out_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_GENERATION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_GENERATION
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)
        # load model from checkpoint ckpt file
        print("Loading model")
        self._model = build_pipeline_from_model_id(MODEL_ID)
        print("Model loaded")

    def process(self, data):
        lyrics_analysis = json.loads(data["lyrics_analysis"].data)
        print(f"Lyrics analysis: {lyrics_analysis}")

        music_style = json.loads(data["music_style"].data)
        print(f"Music style: {music_style}")

        prompt = prompt_builder(lyrics_analysis, music_style)
        print(f"Prompt: {prompt}")

        image = self._model(
            prompt=prompt,
            negative_prompts_embeds=negative_prompts,
            guidance_scale=GUIDANCE_SCALE,
        ).images[0]
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        return {
            "image": TaskData(
                data=image_bytes,
                type=FieldDescriptionType.IMAGE_PNG,
            ),
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Create tmp directory
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_summary = """
Generate art image (album cover) from lyrics and sentiments.
"""

api_description = """Album Cover Art Generation is an image generation API
that allows you to generate an art image (album cover)
from lyrics and music style generated with (stabilityai/stable-diffusion-2-base)
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Album Cover Art Generation API.",
    description=api_description,
    version="0.0.1",
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)


class LyricsAnalysis(BaseModel):
    language: str
    sentiments: dict[str, float]
    top_words: list[str]


class MusicStyle(BaseModel):
    genre_top: str


class Data(BaseModel):
    lyrics_analysis: LyricsAnalysis
    music_style: MusicStyle


@app.post("/process", tags=['Process'])
async def handle_process(data: Data):
    # delete previous temp files
    for f in os.listdir("./tmp"):
        if f.startswith("tmp"):
            os.remove(f"./tmp/{f}")
    lyrics_analysis = data.lyrics_analysis
    music_style = data.music_style

    lyrics_analysis = json.dumps(lyrics_analysis.dict())
    music_style = json.dumps(music_style.dict())

    print("Calling art generation service")
    result = MyService().process(
        {
            "lyrics_analysis":
                TaskData(data=lyrics_analysis.encode(), type=FieldDescriptionType.APPLICATION_JSON),
            "music_style":
                TaskData(data=music_style.encode(), type=FieldDescriptionType.APPLICATION_JSON),
        })

    image = result["image"].data

    with tempfile.NamedTemporaryFile(prefix="./tmp", suffix=".png", delete=False) as temp_file:
        temp_file.write(image)
        temp_file.close()
        return FileResponse(temp_file.name, media_type="image/png", filename="image.png")
