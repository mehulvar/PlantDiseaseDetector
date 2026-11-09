"""FastAPI application: plant-health assessment from leaf images and/or soil data."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.config import Settings, load_settings
from app.rate_limit import RateLimiter
from app.schemas import AssessmentOut, Crop, ErrorOut, HealthOut, MetaOut, SampleOut, SoilReading
from app.services.assessment import assess
from app.services.preprocess import InvalidImageError, load_image
from app.services.registry import Services, build_meta, get_services, load_samples

VERSION = "1.0.0"
LOGGER = logging.getLogger("plant_api")
ERROR_RESPONSES = {400: {"model": ErrorOut}, 413: {"model": ErrorOut}, 422: {"model": ErrorOut},
                   429: {"model": ErrorOut}}


def format_validation_error(error: ValidationError) -> str:
    messages = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "soil"
        messages.append(f"{location}: {item['msg']}")
    return "; ".join(messages)


def parse_soil(raw: str | None) -> SoilReading | None:
    if raw is None or not raw.strip():
        return None
    try:
        return SoilReading.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid soil data - {format_validation_error(exc)}") from exc


async def read_upload(upload: UploadFile, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {limit // (1024 * 1024)} MB limit.")
    return data


def run_assessment(services: Services, image_bytes: bytes | None, soil: SoilReading | None,
                   crop: str | None) -> AssessmentOut:
    image_prediction = None
    if image_bytes is not None:
        image_prediction = services.image.predict(load_image(image_bytes), crop_hint=crop)
    readings = soil.provided() if soil is not None else {}
    soil_prediction = services.soil.predict(readings) if soil is not None else None
    return assess(image_prediction, soil_prediction, readings, services.soil.training_prior_healthy)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(
        title="Plant Health Detector API",
        version=VERSION,
        description="Detects leaf diseases from photos and plant stress from soil readings, "
                    "and fuses both into one health assessment.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    limiter = RateLimiter(limit=settings.rate_limit_per_minute, window_seconds=60)

    def services_dep() -> Services:
        return get_services(settings.models_dir)

    def rate_limit(request: Request) -> None:
        client = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if not limiter.allow(client or (request.client.host if request.client else "unknown")):
            raise HTTPException(status_code=429, detail="Too many requests - please wait a minute.")

    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            image_model=(settings.models_dir / "leaf_disease.onnx").is_file(),
            soil_model=(settings.models_dir / "soil_stress.joblib").is_file(),
            version=VERSION,
        )

    @app.get("/api/meta", response_model=MetaOut)
    def meta(services: Services = Depends(services_dep)) -> MetaOut:
        return build_meta(services)

    @app.get("/api/samples", response_model=list[SampleOut])
    def samples() -> list[SampleOut]:
        return load_samples(settings.samples_dir)

    @app.get("/api/samples/images/{filename}", response_class=FileResponse,
             responses={404: {"model": ErrorOut}})
    def sample_image(filename: str) -> FileResponse:
        images_dir = (settings.samples_dir / "images").resolve()
        path = (images_dir / filename).resolve()
        if path.parent != images_dir or not path.is_file():
            raise HTTPException(status_code=404, detail="Sample image not found.")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/assess", response_model=AssessmentOut, responses=ERROR_RESPONSES,
              dependencies=[Depends(rate_limit)])
    async def assess_plant(
        image: UploadFile | None = File(default=None, description="Leaf photo (JPEG/PNG)."),
        soil: str | None = Form(default=None, description="JSON object of soil/sensor readings."),
        crop: Crop | None = Form(default=None, description="Optional crop to narrow the diagnosis."),
        services: Services = Depends(services_dep),
    ) -> AssessmentOut:
        soil_reading = parse_soil(soil)
        image_bytes = await read_upload(image, settings.max_upload_bytes) if image is not None else None
        if image_bytes is None and soil_reading is None:
            raise HTTPException(status_code=400, detail="Provide a leaf image, soil readings, or both.")
        try:
            return await run_in_threadpool(run_assessment, services, image_bytes, soil_reading, crop)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
