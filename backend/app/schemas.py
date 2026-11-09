"""Request and response schemas for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from app.services.soil_features import FEATURES

Crop = Literal["pepper", "potato", "tomato"]
HealthStatus = Literal["healthy", "at_risk", "unhealthy"]


def _soil_reading_fields() -> dict:
    return {
        f.key: (
            float | None,
            Field(default=None, ge=f.minimum, le=f.maximum, description=f"{f.label} ({f.unit})"),
        )
        for f in FEATURES
    }


_SoilReadingBase = create_model(
    "_SoilReadingBase", __config__=ConfigDict(extra="forbid"), **_soil_reading_fields()
)


class SoilReading(_SoilReadingBase):
    """Any subset of sensor readings; at least one value is required."""

    @model_validator(mode="after")
    def require_one_reading(self) -> "SoilReading":
        if not self.provided():
            raise ValueError("Provide at least one soil or environment reading.")
        return self

    def provided(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ClassProbabilityOut(BaseModel):
    class_id: str
    name: str
    crop: str
    probability: float


class DiseaseInfoOut(BaseModel):
    class_id: str
    crop: str
    name: str
    healthy: bool
    pathogen: str
    pathogen_type: str
    summary: str
    symptoms: list[str]
    management: list[str]


class ImageResultOut(BaseModel):
    prediction: ClassProbabilityOut
    top_predictions: list[ClassProbabilityOut]
    p_healthy: float
    predicted_crop: str
    crop_hint: Crop | None
    low_confidence: bool
    crop_mismatch: bool
    disease: DiseaseInfoOut


class SoilDriverOut(BaseModel):
    feature: str
    label: str
    value: float
    unit: str
    healthy_low: float
    healthy_high: float
    impact: float
    message: str


class SoilResultOut(BaseModel):
    status: str
    probabilities: dict[str, float]
    p_healthy: float
    provided_features: list[str]
    informative: bool
    drivers: list[SoilDriverOut]
    out_of_training_range: list[str]


class FusionOut(BaseModel):
    health_score: float = Field(description="Fused probability the plant is healthy, 0-100.")
    status: HealthStatus
    agreement: Literal["agree", "conflict", "single_source"]
    sources: list[Literal["image", "soil"]]
    contributions: dict[str, float]
    headline: str
    explanation: str


class AssessmentOut(BaseModel):
    assessment: FusionOut
    image: ImageResultOut | None
    soil: SoilResultOut | None
    recommendations: list[str]
    warnings: list[str]
    disclaimer: str


class FeatureOut(BaseModel):
    key: str
    label: str
    unit: str
    group: str
    minimum: float
    maximum: float
    step: float
    description: str
    healthy_low: float
    healthy_high: float
    typical_min: float
    typical_max: float


class SampleOut(BaseModel):
    id: str
    title: str
    description: str
    crop: Crop | None
    image_url: str | None
    expected_label: str | None
    soil: dict[str, float] | None


class MetaOut(BaseModel):
    crops: list[Crop]
    classes: list[DiseaseInfoOut]
    soil_features: list[FeatureOut]
    soil_classes: list[str]
    metrics: dict


class HealthOut(BaseModel):
    status: Literal["ok"]
    image_model: bool
    soil_model: bool
    version: str


class ErrorOut(BaseModel):
    detail: str
