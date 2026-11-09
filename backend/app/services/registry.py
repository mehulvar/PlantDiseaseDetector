"""Model loading (cached per process), API metadata and demo samples."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.schemas import FeatureOut, MetaOut, SampleOut
from app.services.assessment import disease_out
from app.services.image_model import ImageClassifier
from app.services.soil_features import FEATURES, SOIL_CLASSES
from app.services.soil_model import SoilClassifier

SAMPLE_IMAGE_ROUTE = "/api/samples/images"


@dataclass(frozen=True)
class Services:
    image: ImageClassifier
    soil: SoilClassifier
    fusion_metrics: dict


_LOAD_LOCK = threading.Lock()


def get_services(models_dir: Path) -> Services:
    """Load models once per process; concurrent cold-start requests wait for one load."""
    with _LOAD_LOCK:
        return _load_services(models_dir)


@lru_cache(maxsize=4)
def _load_services(models_dir: Path) -> Services:
    fusion_path = models_dir / "fusion_metrics.json"
    return Services(
        image=ImageClassifier(models_dir / "leaf_disease.onnx", models_dir / "leaf_disease.json"),
        soil=SoilClassifier(models_dir / "soil_stress.joblib", models_dir / "soil_stress.json"),
        fusion_metrics=json.loads(fusion_path.read_text()) if fusion_path.is_file() else {},
    )


def build_meta(services: Services) -> MetaOut:
    profile = services.soil.profile
    image_meta, soil_meta = services.image.metadata, services.soil.metadata
    return MetaOut(
        crops=list(services.image.crops),
        classes=[disease_out(c) for c in services.image.class_ids],
        soil_features=[
            FeatureOut(
                key=f.key, label=f.label, unit=f.unit, group=f.group, minimum=f.minimum,
                maximum=f.maximum, step=f.step, description=f.description,
                healthy_low=profile[f.key]["healthy_p10"], healthy_high=profile[f.key]["healthy_p90"],
                typical_min=profile[f.key]["train_min"], typical_max=profile[f.key]["train_max"],
            )
            for f in FEATURES
        ],
        soil_classes=list(SOIL_CLASSES),
        metrics={
            "image": {
                "architecture": image_meta["architecture"],
                "test_accuracy": image_meta["test_accuracy"],
                "macro_f1": image_meta["macro_f1"],
                "test_size": image_meta["test_size"],
                "classes": len(services.image.class_ids),
            },
            "soil": {
                "evaluation": soil_meta["evaluation"],
                "rows": soil_meta["rows"],
                "scenarios": soil_meta["scenarios"],
            },
            "fusion": services.fusion_metrics,
        },
    )


@lru_cache(maxsize=4)
def _read_samples(samples_file: Path) -> tuple[SampleOut, ...]:
    if not samples_file.is_file():
        return ()
    raw = json.loads(samples_file.read_text())
    samples = []
    for item in raw:
        image = item.get("image")
        samples.append(
            SampleOut(
                id=item["id"], title=item["title"], description=item["description"],
                crop=item.get("crop"), expected_label=item.get("expected_label"),
                image_url=f"{SAMPLE_IMAGE_ROUTE}/{image}" if image else None,
                soil=item.get("soil"),
            )
        )
    return tuple(samples)


def load_samples(samples_dir: Path) -> list[SampleOut]:
    return list(_read_samples(samples_dir / "samples.json"))
