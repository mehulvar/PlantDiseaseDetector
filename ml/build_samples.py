"""Build the demo samples served by the API (backend/samples).

Images come only from the held-out *test* split, so the demo never shows images
the model trained on. Soil readings are real rows from the soil dataset.

Usage:  python ml/build_samples.py   (after train_image.py and train_soil.py)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.soil_features import FEATURES  # noqa: E402

LOGGER = logging.getLogger("build_samples")
OUTPUTS_DIR = ROOT / "ml" / "outputs"
MODELS_DIR = ROOT / "backend" / "models"
SAMPLES_DIR = ROOT / "backend" / "samples"
SEED = 11
SAMPLE_IMAGE_SIZE = 384
KIT_FEATURES = ("soil_moisture", "soil_ph", "nitrogen", "phosphorus", "potassium",
                "ambient_temperature", "humidity")


@dataclass(frozen=True)
class SampleSpec:
    id: str
    title: str
    description: str
    image_class: str | None
    soil_label: str | None
    soil_features: tuple[str, ...] | None = None  # None = all features
    soil_order: str = "random"  # or "low_nitrogen" / "low_moisture"


SPECS = (
    SampleSpec("healthy-tomato", "Healthy tomato", "A clean tomato leaf and a balanced soil test.",
               "tomato_healthy", "Healthy"),
    SampleSpec("tomato-late-blight", "Tomato late blight", "Blighted leaf from a field under high stress.",
               "tomato_late_blight", "High Stress"),
    SampleSpec("potato-early-blight", "Potato early blight",
               "Early blight lesions; the soil test shows low nitrogen.",
               "potato_early_blight", "Moderate Stress", KIT_FEATURES, "low_nitrogen"),
    SampleSpec("pepper-bacterial-spot", "Pepper bacterial spot", "Photo only - no soil data.",
               "pepper_bacterial_spot", None),
    SampleSpec("tomato-leaf-curl", "Tomato yellow leaf curl", "Photo only - a whitefly-spread virus.",
               "tomato_yellow_leaf_curl_virus", None),
    SampleSpec("tomato-septoria", "Tomato septoria leaf spot", "Leaf spots plus a basic soil-kit reading.",
               "tomato_septoria_leaf_spot", "Moderate Stress", KIT_FEATURES),
    SampleSpec("healthy-leaf-dry-soil", "Healthy leaf, dry soil",
               "The leaf looks fine but the soil is parched - an early warning case.",
               "pepper_healthy", "High Stress", None, "low_moisture"),
    SampleSpec("healthy-potato", "Healthy potato", "Healthy potato leaf with good growing conditions.",
               "potato_healthy", "Healthy", KIT_FEATURES),
    SampleSpec("tomato-spider-mites", "Tomato spider mites", "Photo only - mite stippling damage.",
               "tomato_spider_mites", None),
    SampleSpec("soil-only-stressed", "Soil test: stressed field", "Sensor readings only - no photo.",
               None, "High Stress", None, "low_moisture"),
    SampleSpec("soil-only-healthy", "Soil test: healthy field", "Sensor readings only - no photo.",
               None, "Healthy"),
)


def pick_image(spec: SampleSpec, test: pd.DataFrame, probs: np.ndarray, class_ids: list[str],
               rng: np.random.Generator) -> Path:
    """A random held-out test image of the class that the model classifies confidently."""
    target = class_ids.index(spec.image_class)
    candidates = test.index[(test.class_id == spec.image_class) & (probs.argmax(1) == target)
                            & (probs[:, target] > 0.9)]
    return ROOT / test.loc[rng.choice(candidates), "path"]


def pick_soil(spec: SampleSpec, soil: pd.DataFrame, soil_probs: np.ndarray,
              rng: np.random.Generator) -> dict[str, float]:
    """A real dataset row of the requested label that the soil model got right out-of-fold."""
    classes = ["Healthy", "Moderate Stress", "High Stress"]
    label_index = classes.index(spec.soil_label)
    pool = soil[(soil.Plant_Health_Status == spec.soil_label) & (soil_probs.argmax(1) == label_index)]
    if spec.soil_order == "low_nitrogen":
        pool = pool.nsmallest(40, "Nitrogen_Level")
    elif spec.soil_order == "low_moisture":
        pool = pool.nsmallest(40, "Soil_Moisture")
    row = pool.iloc[rng.integers(len(pool))]
    keys = spec.soil_features or tuple(f.key for f in FEATURES)
    by_key = {f.key: f for f in FEATURES}
    return {k: round(float(row[by_key[k].source_column]), 2 if k != "electrochemical_signal" else 3)
            for k in keys}


def save_image(source: Path, destination: Path) -> None:
    with Image.open(source) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((SAMPLE_IMAGE_SIZE, SAMPLE_IMAGE_SIZE))
        rgb.save(destination, format="JPEG", quality=90)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(SEED)
    split = pd.read_csv(OUTPUTS_DIR / "image_split.csv")
    test = split[split.split == "test"].reset_index(drop=True)
    probs = np.load(OUTPUTS_DIR / "image_test_probabilities.npy")
    class_ids = json.loads((MODELS_DIR / "leaf_disease.json").read_text())["class_ids"]
    soil = pd.read_csv(ROOT / "data" / "plant_health_data.csv")
    soil_probs = np.load(OUTPUTS_DIR / "soil_cv_probabilities.npy")

    images_dir = SAMPLES_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for stale in images_dir.glob("*.jpg"):
        stale.unlink()

    samples = []
    for spec in SPECS:
        image_name = None
        if spec.image_class:
            image_name = f"{spec.id}.jpg"
            save_image(pick_image(spec, test, probs, class_ids, rng), images_dir / image_name)
        samples.append({
            "id": spec.id, "title": spec.title, "description": spec.description,
            "crop": spec.image_class.split("_", 1)[0] if spec.image_class else None,
            "image": image_name, "expected_label": spec.image_class,
            "soil": pick_soil(spec, soil, soil_probs, rng) if spec.soil_label else None,
        })
    (SAMPLES_DIR / "samples.json").write_text(json.dumps(samples, indent=2))
    LOGGER.info("Wrote %d samples to %s", len(samples), SAMPLES_DIR)


if __name__ == "__main__":
    main()
