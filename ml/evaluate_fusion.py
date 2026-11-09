"""Evaluate whether fusing image + soil evidence improves healthy/unhealthy decisions.

IMPORTANT - simulated pairing. The image dataset (PlantVillage) and the soil
dataset are independent: no plant has both a photo and a sensor reading. We
therefore build *simulated pairs*: each held-out test leaf is paired with an
out-of-fold soil reading (from plants unseen by the soil model) whose health
state matches the leaf's with probability rho, and is the opposite otherwise.
rho = 1.0 assumes soil stress and visible disease always co-occur; lower values
model weaker real-world correlation. Results show how the fusion rule behaves
under those assumptions - real paired field data is needed to confirm them.

Two image-quality conditions are tested: the clean lab photos, and "field"
photos degraded with blur, poor exposure, sensor noise and JPEG compression.

Usage:  python ml/evaluate_fusion.py   (after train_image.py and train_soil.py)
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.fusion import (  # noqa: E402
    IMAGE_WEIGHT, SOIL_WEIGHT, fuse, image_evidence, soil_evidence,
)
from app.services.labels import is_healthy_class  # noqa: E402
from app.services.preprocess import image_to_tensor  # noqa: E402
from app.services.soil_features import SOIL_CLASSES  # noqa: E402
from app.services.soil_model import SoilClassifier  # noqa: E402

LOGGER = logging.getLogger("evaluate_fusion")
OUTPUTS_DIR = ROOT / "ml" / "outputs"
MODELS_DIR = ROOT / "backend" / "models"
SEED = 7
REPEATS = 5
RHOS = (1.0, 0.8, 0.6)
SOIL_SCENARIOS = ("all_features", "soil_test_kit", "moisture_only", "environment_only")
WEIGHT_SWEEP = (0.25, 0.5, 0.75, 1.0)
HEALTHY_INDEX = SOIL_CLASSES.index("Healthy")


def degrade(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Simulate a low-quality field photo: blur, bad exposure, noise, heavy JPEG."""
    small = image.resize((80, 80), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    exposed = ImageEnhance.Brightness(small).enhance(rng.uniform(0.55, 1.45))
    flat = ImageEnhance.Contrast(exposed).enhance(rng.uniform(0.6, 0.9))
    noisy = np.asarray(flat, dtype=np.float32) + rng.normal(0, 14, (*flat.size[::-1], 3))
    buffer = io.BytesIO()
    Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).save(buffer, format="JPEG", quality=20)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def degraded_probabilities(test: pd.DataFrame) -> np.ndarray:
    cache = OUTPUTS_DIR / "image_test_probabilities_degraded.npy"
    if cache.is_file():
        return np.load(cache)
    session = ort.InferenceSession(str(MODELS_DIR / "leaf_disease.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(SEED)
    probs = []
    for rel_path in test.path:
        with Image.open(ROOT / rel_path) as img:
            tensor = image_to_tensor(degrade(img.convert("RGB"), rng))
        probs.append(session.run(None, {"image": tensor})[0][0])
    result = np.stack(probs)
    np.save(cache, result)
    return result


def pair_soil(leaf_healthy: np.ndarray, soil_labels: np.ndarray, rho: float,
              rng: np.random.Generator) -> np.ndarray:
    """Pick a soil row index for each leaf: matching health state with probability rho."""
    healthy_rows = np.flatnonzero(soil_labels == HEALTHY_INDEX)
    stressed_rows = np.flatnonzero(soil_labels != HEALTHY_INDEX)
    matches = rng.random(len(leaf_healthy)) < rho
    want_healthy = np.where(matches, leaf_healthy, ~leaf_healthy)
    return np.where(
        want_healthy,
        rng.choice(healthy_rows, len(leaf_healthy)),
        rng.choice(stressed_rows, len(leaf_healthy)),
    )


def decisions(image_p: np.ndarray | None, soil_p: np.ndarray | None, soil_prior: float,
              soil_weight: float) -> np.ndarray:
    """Healthy (True) / unhealthy decisions using the production fusion rule."""
    n = len(image_p) if image_p is not None else len(soil_p)
    out = np.empty(n, dtype=bool)
    for i in range(n):
        evidence = []
        if image_p is not None:
            evidence.append(image_evidence(float(image_p[i])))
        if soil_p is not None:
            evidence.append(soil_evidence(float(soil_p[i]), soil_prior, soil_weight))
        out[i] = fuse(tuple(evidence)).p_healthy >= 0.5
    return out


def scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tpr = (predicted & truth).sum() / truth.sum()
    tnr = (~predicted & ~truth).sum() / (~truth).sum()
    return {
        "accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "diseased_recall": float(tnr),
    }


def mean_scores(runs: list[dict[str, float]]) -> dict[str, float]:
    return {k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]}


def evaluate_condition(image_healthy_p: np.ndarray, truth: np.ndarray, soil_labels: np.ndarray,
                       soil_prior: float, soil_weight: float = SOIL_WEIGHT) -> list[dict]:
    results = []
    image_only = scores(truth, decisions(image_healthy_p, None, soil_prior, soil_weight))
    soil_meta = json.loads((MODELS_DIR / "soil_stress.json").read_text())
    for scenario in SOIL_SCENARIOS:
        soil_probs = np.load(OUTPUTS_DIR / f"soil_cv_probabilities_{scenario}.npy")[:, HEALTHY_INDEX]
        features = soil_meta["scenarios"][scenario]["features"]
        # Mirror the API: readings without moisture or nitrogen get zero weight.
        informative = any(k in features for k in SoilClassifier.KEY_FEATURES)
        weight = soil_weight if informative else 0.0
        for rho in RHOS:
            rng = np.random.default_rng(SEED)
            soil_runs, fused_runs = [], []
            for _ in range(REPEATS):
                rows = pair_soil(truth, soil_labels, rho, rng)
                soil_runs.append(scores(truth, decisions(None, soil_probs[rows], soil_prior, soil_weight)))
                fused_runs.append(scores(truth, decisions(image_healthy_p, soil_probs[rows], soil_prior, weight)))
            results.append({
                "soil_inputs": scenario, "rho": rho,
                "image_only": {k: round(v, 4) for k, v in image_only.items()},
                "soil_only": mean_scores(soil_runs), "fused": mean_scores(fused_runs),
            })
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split = pd.read_csv(OUTPUTS_DIR / "image_split.csv")
    test = split[split.split == "test"].reset_index(drop=True)
    class_ids = json.loads((MODELS_DIR / "leaf_disease.json").read_text())["class_ids"]
    healthy_mask = np.array([is_healthy_class(c) for c in class_ids])
    truth = test.class_id.map(is_healthy_class).to_numpy(dtype=bool)

    soil_labels = np.load(OUTPUTS_DIR / "soil_cv_labels.npy")
    soil_prior = float((soil_labels == HEALTHY_INDEX).mean())
    conditions = {
        "clean_photos": np.load(OUTPUTS_DIR / "image_test_probabilities.npy")[:, healthy_mask].sum(1),
        "field_quality_photos": degraded_probabilities(test)[:, healthy_mask].sum(1),
    }
    report = {
        "method": "Simulated pairing of held-out leaf images with out-of-fold soil readings; "
                  "rho = probability that the soil health state matches the leaf's.",
        "test_leaves": int(len(test)),
        "healthy_share": round(float(truth.mean()), 4),
        "weights": {"image": IMAGE_WEIGHT, "soil": SOIL_WEIGHT},
        "conditions": {},
    }
    for name, image_p in conditions.items():
        report["conditions"][name] = evaluate_condition(image_p, truth, soil_labels, soil_prior)
        for row in report["conditions"][name]:
            LOGGER.info("%-21s %-17s rho=%.1f image=%.4f soil=%.4f fused=%.4f (balanced acc)",
                        name, row["soil_inputs"], row["rho"], row["image_only"]["balanced_accuracy"],
                        row["soil_only"]["balanced_accuracy"], row["fused"]["balanced_accuracy"])

    sweep = {}
    for weight in WEIGHT_SWEEP:
        rows = evaluate_condition(conditions["field_quality_photos"], truth, soil_labels, soil_prior, weight)
        sweep[str(weight)] = {f"{r['soil_inputs']}@rho={r['rho']}": r["fused"]["balanced_accuracy"] for r in rows}
    report["soil_weight_sensitivity_field_photos"] = sweep

    (OUTPUTS_DIR / "fusion_metrics.json").write_text(json.dumps(report, indent=2))
    (MODELS_DIR / "fusion_metrics.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("Wrote fusion metrics")


if __name__ == "__main__":
    main()
