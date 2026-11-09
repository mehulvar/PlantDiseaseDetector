"""Train the soil/environment plant-stress classifier.

The model predicts Healthy / Moderate Stress / High Stress from sensor readings.
Real users rarely own every sensor, so the model must cope with missing values:
we train a HistGradientBoostingClassifier (native NaN support) on the original
rows plus copies with randomly masked features, so it learns sensible fallbacks.

Evaluation is grouped by Plant_ID (unseen plants), and is reported for the full
feature set and for realistic partial inputs (e.g. a basic soil test kit).

Usage:  python ml/train_soil.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.soil_features import FEATURES, SOIL_CLASSES  # noqa: E402

LOGGER = logging.getLogger("train_soil")
DATA_PATH = ROOT / "data" / "plant_health_data.csv"
MODELS_DIR = ROOT / "backend" / "models"
OUTPUTS_DIR = ROOT / "ml" / "outputs"
SEED = 42
MASK_COPIES = 6
MASK_PROBABILITY = 0.35
CV_FOLDS = 5

FEATURE_SUBSETS = {
    "all_features": [f.key for f in FEATURES],
    "soil_test_kit": ["soil_moisture", "soil_ph", "nitrogen", "phosphorus", "potassium"],
    "npk_and_ph_only": ["soil_ph", "nitrogen", "phosphorus", "potassium"],
    "environment_only": ["ambient_temperature", "humidity", "light_intensity"],
    "moisture_only": ["soil_moisture"],
}


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATA_PATH)
    missing = {f.source_column for f in FEATURES} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    x = frame[[f.source_column for f in FEATURES]].set_axis([f.key for f in FEATURES], axis=1)
    y = frame["Plant_Health_Status"].map({c: i for i, c in enumerate(SOIL_CLASSES)})
    if y.isna().any():
        raise ValueError(f"Unexpected labels: {set(frame.Plant_Health_Status) - set(SOIL_CLASSES)}")
    return x.astype(float), y.to_numpy(dtype=int), frame["Plant_ID"].to_numpy()


def augment_with_masks(x: pd.DataFrame, y: np.ndarray, rng: np.random.Generator):
    """Return the original rows plus copies with random feature subsets set to NaN."""
    copies_x, copies_y = [x], [y]
    for _ in range(MASK_COPIES):
        mask = rng.random(x.shape) < MASK_PROBABILITY
        all_masked = mask.all(axis=1)
        mask[all_masked, rng.integers(0, x.shape[1], all_masked.sum())] = False
        copies_x.append(x.mask(mask))
        copies_y.append(y)
    return pd.concat(copies_x, ignore_index=True), np.concatenate(copies_y)


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=SEED,
    )


def fit_model(x: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    # Fit on a plain array (columns in FEATURES order) so serving needs no pandas.
    return build_model().fit(x.to_numpy(), y)


def restrict(x: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    return x.assign(**{c: np.nan for c in x.columns if c not in keep})


def cross_validate(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict:
    """Grouped CV: every fold tests on plants never seen in training."""
    rng = np.random.default_rng(SEED)
    predictions = {name: np.zeros((len(y), len(SOIL_CLASSES))) for name in FEATURE_SUBSETS}
    for train_idx, test_idx in GroupKFold(n_splits=CV_FOLDS).split(x, y, groups):
        model = fit_model(*augment_with_masks(x.iloc[train_idx], y[train_idx], rng))
        for name, keep in FEATURE_SUBSETS.items():
            predictions[name][test_idx] = model.predict_proba(restrict(x.iloc[test_idx], keep).to_numpy())
    results = {}
    for name, probs in predictions.items():
        y_pred = probs.argmax(1)
        results[name] = {
            "features": FEATURE_SUBSETS[name],
            "accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "macro_f1": round(float(f1_score(y, y_pred, average="macro")), 4),
        }
        LOGGER.info("%-18s acc=%.4f macro_f1=%.4f", name, results[name]["accuracy"], results[name]["macro_f1"])
    full_pred = predictions["all_features"].argmax(1)
    report = classification_report(y, full_pred, target_names=list(SOIL_CLASSES), output_dict=True)
    np.save(OUTPUTS_DIR / "soil_cv_probabilities.npy", predictions["all_features"])
    np.save(OUTPUTS_DIR / "soil_cv_labels.npy", y)
    for name, probs in predictions.items():
        np.save(OUTPUTS_DIR / f"soil_cv_probabilities_{name}.npy", probs)
    return {
        "scenarios": results,
        "per_class_all_features": {c: {k: round(v, 4) for k, v in report[c].items()} for c in SOIL_CLASSES},
        "confusion_matrix_all_features": confusion_matrix(y, full_pred).tolist(),
    }


def feature_profile(x: pd.DataFrame, y: np.ndarray) -> dict:
    """Training ranges plus the healthy-class profile used for explanations."""
    healthy = x[y == SOIL_CLASSES.index("Healthy")]
    return {
        key: {
            "train_min": round(float(x[key].min()), 4),
            "train_max": round(float(x[key].max()), 4),
            "healthy_median": round(float(healthy[key].median()), 4),
            "healthy_p10": round(float(healthy[key].quantile(0.10)), 4),
            "healthy_p90": round(float(healthy[key].quantile(0.90)), 4),
        }
        for key in x.columns
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    x, y, groups = load_dataset()
    LOGGER.info("Loaded %d rows, %d plants, class counts %s", len(y), len(set(groups)), np.bincount(y))

    cv = cross_validate(x, y, groups)
    final = fit_model(*augment_with_masks(x, y, np.random.default_rng(SEED)))
    joblib.dump(final, MODELS_DIR / "soil_stress.joblib")

    metadata = {
        "classes": list(SOIL_CLASSES),
        "features": [f.key for f in FEATURES],
        "model": "HistGradientBoostingClassifier trained with random feature masking",
        "sklearn_version": sklearn.__version__,
        "rows": int(len(y)),
        "evaluation": f"{CV_FOLDS}-fold GroupKFold by Plant_ID (unseen plants)",
        "profile": feature_profile(x, y),
        **cv,
    }
    (MODELS_DIR / "soil_stress.json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Saved soil model to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
