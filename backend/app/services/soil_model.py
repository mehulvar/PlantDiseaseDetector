"""Soil/environment stress classifier with per-feature explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.services.soil_features import FEATURE_KEYS, FEATURES_BY_KEY, SOIL_CLASSES

# Features whose counterfactual shifts P(healthy) by at least this much are reported.
DRIVER_MIN_IMPACT = 0.05
MAX_DRIVERS = 4


@dataclass(frozen=True)
class SoilDriver:
    feature: str
    value: float
    healthy_low: float
    healthy_high: float
    impact: float  # increase in P(healthy) if this reading were at the healthy median
    message: str


@dataclass(frozen=True)
class SoilPrediction:
    probabilities: dict[str, float]
    status: str
    p_healthy: float
    provided: tuple[str, ...]
    informative: bool
    drivers: tuple[SoilDriver, ...]
    out_of_range: tuple[str, ...]


class SoilClassifier:
    # Readings that carry most of the signal in the training data; without at
    # least one of them the model can only return (roughly) its prior.
    KEY_FEATURES = ("soil_moisture", "nitrogen")

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.metadata = json.loads(metadata_path.read_text())
        if tuple(self.metadata["features"]) != FEATURE_KEYS:
            raise ValueError("Soil model features do not match the feature registry.")
        if tuple(self.metadata["classes"]) != SOIL_CLASSES:
            raise ValueError("Soil model classes do not match the expected classes.")
        self._model = joblib.load(model_path)
        self.profile: dict[str, dict[str, float]] = self.metadata["profile"]
        self.training_prior_healthy = self._training_prior()

    def _training_prior(self) -> float:
        per_class = self.metadata["per_class_all_features"]
        total = sum(v["support"] for v in per_class.values())
        return per_class["Healthy"]["support"] / total

    @staticmethod
    def _matrix(rows: list[dict[str, float]]) -> np.ndarray:
        """Rows of readings -> array in FEATURE_KEYS column order, NaN for missing."""
        return np.array([[row.get(k, np.nan) for k in FEATURE_KEYS] for row in rows], dtype=float)

    def _p_healthy(self, matrix: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(matrix)[:, SOIL_CLASSES.index("Healthy")]

    def _drivers(self, readings: dict[str, float], base_p: float) -> tuple[SoilDriver, ...]:
        """Counterfactual explanation: move each reading to the healthy median."""
        keys = list(readings)
        rows = [{**readings, key: self.profile[key]["healthy_median"]} for key in keys]
        impacts = self._p_healthy(self._matrix(rows)) - base_p
        drivers = []
        for key, impact in sorted(zip(keys, impacts), key=lambda kv: -kv[1]):
            if impact < DRIVER_MIN_IMPACT or len(drivers) == MAX_DRIVERS:
                continue
            drivers.append(self._describe_driver(key, readings[key], float(impact)))
        return tuple(drivers)

    def _describe_driver(self, key: str, value: float, impact: float) -> SoilDriver:
        feature = FEATURES_BY_KEY[key]
        low, high = self.profile[key]["healthy_p10"], self.profile[key]["healthy_p90"]
        healthy_range = f"{low:g}-{high:g} {feature.unit}"
        reading = f"{feature.label} of {value:g} {feature.unit}"
        if value < low:
            message = f"{reading} is below the range seen in healthy plants ({healthy_range})."
        elif value > high:
            message = f"{reading} is above the range seen in healthy plants ({healthy_range})."
        else:
            message = (
                f"{reading} is inside the healthy range ({healthy_range}), but moving it toward "
                "typical healthy values would still improve the outlook."
            )
        return SoilDriver(key, value, low, high, round(impact, 4), message)

    def predict(self, readings: dict[str, float]) -> SoilPrediction:
        if not readings:
            raise ValueError("At least one soil or environment reading is required.")
        unknown = set(readings) - set(FEATURE_KEYS)
        if unknown:
            raise ValueError(f"Unknown readings: {sorted(unknown)}")
        probs = self._model.predict_proba(self._matrix([readings]))[0]
        probabilities = {c: float(p) for c, p in zip(SOIL_CLASSES, probs)}
        p_healthy = probabilities["Healthy"]
        out_of_range = tuple(
            k for k, v in readings.items()
            if not self.profile[k]["train_min"] <= v <= self.profile[k]["train_max"]
        )
        return SoilPrediction(
            probabilities=probabilities,
            status=max(probabilities, key=probabilities.get),
            p_healthy=p_healthy,
            provided=tuple(k for k in FEATURE_KEYS if k in readings),
            informative=any(k in readings for k in self.KEY_FEATURES),
            drivers=self._drivers(readings, p_healthy),
            out_of_range=out_of_range,
        )
