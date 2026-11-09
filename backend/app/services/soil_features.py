"""Single source of truth for soil/environment sensor features.

Bounds are *physical plausibility* limits used for input validation; they are
deliberately wider than the training data range. Readings outside the training
range are accepted but flagged as extrapolation in the response.
"""

from __future__ import annotations

from dataclasses import dataclass

SOIL_CLASSES = ("Healthy", "Moderate Stress", "High Stress")


@dataclass(frozen=True)
class Feature:
    key: str
    source_column: str
    label: str
    unit: str
    group: str
    minimum: float
    maximum: float
    step: float
    description: str


FEATURES: tuple[Feature, ...] = (
    Feature("soil_moisture", "Soil_Moisture", "Soil moisture", "%", "soil", 0, 100, 0.1,
            "Volumetric water content of the root zone."),
    Feature("soil_ph", "Soil_pH", "Soil pH", "pH", "soil", 0, 14, 0.01,
            "Acidity/alkalinity of the soil."),
    Feature("nitrogen", "Nitrogen_Level", "Nitrogen (N)", "mg/kg", "soil", 0, 500, 0.1,
            "Available nitrogen from a soil test."),
    Feature("phosphorus", "Phosphorus_Level", "Phosphorus (P)", "mg/kg", "soil", 0, 500, 0.1,
            "Available phosphorus from a soil test."),
    Feature("potassium", "Potassium_Level", "Potassium (K)", "mg/kg", "soil", 0, 500, 0.1,
            "Available potassium from a soil test."),
    Feature("soil_temperature", "Soil_Temperature", "Soil temperature", "°C", "soil", -20, 60, 0.1,
            "Temperature of the soil at root depth."),
    Feature("ambient_temperature", "Ambient_Temperature", "Air temperature", "°C", "environment",
            -20, 60, 0.1, "Air temperature around the plant."),
    Feature("humidity", "Humidity", "Relative humidity", "%", "environment", 0, 100, 0.1,
            "Relative humidity of the air."),
    Feature("light_intensity", "Light_Intensity", "Light intensity", "lux", "environment",
            0, 200_000, 1, "Light reaching the canopy."),
    Feature("chlorophyll_content", "Chlorophyll_Content", "Chlorophyll content", "SPAD", "plant",
            0, 100, 0.1, "Leaf chlorophyll meter reading."),
    Feature("electrochemical_signal", "Electrochemical_Signal", "Electrochemical signal", "mV",
            "plant", 0, 10, 0.001, "Plant electrophysiology sensor reading."),
)

FEATURE_KEYS = tuple(f.key for f in FEATURES)
FEATURES_BY_KEY = {f.key: f for f in FEATURES}
