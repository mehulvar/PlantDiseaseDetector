"""Canonical image-class identifiers and the PlantVillage folders they come from."""

from __future__ import annotations

from types import MappingProxyType

# PlantVillage folder name -> canonical class id used everywhere else.
FOLDER_TO_CLASS_ID = MappingProxyType(
    {
        "Pepper__bell___Bacterial_spot": "pepper_bacterial_spot",
        "Pepper__bell___healthy": "pepper_healthy",
        "Potato___Early_blight": "potato_early_blight",
        "Potato___Late_blight": "potato_late_blight",
        "Potato___healthy": "potato_healthy",
        "Tomato_Bacterial_spot": "tomato_bacterial_spot",
        "Tomato_Early_blight": "tomato_early_blight",
        "Tomato_Late_blight": "tomato_late_blight",
        "Tomato_Leaf_Mold": "tomato_leaf_mold",
        "Tomato_Septoria_leaf_spot": "tomato_septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "tomato_spider_mites",
        "Tomato__Target_Spot": "tomato_target_spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "tomato_yellow_leaf_curl_virus",
        "Tomato__Tomato_mosaic_virus": "tomato_mosaic_virus",
        "Tomato_healthy": "tomato_healthy",
    }
)

CROPS = ("pepper", "potato", "tomato")


def crop_of(class_id: str) -> str:
    """Return the crop prefix of a class id, e.g. 'tomato_late_blight' -> 'tomato'."""
    return class_id.split("_", 1)[0]


def is_healthy_class(class_id: str) -> bool:
    return class_id.endswith("_healthy")
