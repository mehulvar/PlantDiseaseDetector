"""Turn raw model outputs into a fused, human-readable plant-health assessment."""

from __future__ import annotations

from app.schemas import (
    AssessmentOut,
    ClassProbabilityOut,
    DiseaseInfoOut,
    FusionOut,
    ImageResultOut,
    SoilDriverOut,
    SoilResultOut,
)
from app.services.fusion import SOIL_WEIGHT, Evidence, FusionResult, fuse, image_evidence, soil_evidence
from app.services.image_model import ClassProbability, ImagePrediction
from app.services.knowledge import CLASS_INFO, CROP_PH_RANGES, favourable_conditions
from app.services.soil_features import FEATURES_BY_KEY
from app.services.soil_model import SoilDriver, SoilPrediction

DISCLAIMER = (
    "Decision support only. The image model recognises 15 pepper, potato and tomato leaf "
    "classes; the soil model was trained on a small sensor dataset. Confirm serious "
    "diagnoses with a local agronomist or extension service."
)

SOIL_ACTIONS: dict[str, tuple[str, str]] = {
    # key: (advice when below healthy range, advice when above healthy range)
    "soil_moisture": (
        "Soil is drier than healthy plants experienced: irrigate deeply and mulch to retain moisture.",
        "Soil is wetter than healthy plants experienced: reduce watering and improve drainage.",
    ),
    "nitrogen": (
        "Nitrogen is low: side-dress with a nitrogen fertiliser or compost, guided by a soil test.",
        "Nitrogen is high: hold back nitrogen fertiliser to avoid soft, disease-prone growth.",
    ),
    "phosphorus": (
        "Phosphorus is low: apply a phosphorus fertiliser according to soil-test recommendations.",
        "Phosphorus is high: avoid further phosphorus applications.",
    ),
    "potassium": (
        "Potassium is low: apply potash (e.g. sulphate of potash) per soil-test recommendations.",
        "Potassium is high: avoid further potassium applications.",
    ),
}


def class_probability_out(item: ClassProbability) -> ClassProbabilityOut:
    info = CLASS_INFO[item.class_id]
    return ClassProbabilityOut(
        class_id=item.class_id, name=info.name, crop=info.crop, probability=round(item.probability, 4)
    )


def disease_out(class_id: str) -> DiseaseInfoOut:
    info = CLASS_INFO[class_id]
    return DiseaseInfoOut(
        class_id=info.class_id, crop=info.crop, name=info.name, healthy=info.healthy,
        pathogen=info.pathogen, pathogen_type=info.pathogen_type, summary=info.summary,
        symptoms=list(info.symptoms), management=list(info.management),
    )


def image_result_out(prediction: ImagePrediction) -> ImageResultOut:
    return ImageResultOut(
        prediction=class_probability_out(prediction.best),
        top_predictions=[class_probability_out(p) for p in prediction.top],
        p_healthy=round(prediction.p_healthy, 4),
        predicted_crop=prediction.predicted_crop,
        crop_hint=prediction.crop_hint,
        low_confidence=prediction.low_confidence,
        crop_mismatch=prediction.crop_mismatch,
        disease=disease_out(prediction.best.class_id),
    )


def driver_out(driver: SoilDriver) -> SoilDriverOut:
    feature = FEATURES_BY_KEY[driver.feature]
    return SoilDriverOut(
        feature=driver.feature, label=feature.label, value=driver.value, unit=feature.unit,
        healthy_low=driver.healthy_low, healthy_high=driver.healthy_high,
        impact=driver.impact, message=driver.message,
    )


def soil_result_out(prediction: SoilPrediction) -> SoilResultOut:
    return SoilResultOut(
        status=prediction.status,
        probabilities={k: round(v, 4) for k, v in prediction.probabilities.items()},
        p_healthy=round(prediction.p_healthy, 4),
        provided_features=list(prediction.provided),
        informative=prediction.informative,
        drivers=[driver_out(d) for d in prediction.drivers],
        out_of_training_range=list(prediction.out_of_range),
    )


def build_evidence(
    image: ImagePrediction | None, soil: SoilPrediction | None, soil_prior: float
) -> tuple[Evidence, ...]:
    evidence = []
    if image is not None:
        evidence.append(image_evidence(image.p_healthy))
    if soil is not None:
        # Readings without moisture or nitrogen carry almost no signal: weight them out.
        weight = SOIL_WEIGHT if soil.informative else 0.0
        evidence.append(soil_evidence(soil.p_healthy, soil_prior, weight))
    return tuple(evidence)


def headline_for(fusion: FusionResult, image: ImagePrediction | None) -> str:
    disease = None
    if image is not None and not CLASS_INFO[image.best.class_id].healthy:
        info = CLASS_INFO[image.best.class_id]
        disease = f"{info.name.lower()} on {info.crop}"
    if fusion.status == "healthy":
        return "Your plant looks healthy"
    if fusion.status == "at_risk":
        if disease:
            return f"Possible {disease} - plant may be at risk"
        if image is not None and fusion.agreement == "conflict":
            return "Leaf looks healthy, but growing conditions put it at risk"
        return "Your plant may be at risk"
    return f"Signs of {disease} detected" if disease else "Your plant appears to be under stress"


def explanation_for(
    fusion: FusionResult, image: ImagePrediction | None, soil: SoilPrediction | None
) -> str:
    parts = []
    if image is not None:
        info = CLASS_INFO[image.best.class_id]
        label = "a healthy leaf" if info.healthy else f"{info.name.lower()}"
        parts.append(
            f"The leaf image most resembles {label} on {info.crop} "
            f"({image.best.probability:.0%} confidence)."
        )
    if soil is not None:
        parts.append(
            f"Soil and sensor readings indicate '{soil.status}' "
            f"({soil.probabilities[soil.status]:.0%} probability)."
        )
    if fusion.agreement == "agree":
        parts.append("Both sources point the same way, which strengthens the overall assessment.")
    elif fusion.agreement == "conflict":
        parts.append(
            "The two sources disagree, so the combined score is more cautious. A leaf can look "
            "healthy while growing conditions put it at risk, or be diseased despite good soil."
        )
    parts.append(f"Combined health score: {fusion.p_healthy:.0%}.")
    return " ".join(parts)


def soil_recommendations(soil: SoilPrediction) -> list[str]:
    return [soil_action(driver) for driver in soil.drivers]


def soil_action(driver: SoilDriver) -> str:
    """Concrete advice for a reading outside the healthy range, else the driver message."""
    actions = SOIL_ACTIONS.get(driver.feature)
    if actions is None:
        return driver.message
    if driver.value < driver.healthy_low:
        return actions[0]
    if driver.value > driver.healthy_high:
        return actions[1]
    return driver.message


def ph_recommendation(crop: str | None, readings: dict[str, float]) -> list[str]:
    ph = readings.get("soil_ph")
    if crop not in CROP_PH_RANGES or ph is None:
        return []
    low, high = CROP_PH_RANGES[crop]
    if ph < low:
        return [f"Soil pH {ph:g} is below the {low}-{high} range recommended for {crop}: "
                "consider liming, guided by a soil test."]
    if ph > high:
        return [f"Soil pH {ph:g} is above the {low}-{high} range recommended for {crop}: "
                "consider elemental sulphur or acidifying fertilisers, guided by a soil test."]
    return []


def recommendations_for(
    image: ImagePrediction | None, soil: SoilPrediction | None, readings: dict[str, float],
    status: str,
) -> list[str]:
    items: list[str] = []
    if image is not None and not CLASS_INFO[image.best.class_id].healthy:
        items.extend(CLASS_INFO[image.best.class_id].management)
        items.extend(favourable_conditions(image.best.class_id, readings))
    if soil is not None:
        items.extend(soil_recommendations(soil))
    crop = (image.crop_hint or image.predicted_crop) if image is not None else None
    items.extend(ph_recommendation(crop, readings))
    if not items and status == "healthy":
        items.append("No action needed right now - keep monitoring leaves and soil regularly.")
    elif not items:
        items.append(
            "Growing conditions look stressed but no single reading explains it: re-test the soil, "
            "check irrigation and fertility, and photograph a leaf for a more specific diagnosis."
        )
    return list(dict.fromkeys(items))


def warnings_for(image: ImagePrediction | None, soil: SoilPrediction | None) -> list[str]:
    warnings = []
    if image is not None and image.low_confidence:
        warnings.append(
            "The image model is not confident. Photograph a single leaf in good, even light, "
            "filling the frame. Only pepper, potato and tomato leaves are supported."
        )
    if image is not None and image.crop_mismatch:
        warnings.append(
            f"You selected {image.crop_hint}, but this photo looks more like a {image.predicted_crop} leaf. "
            "The diagnosis is limited to the crop you chose - check the selection."
        )
    if soil is not None and not soil.informative:
        warnings.append(
            "Without soil moisture or nitrogen readings the soil model has little to go on, "
            "so the soil result was not used in the combined score."
        )
    if soil is not None and soil.out_of_range:
        names = ", ".join(FEATURES_BY_KEY[k].label.lower() for k in soil.out_of_range)
        warnings.append(
            f"Some readings ({names}) are outside the range the soil model was trained on; "
            "treat the soil result as an extrapolation."
        )
    return warnings


def assess(
    image: ImagePrediction | None,
    soil: SoilPrediction | None,
    readings: dict[str, float],
    soil_prior: float,
) -> AssessmentOut:
    evidence = build_evidence(image, soil, soil_prior)
    fusion = fuse(evidence)
    return AssessmentOut(
        assessment=FusionOut(
            health_score=round(fusion.p_healthy * 100, 1),
            status=fusion.status,
            agreement=fusion.agreement,
            sources=[e.source for e in evidence],
            contributions={k: round(v, 3) for k, v in fusion.contributions.items()},
            headline=headline_for(fusion, image),
            explanation=explanation_for(fusion, image, soil),
        ),
        image=image_result_out(image) if image is not None else None,
        soil=soil_result_out(soil) if soil is not None else None,
        recommendations=recommendations_for(image, soil, readings, fusion.status),
        warnings=warnings_for(image, soil),
        disclaimer=DISCLAIMER,
    )
