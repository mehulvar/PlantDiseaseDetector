"""Agronomic reference information for every class the image model predicts.

Content summarises widely published extension guidance (e.g. university
extension services). It is decision support, not a substitute for a local
agronomist or a lab diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Condition:
    """An environmental condition that favours a disease, checked against readings."""

    feature: str
    operator: str  # ">=" or "<="
    threshold: float
    note: str


@dataclass(frozen=True)
class ClassInfo:
    class_id: str
    crop: str
    name: str
    healthy: bool
    pathogen: str
    pathogen_type: str
    summary: str
    symptoms: tuple[str, ...]
    management: tuple[str, ...]
    favoured_by: tuple[Condition, ...] = field(default_factory=tuple)


HIGH_HUMIDITY = Condition("humidity", ">=", 85, "High humidity favours this disease.")
WARM_WEATHER = Condition("ambient_temperature", ">=", 24, "Warm temperatures favour this disease.")
COOL_WEATHER = Condition("ambient_temperature", "<=", 22, "Cool temperatures favour this disease.")
HOT_WEATHER = Condition("ambient_temperature", ">=", 27, "Hot temperatures favour this pest.")
DRY_AIR = Condition("humidity", "<=", 50, "Dry air favours this pest.")
LOW_NITROGEN = Condition("nitrogen", "<=", 20, "Nitrogen-stressed plants are more susceptible.")

BACTERIAL_SPOT_MANAGEMENT = (
    "Use certified disease-free seed and transplants.",
    "Avoid overhead irrigation and working among wet plants.",
    "Apply copper-based bactericides preventively, ideally tank-mixed with mancozeb.",
    "Rotate away from peppers and tomatoes for 2-3 years and remove crop debris.",
)
EARLY_BLIGHT_MANAGEMENT = (
    "Remove and destroy infected lower leaves; mulch to stop soil splash.",
    "Keep plants well fed - nitrogen-stressed plants are more susceptible.",
    "Apply protectant fungicides such as chlorothalonil or mancozeb when lesions first appear.",
    "Rotate crops for at least 2 years and clear infected debris after harvest.",
)
LATE_BLIGHT_MANAGEMENT = (
    "Act quickly: late blight spreads rapidly and can destroy a field within days.",
    "Remove and bag or destroy infected plants; do not compost them.",
    "Apply protectant fungicides (e.g. chlorothalonil, mancozeb) before and during wet weather.",
    "Plant certified disease-free seed and eliminate volunteer plants and cull piles.",
)

CLASS_INFO: dict[str, ClassInfo] = {
    "pepper_bacterial_spot": ClassInfo(
        "pepper_bacterial_spot", "pepper", "Bacterial spot", False,
        "Xanthomonas spp.", "bacterium",
        "A bacterial disease that spots leaves and fruit and causes leaf drop.",
        ("Small, water-soaked spots that turn brown", "Yellowing around lesions", "Leaf drop and scabby fruit spots"),
        BACTERIAL_SPOT_MANAGEMENT, (WARM_WEATHER, HIGH_HUMIDITY),
    ),
    "pepper_healthy": ClassInfo(
        "pepper_healthy", "pepper", "Healthy", True, "None", "none",
        "No visible disease symptoms on this bell pepper leaf.",
        ("Uniform green colour", "No spots, lesions or curling"),
        ("Keep monitoring weekly, especially after warm wet weather.",),
    ),
    "potato_early_blight": ClassInfo(
        "potato_early_blight", "potato", "Early blight", False,
        "Alternaria solani", "fungus",
        "A common fungal leaf disease that starts on older leaves and reduces yield.",
        ("Brown lesions with concentric 'target' rings", "Yellowing around lesions", "Starts on lower, older leaves"),
        EARLY_BLIGHT_MANAGEMENT, (WARM_WEATHER, LOW_NITROGEN),
    ),
    "potato_late_blight": ClassInfo(
        "potato_late_blight", "potato", "Late blight", False,
        "Phytophthora infestans", "oomycete",
        "A fast-moving, highly destructive disease of potato foliage and tubers.",
        ("Large, dark, water-soaked lesions", "White fuzzy growth on leaf undersides in humid weather", "Rapid collapse of foliage"),
        LATE_BLIGHT_MANAGEMENT, (COOL_WEATHER, HIGH_HUMIDITY),
    ),
    "potato_healthy": ClassInfo(
        "potato_healthy", "potato", "Healthy", True, "None", "none",
        "No visible disease symptoms on this potato leaf.",
        ("Uniform green colour", "No lesions or wilting"),
        ("Scout regularly for blight during cool, humid weather.",),
    ),
    "tomato_bacterial_spot": ClassInfo(
        "tomato_bacterial_spot", "tomato", "Bacterial spot", False,
        "Xanthomonas spp.", "bacterium",
        "A bacterial disease causing leaf spots, defoliation and blemished fruit.",
        ("Small, dark, greasy-looking spots", "Spots may merge and leaves yellow", "Raised scabby spots on fruit"),
        BACTERIAL_SPOT_MANAGEMENT, (WARM_WEATHER, HIGH_HUMIDITY),
    ),
    "tomato_early_blight": ClassInfo(
        "tomato_early_blight", "tomato", "Early blight", False,
        "Alternaria spp.", "fungus",
        "A fungal disease that defoliates plants from the bottom up.",
        ("Brown spots with concentric rings", "Yellow halo around spots", "Lower leaves affected first"),
        EARLY_BLIGHT_MANAGEMENT, (WARM_WEATHER, LOW_NITROGEN),
    ),
    "tomato_late_blight": ClassInfo(
        "tomato_late_blight", "tomato", "Late blight", False,
        "Phytophthora infestans", "oomycete",
        "A devastating disease that can kill tomato plants within days in wet weather.",
        ("Large greasy grey-green to brown blotches", "White mould on leaf undersides", "Firm brown patches on fruit"),
        LATE_BLIGHT_MANAGEMENT, (COOL_WEATHER, HIGH_HUMIDITY),
    ),
    "tomato_leaf_mold": ClassInfo(
        "tomato_leaf_mold", "tomato", "Leaf mold", False,
        "Passalora fulva", "fungus",
        "A fungal disease of humid, poorly ventilated conditions, common in greenhouses.",
        ("Pale yellow patches on upper leaf surface", "Olive-green velvety mould underneath", "Leaves wither and drop"),
        (
            "Lower humidity: ventilate, space plants and avoid wetting leaves.",
            "Remove infected leaves and clean up debris.",
            "Grow resistant varieties where leaf mold recurs.",
            "Apply a labelled fungicide if the disease is spreading.",
        ),
        (HIGH_HUMIDITY,),
    ),
    "tomato_septoria_leaf_spot": ClassInfo(
        "tomato_septoria_leaf_spot", "tomato", "Septoria leaf spot", False,
        "Septoria lycopersici", "fungus",
        "A fungal leaf spot that causes heavy defoliation, starting low on the plant.",
        ("Many small round spots with dark borders and grey centres", "Tiny black dots inside spots", "Lower leaves yellow and fall"),
        (
            "Remove infected lower leaves and mulch to prevent soil splash.",
            "Water at the base of plants rather than overhead.",
            "Apply chlorothalonil or copper fungicides at first symptoms.",
            "Rotate crops and remove infected debris and nearby nightshade weeds.",
        ),
        (HIGH_HUMIDITY,),
    ),
    "tomato_spider_mites": ClassInfo(
        "tomato_spider_mites", "tomato", "Two-spotted spider mite", False,
        "Tetranychus urticae", "pest (mite)",
        "Tiny sap-feeding mites that thrive in hot, dry conditions.",
        ("Fine yellow or white stippling on leaves", "Fine webbing on leaf undersides", "Bronzed, drying leaves"),
        (
            "Spray leaf undersides with water to knock mites off.",
            "Use insecticidal soap or horticultural oil, covering leaf undersides.",
            "Release or conserve predatory mites; avoid broad-spectrum insecticides.",
            "Keep plants well watered - drought-stressed plants suffer more.",
        ),
        (HOT_WEATHER, DRY_AIR),
    ),
    "tomato_target_spot": ClassInfo(
        "tomato_target_spot", "tomato", "Target spot", False,
        "Corynespora cassiicola", "fungus",
        "A fungal disease of warm, humid climates that affects leaves, stems and fruit.",
        ("Brown lesions with concentric rings and light centres", "Lesions can merge and leaves drop", "Sunken spots on fruit"),
        (
            "Improve airflow by pruning and staking plants.",
            "Remove infected leaves and crop debris.",
            "Apply labelled fungicides (e.g. chlorothalonil) when disease is active.",
            "Rotate away from tomatoes and other hosts.",
        ),
        (WARM_WEATHER, HIGH_HUMIDITY),
    ),
    "tomato_yellow_leaf_curl_virus": ClassInfo(
        "tomato_yellow_leaf_curl_virus", "tomato", "Yellow leaf curl virus", False,
        "Tomato yellow leaf curl virus (TYLCV)", "virus",
        "A whitefly-transmitted virus that stunts plants and severely cuts yield.",
        ("Leaves curl upward and cup", "Yellow leaf margins", "Stunted growth and flower drop"),
        (
            "There is no cure: remove and destroy infected plants promptly.",
            "Control whiteflies with insect netting, reflective mulch or labelled insecticides.",
            "Plant TYLCV-resistant varieties.",
            "Keep fields free of weeds that host whiteflies.",
        ),
        (HOT_WEATHER,),
    ),
    "tomato_mosaic_virus": ClassInfo(
        "tomato_mosaic_virus", "tomato", "Mosaic virus", False,
        "Tomato mosaic virus (ToMV)", "virus",
        "A very stable virus spread by contact, tools, hands and seed.",
        ("Light and dark green mosaic mottling", "Distorted, fern-like leaves", "Stunting and uneven fruit ripening"),
        (
            "There is no cure: remove and destroy infected plants.",
            "Disinfect tools and wash hands between plants.",
            "Use certified virus-free seed and resistant varieties.",
            "Avoid handling plants after using tobacco products.",
        ),
    ),
    "tomato_healthy": ClassInfo(
        "tomato_healthy", "tomato", "Healthy", True, "None", "none",
        "No visible disease symptoms on this tomato leaf.",
        ("Uniform green colour", "No spots, mould or curling"),
        ("Keep monitoring, especially lower leaves during warm, wet spells.",),
    ),
}

# Commonly recommended soil pH ranges by crop.
CROP_PH_RANGES: dict[str, tuple[float, float]] = {
    "tomato": (6.2, 6.8),
    "pepper": (6.0, 6.8),
    "potato": (5.0, 6.5),
}


def condition_matches(condition: Condition, value: float) -> bool:
    if condition.operator == ">=":
        return value >= condition.threshold
    if condition.operator == "<=":
        return value <= condition.threshold
    raise ValueError(f"Unknown operator {condition.operator!r}")


def favourable_conditions(class_id: str, readings: dict[str, float]) -> list[str]:
    """Notes for every disease-favouring condition present in the readings."""
    info = CLASS_INFO[class_id]
    notes = []
    for condition in info.favoured_by:
        value = readings.get(condition.feature)
        if value is not None and condition_matches(condition, value):
            notes.append(f"{condition.note} ({condition.feature.replace('_', ' ')}: {value:g})")
    return notes
