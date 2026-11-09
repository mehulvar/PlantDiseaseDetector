import io

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.rate_limit import RateLimiter
from app.schemas import SoilReading
from app.services.knowledge import CLASS_INFO, favourable_conditions
from app.services.labels import FOLDER_TO_CLASS_ID, crop_of, is_healthy_class
from app.services.preprocess import CROP_SIZE, InvalidImageError, image_to_tensor, load_image

pytestmark = pytest.mark.unit


def encode(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_image_to_tensor_shape_and_range_for_non_square_image():
    tensor = image_to_tensor(Image.new("RGB", (640, 300), (255, 128, 0)))
    assert tensor.shape == (1, 3, CROP_SIZE, CROP_SIZE)
    assert tensor.dtype == np.float32
    assert tensor[0, 0].max() == pytest.approx(1.0)
    assert tensor[0, 2].max() == pytest.approx(0.0)


def test_load_image_converts_rgba_and_grayscale_to_rgb():
    assert load_image(encode(Image.new("RGBA", (50, 50)))).mode == "RGB"
    assert load_image(encode(Image.new("L", (50, 50)))).mode == "RGB"


def test_load_image_applies_exif_orientation():
    image = Image.new("RGB", (100, 40))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    loaded = load_image(encode(image, "JPEG", exif=exif.tobytes()))
    assert loaded.size == (40, 100)


@pytest.mark.parametrize("payload", [b"", b"not an image at all"])
def test_load_image_rejects_invalid_bytes(payload):
    with pytest.raises(InvalidImageError):
        load_image(payload)


def test_soil_reading_requires_at_least_one_value():
    with pytest.raises(ValidationError):
        SoilReading()


def test_soil_reading_rejects_out_of_bounds_and_unknown_fields():
    with pytest.raises(ValidationError):
        SoilReading(soil_ph=15)
    with pytest.raises(ValidationError):
        SoilReading(soil_ph=6.5, magic=1)


def test_soil_reading_provided_drops_missing_values():
    assert SoilReading(soil_ph=6.5, nitrogen=30).provided() == {"soil_ph": 6.5, "nitrogen": 30}


def test_rate_limiter_blocks_after_limit_and_recovers():
    now = [0.0]
    limiter = RateLimiter(limit=2, window_seconds=10, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 10.5
    assert limiter.allow("a")


def test_rate_limiter_validates_arguments():
    with pytest.raises(ValueError):
        RateLimiter(limit=0, window_seconds=1)


def test_every_class_has_knowledge_and_consistent_crop():
    for class_id in FOLDER_TO_CLASS_ID.values():
        info = CLASS_INFO[class_id]
        assert info.crop == crop_of(class_id)
        assert info.healthy == is_healthy_class(class_id)
        assert info.management


def test_favourable_conditions_reports_matching_readings_only():
    notes = favourable_conditions("tomato_late_blight", {"humidity": 92, "ambient_temperature": 30})
    assert len(notes) == 1 and "humidity" in notes[0]
    assert favourable_conditions("tomato_late_blight", {}) == []
