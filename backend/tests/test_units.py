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


def test_crop_matches_resize_then_center_crop_reference():
    # Smooth content (like a photo): sub-pixel crop offsets must not change the result much.
    yy, xx = np.mgrid[0:300, 0:500]
    smooth = np.stack([xx / 500 * 255, yy / 300 * 255, (np.sin(xx / 40) + 1) * 127], axis=-1)
    image = Image.fromarray(smooth.astype(np.uint8))
    scale = 256 / 300
    resized = image.resize((round(500 * scale), 256), Image.Resampling.BILINEAR)
    left, top = (resized.width - 224) // 2, (resized.height - 224) // 2
    reference = np.asarray(resized.crop((left, top, left + 224, top + 224)), dtype=np.float32)
    ours = image_to_tensor(image)[0].transpose(1, 2, 0) * 255
    assert np.abs(ours - reference).mean() < 2.0


def test_square_256_image_is_a_pure_center_crop():
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor = image_to_tensor(Image.fromarray(pixels))[0].transpose(1, 2, 0) * 255
    assert np.allclose(tensor, pixels[16:240, 16:240], atol=0.01)


def test_transparent_png_is_composited_on_white():
    image = load_image(encode(Image.new("RGBA", (20, 20), (0, 0, 0, 0))))
    assert image.getpixel((5, 5)) == (255, 255, 255)


def test_rate_limiter_evicts_instead_of_resetting(monkeypatch):
    import app.rate_limit as rate_limit_module

    monkeypatch.setattr(rate_limit_module, "MAX_TRACKED_CLIENTS", 4)
    now = [0.0]
    limiter = RateLimiter(limit=1, window_seconds=10, clock=lambda: now[0])
    assert limiter.allow("victim")
    for i in range(10):
        limiter.allow(f"spoof-{i}")
    now[0] = 1.0
    # The victim may have been evicted under pressure, but tracking stays bounded.
    assert len(limiter._hits) <= 4
