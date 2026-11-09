"""Image preprocessing shared by training-time evaluation and the serving API.

Keeping a single implementation guarantees that the accuracy measured offline
is the accuracy users actually get: the exported ONNX model expects exactly the
tensor produced by :func:`image_to_tensor`.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
MAX_IMAGE_PIXELS = 24_000_000  # guards against decompression bombs / memory exhaustion
MAX_ASPECT_RATIO = 10.0
JPEG_DRAFT_SIZE = (1024, 1024)  # decode large JPEGs at reduced scale (we only need 224 px)
# The JPEG decoder also opens multi-picture (MPO) files produced by some phones.
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be used as a leaf photo."""


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite transparent images onto white instead of black."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16/32-bit greyscale: rescale to 8-bit rather than saturating.
        array = np.asarray(image, dtype=np.float64)
        span = array.max() - array.min() or 1.0
        return Image.fromarray(((array - array.min()) / span * 255).astype(np.uint8)).convert("RGB")
    return image.convert("RGB")


def load_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image, honouring EXIF orientation."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as probe:
            width, height = probe.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Image is too large (over 24 megapixels).")
            if max(width, height) / max(1, min(width, height)) > MAX_ASPECT_RATIO:
                raise InvalidImageError("Image is too long and thin - please upload a normal photo of a leaf.")
            if probe.format in ("JPEG", "MPO"):
                probe.draft("RGB", JPEG_DRAFT_SIZE)
            probe.load()
            return _flatten_alpha(ImageOps.exif_transpose(probe))
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file could not be read as an image (use JPEG, PNG or WebP).") from exc


def center_crop_box(width: int, height: int) -> tuple[float, float, float, float]:
    """The source region that 'resize short side to 256, centre-crop 224' would keep."""
    side = min(width, height) * CROP_SIZE / RESIZE_SHORT_SIDE
    left = (width - side) / 2
    top = (height - side) / 2
    return (left, top, left + side, top + side)


def resize_and_center_crop(image: Image.Image) -> Image.Image:
    """Equivalent to resizing the short side to 256 px and taking the central 224x224.

    Only the cropped region is resampled, so memory use is bounded by the
    output size regardless of the input's dimensions or aspect ratio.
    """
    box = center_crop_box(*image.size)
    return image.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR, box=box)


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Convert an RGB image to a float32 NCHW tensor in [0, 1], shape (1, 3, 224, 224).

    ImageNet mean/std normalisation is baked into the exported ONNX graph.
    """
    cropped = resize_and_center_crop(image.convert("RGB"))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[np.newaxis, ...]
