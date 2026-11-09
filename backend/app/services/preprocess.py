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
MAX_IMAGE_PIXELS = 40_000_000  # ~40 MP; guards against decompression bombs


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def load_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image, honouring EXIF orientation."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Image is too large (over 40 megapixels).")
            probe.load()
            oriented = ImageOps.exif_transpose(probe)
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file could not be read as an image.") from exc


def resize_and_center_crop(image: Image.Image) -> Image.Image:
    """Resize the short side to 256 px, then take the central 224x224 crop."""
    width, height = image.size
    scale = RESIZE_SHORT_SIDE / min(width, height)
    new_size = (max(CROP_SIZE, round(width * scale)), max(CROP_SIZE, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    left = (resized.width - CROP_SIZE) // 2
    top = (resized.height - CROP_SIZE) // 2
    return resized.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Convert an RGB image to a float32 NCHW tensor in [0, 1], shape (1, 3, 224, 224).

    ImageNet mean/std normalisation is baked into the exported ONNX graph.
    """
    cropped = resize_and_center_crop(image.convert("RGB"))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[np.newaxis, ...]
