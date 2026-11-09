"""Leaf-image disease classifier backed by an ONNX MobileNetV3 model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.services.labels import crop_of, is_healthy_class
from app.services.preprocess import image_to_tensor

TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class ClassProbability:
    class_id: str
    probability: float


@dataclass(frozen=True)
class ImagePrediction:
    top: tuple[ClassProbability, ...]
    p_healthy: float
    predicted_crop: str
    crop_hint: str | None
    low_confidence: bool
    crop_mismatch: bool  # the photo looks like a different crop than the hint

    @property
    def best(self) -> ClassProbability:
        return self.top[0]


class ImageClassifier:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.metadata = json.loads(metadata_path.read_text())
        self.class_ids: tuple[str, ...] = tuple(self.metadata["class_ids"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def crops(self) -> tuple[str, ...]:
        return tuple(sorted({crop_of(c) for c in self.class_ids}))

    def probabilities(self, image: Image.Image) -> np.ndarray:
        output = self._session.run(None, {self._input_name: image_to_tensor(image)})[0][0]
        return np.asarray(output, dtype=np.float64)

    def predict(self, image: Image.Image, crop_hint: str | None = None) -> ImagePrediction:
        """Classify a leaf. A known crop restricts and renormalises the candidates."""
        if crop_hint is not None and crop_hint not in self.crops:
            raise ValueError(f"Unknown crop {crop_hint!r}; expected one of {self.crops}.")
        raw = self.probabilities(image)
        crop_totals = {crop: 0.0 for crop in self.crops}
        for class_id, p in zip(self.class_ids, raw):
            crop_totals[crop_of(class_id)] += float(p)
        probs = raw
        if crop_hint is not None:
            mask = np.array([crop_of(c) == crop_hint for c in self.class_ids])
            masked = np.where(mask, raw, 0.0)
            probs = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()

        order = np.argsort(probs)[::-1]
        top = tuple(ClassProbability(self.class_ids[i], float(probs[i])) for i in order[:TOP_K])
        p_healthy = float(sum(p for c, p in zip(self.class_ids, probs) if is_healthy_class(c)))
        return ImagePrediction(
            top=top,
            p_healthy=p_healthy,
            predicted_crop=max(crop_totals, key=crop_totals.get),
            crop_hint=crop_hint,
            # Judge confidence on the unrestricted output so a crop hint cannot
            # manufacture certainty for an unfamiliar photo.
            low_confidence=float(raw.max()) < LOW_CONFIDENCE_THRESHOLD
            or top[0].probability < LOW_CONFIDENCE_THRESHOLD,
            crop_mismatch=crop_hint is not None and crop_totals[crop_hint] < 0.5,
        )
