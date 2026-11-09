"""Train the leaf-image disease classifier and export it to ONNX for serving.

Pipeline
--------
1. Index every readable image in the PlantVillage class folders (15 classes).
2. Stratified 80/10/10 train/val/test split (seeded, saved to ml/outputs/image_split.csv).
3. Fine-tune an ImageNet-pretrained MobileNetV3-Large with augmentation that
   mimics field photos (colour jitter, random crops, rotation, blur).
4. Fit a softmax temperature on the validation split (probability calibration).
5. Export to ONNX with ImageNet normalisation + temperature + softmax baked in.
6. Re-evaluate the *exported ONNX model* on the untouched test split using the
   exact preprocessing code the API uses, and write metrics to backend/models.

Usage:  python ml/train_image.py --data-dir "Image Data/PlantVillage"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.labels import FOLDER_TO_CLASS_ID  # noqa: E402
from app.services.preprocess import CROP_SIZE, image_to_tensor  # noqa: E402

LOGGER = logging.getLogger("train_image")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42
MODELS_DIR = ROOT / "backend" / "models"
OUTPUTS_DIR = ROOT / "ml" / "outputs"


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    workers: int
    evaluate_only: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def index_images(data_dir: Path) -> pd.DataFrame:
    """Return a DataFrame of (path, class_id) for every decodable image."""
    rows = []
    for folder, class_id in FOLDER_TO_CLASS_ID.items():
        class_dir = data_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:  # noqa: BLE001 - any decode failure means skip the file
                LOGGER.warning("Skipping unreadable image %s", path)
                continue
            rows.append({"path": str(path.relative_to(ROOT)), "class_id": class_id})
    return pd.DataFrame(rows)


def split_images(index: pd.DataFrame) -> pd.DataFrame:
    train, rest = train_test_split(index, test_size=0.2, stratify=index.class_id, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.class_id, random_state=SEED)
    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    ).reset_index(drop=True)


class LeafDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_ids: list[str], transform) -> None:
        self.paths = [ROOT / p for p in frame.path]
        lookup = {c: i for i, c in enumerate(class_ids)}
        self.labels = [lookup[c] for c in frame.class_id]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as img:
            return self.transform(img.convert("RGB")), self.labels[index]


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(CROP_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
        ]
    )


def eval_transform(image: Image.Image) -> torch.Tensor:
    """Identical to serving preprocessing (minus the batch dimension)."""
    return torch.from_numpy(image_to_tensor(image)[0])


class ServingModel(nn.Module):
    """Wraps the backbone with normalisation, temperature scaling and softmax."""

    def __init__(self, backbone: nn.Module, temperature: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x) / self.temperature, dim=1)


def build_backbone(num_classes: int) -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def collect_logits(model: ServingModel, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model.logits(images.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def train(config: TrainConfig, split: pd.DataFrame, class_ids: list[str]) -> ServingModel:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    LOGGER.info("Training on %s", device)
    loader_kwargs = {"num_workers": config.workers, "persistent_workers": config.workers > 0}
    train_ds = LeafDataset(split[split.split == "train"], class_ids, train_transform())
    val_ds = LeafDataset(split[split.split == "val"], class_ids, eval_transform)
    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, config.batch_size, **loader_kwargs)

    model = ServingModel(build_backbone(len(class_ids)), temperature=1.0).to(device)
    weights = class_weights(train_ds.labels, len(class_ids)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=config.epochs * len(train_loader)
    )

    best_acc, best_state = -1.0, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model.logits(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        logits, labels = collect_logits(model, val_loader, device)
        val_acc = (logits.argmax(1) == labels).float().mean().item()
        LOGGER.info(
            "epoch %d/%d loss=%.4f val_acc=%.4f (%.0fs)",
            epoch, config.epochs, running / len(train_loader), val_acc, time.time() - started,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logits, labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    LOGGER.info("Best val acc %.4f; fitted temperature %.3f", best_acc, temperature)
    model = model.cpu().eval()
    model.temperature.fill_(temperature)
    return model


def export_onnx(model: ServingModel, path: Path) -> None:
    dummy = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["image"], output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def evaluate_onnx(path: Path, test: pd.DataFrame, class_ids: list[str]) -> dict:
    """Score the exported model on the held-out test set with serving preprocessing."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probabilities = []
    for rel_path in test.path:
        with Image.open(ROOT / rel_path) as img:
            tensor = image_to_tensor(img.convert("RGB"))
        probabilities.append(session.run(None, {"image": tensor})[0][0])
    probs = np.stack(probabilities)
    y_true = test.class_id.map({c: i for i, c in enumerate(class_ids)}).to_numpy()
    y_pred = probs.argmax(1)
    np.save(OUTPUTS_DIR / "image_test_probabilities.npy", probs)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_ids))), target_names=class_ids,
        output_dict=True, zero_division=0,
    )
    return {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_size": int(len(test)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {c: {k: round(v, 4) for k, v in report[c].items()} for c in class_ids},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Image Data" / "PlantVillage")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Re-score the existing ONNX model on the saved test split (no training).")
    args = parser.parse_args()
    return TrainConfig(args.data_dir.resolve(), args.epochs, args.batch_size, args.lr, args.workers,
                       args.evaluate_only)


def reevaluate() -> None:
    """Re-score the exported model (e.g. after a preprocessing change) and refresh its metrics."""
    split = pd.read_csv(OUTPUTS_DIR / "image_split.csv")
    test = split[split.split == "test"].reset_index(drop=True)
    metadata_path = MODELS_DIR / "leaf_disease.json"
    metadata = json.loads(metadata_path.read_text())
    metrics = evaluate_onnx(MODELS_DIR / "leaf_disease.onnx", test, metadata["class_ids"])
    LOGGER.info("ONNX test accuracy %.4f, macro F1 %.4f", metrics["test_accuracy"], metrics["macro_f1"])
    metadata.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (OUTPUTS_DIR / "image_confusion_matrix.json").write_text(
        json.dumps({"class_ids": metadata["class_ids"], "matrix": metrics["confusion_matrix"]})
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = parse_args()
    if config.evaluate_only:
        reevaluate()
        return
    seed_everything(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    index = index_images(config.data_dir)
    split = split_images(index)
    split.to_csv(OUTPUTS_DIR / "image_split.csv", index=False)
    class_ids = sorted(FOLDER_TO_CLASS_ID.values())
    LOGGER.info("Indexed %d images: %s", len(index), split.split.value_counts().to_dict())

    model = train(config, split, class_ids)
    onnx_path = MODELS_DIR / "leaf_disease.onnx"
    export_onnx(model, onnx_path)

    test = split[split.split == "test"].reset_index(drop=True)
    metrics = evaluate_onnx(onnx_path, test, class_ids)
    LOGGER.info("ONNX test accuracy %.4f, macro F1 %.4f", metrics["test_accuracy"], metrics["macro_f1"])

    metadata = {
        "class_ids": class_ids,
        "architecture": "mobilenet_v3_large (ImageNet pretrained, fine-tuned)",
        "input": {"shape": [1, 3, CROP_SIZE, CROP_SIZE], "range": "[0, 1] RGB", "layout": "NCHW"},
        "temperature": float(model.temperature.item()),
        "epochs": config.epochs,
        "class_counts": split.class_id.value_counts().sort_index().to_dict(),
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
    }
    (MODELS_DIR / "leaf_disease.json").write_text(json.dumps(metadata, indent=2))
    (OUTPUTS_DIR / "image_confusion_matrix.json").write_text(
        json.dumps({"class_ids": class_ids, "matrix": metrics["confusion_matrix"]})
    )
    LOGGER.info("Saved model and metadata to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
