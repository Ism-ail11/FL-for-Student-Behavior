from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_label(path: Path) -> torch.Tensor:
    if not path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            if w <= 0 or h <= 0:
                continue
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            rows.append([float(cls), x, y, w, h])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)



def _xywh_to_corners(labels: torch.Tensor) -> np.ndarray:
    if labels.numel() == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    arr = labels.detach().cpu().numpy().astype(np.float32)
    x, y, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )
    return corners


def _corners_to_xywh(corners: np.ndarray, class_ids: np.ndarray) -> torch.Tensor:
    if corners.shape[0] == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    x1 = np.clip(corners[:, :, 0].min(axis=1), 0.0, 1.0)
    y1 = np.clip(corners[:, :, 1].min(axis=1), 0.0, 1.0)
    x2 = np.clip(corners[:, :, 0].max(axis=1), 0.0, 1.0)
    y2 = np.clip(corners[:, :, 1].max(axis=1), 0.0, 1.0)
    w = x2 - x1
    h = y2 - y1
    keep = (w > 1e-4) & (h > 1e-4)
    if not np.any(keep):
        return torch.zeros((0, 5), dtype=torch.float32)
    x = x1[keep] + w[keep] / 2.0
    y = y1[keep] + h[keep] / 2.0
    out = np.stack([class_ids[keep], x, y, w[keep], h[keep]], axis=1)
    return torch.tensor(out, dtype=torch.float32)


def _apply_affine_to_labels(labels: torch.Tensor, matrix: np.ndarray) -> torch.Tensor:
    if labels.numel() == 0:
        return labels
    class_ids = labels[:, 0].detach().cpu().numpy().astype(np.float32)
    corners = _xywh_to_corners(labels)
    ones = np.ones((corners.shape[0], 4, 1), dtype=np.float32)
    homogeneous = np.concatenate([corners, ones], axis=2)
    transformed = homogeneous @ matrix.T
    transformed = transformed[:, :, :2]
    return _corners_to_xywh(transformed, class_ids)


def random_augment(image: Image.Image, labels: torch.Tensor, image_size: int) -> Tuple[Image.Image, torch.Tensor]:
    labels = labels.clone()
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    if labels.numel() > 0 and random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        labels[:, 1] = 1.0 - labels[:, 1]

    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(factor)

    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Contrast(image).enhance(factor)

    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        tx = random.uniform(-0.10, 0.10)
        ty = random.uniform(-0.10, 0.10)
        angle = random.uniform(-10.0, 10.0)
        theta = np.deg2rad(angle)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        cx = cy = 0.5
        matrix = np.array(
            [
                [scale * cos_t, -scale * sin_t, cx + tx - scale * cos_t * cx + scale * sin_t * cy],
                [scale * sin_t, scale * cos_t, cy + ty - scale * sin_t * cx - scale * cos_t * cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        inverse = np.linalg.inv(matrix)
        pil_matrix = (
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[0, 2] * image_size),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(inverse[1, 2] * image_size),
        )
        image = image.transform((image_size, image_size), Image.Transform.AFFINE, pil_matrix, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        labels = _apply_affine_to_labels(labels, matrix)

    return image, labels


class StudentBehaviorDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 320,
        augment: bool = False,
        num_classes: int = 20,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        self.samples = self._read_manifest(self.manifest_path)
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {self.manifest_path}")

    @staticmethod
    def _read_manifest(path: Path) -> List[Tuple[Path, Path]]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        samples: List[Tuple[Path, Path]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = Path(row["image_path"])
                label_path = Path(row["label_path"])
                samples.append((image_path, label_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        labels = read_yolo_label(label_path)
        valid = (labels[:, 0] >= 0) & (labels[:, 0] < self.num_classes) if labels.numel() else torch.zeros(0, dtype=torch.bool)
        labels = labels[valid] if labels.numel() else labels
        if self.augment:
            image, labels = random_augment(image, labels, self.image_size)
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return {
            "image": image_to_tensor(image),
            "targets": labels,
            "image_path": str(image_path),
        }


def detection_collate(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[torch.Tensor] | List[str]]:
    images = torch.stack([item["image"] for item in batch])  # type: ignore[index]
    targets = [item["targets"] for item in batch]  # type: ignore[index]
    paths = [str(item["image_path"]) for item in batch]
    return {"images": images, "targets": targets, "image_paths": paths}
