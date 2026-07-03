from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .dataset import SUPPORTED_EXTENSIONS


def scan_yolo_dataset(root: str | Path, images_dir: str = "images", labels_dir: str = "labels") -> List[Tuple[Path, Path]]:
    dataset_root = Path(root)
    image_root = dataset_root / images_dir
    label_root = dataset_root / labels_dir
    if not image_root.exists():
        raise FileNotFoundError(f"Images directory not found: {image_root}")
    samples: List[Tuple[Path, Path]] = []
    for image_path in sorted(image_root.rglob("*")):
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rel = image_path.relative_to(image_root)
        label_path = (label_root / rel).with_suffix(".txt")
        samples.append((image_path.resolve(), label_path.resolve()))
    if not samples:
        raise ValueError(f"No images found in: {image_root}")
    return samples


def write_manifest(path: str | Path, samples: List[Tuple[Path, Path]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label_path"])
        writer.writeheader()
        for image_path, label_path in samples:
            writer.writerow({"image_path": str(image_path), "label_path": str(label_path)})


def make_train_val_test_splits(
    samples: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    items = samples[:]
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
