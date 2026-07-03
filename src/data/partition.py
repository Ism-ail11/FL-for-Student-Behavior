from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .dataset import read_yolo_label
from .splits import write_manifest

Sample = Tuple[Path, Path]


def read_manifest(path: str | Path) -> List[Sample]:
    rows: List[Sample] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((Path(row["image_path"]), Path(row["label_path"])))
    return rows


def dominant_class(label_path: Path, num_classes: int) -> int:
    labels = read_yolo_label(label_path)
    if labels.numel() == 0:
        return num_classes
    ids = labels[:, 0].long().tolist()
    counts = Counter(ids)
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    return int(random.choice(candidates))


def iid_partition(samples: List[Sample], clients: int, seed: int) -> Dict[int, List[Sample]]:
    rng = random.Random(seed)
    items = samples[:]
    rng.shuffle(items)
    partitions = {k: [] for k in range(clients)}
    for idx, sample in enumerate(items):
        partitions[idx % clients].append(sample)
    return partitions


def dirichlet_noniid_partition(
    samples: List[Sample],
    clients: int,
    alpha: float,
    num_classes: int,
    seed: int,
) -> Dict[int, List[Sample]]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    by_class: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[dominant_class(sample[1], num_classes)].append(sample)
    partitions = {k: [] for k in range(clients)}
    for _, cls_samples in by_class.items():
        random.shuffle(cls_samples)
        props = rng.dirichlet(np.full(clients, alpha, dtype=np.float64))
        raw_counts = np.floor(props * len(cls_samples)).astype(int)
        while raw_counts.sum() < len(cls_samples):
            raw_counts[int(rng.integers(0, clients))] += 1
        start = 0
        for k, count in enumerate(raw_counts.tolist()):
            partitions[k].extend(cls_samples[start:start + count])
            start += count
    for k in range(clients):
        random.shuffle(partitions[k])
    return partitions


def write_client_manifests(base_dir: str | Path, partitions: Dict[int, List[Sample]]) -> List[Path]:
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for k, samples in partitions.items():
        path = output_dir / f"client_{k}.csv"
        write_manifest(path, samples)
        paths.append(path)
    return paths
