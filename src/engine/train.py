from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.decode import decode_predictions
from src.utils.metrics import detection_metrics


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    proximal_reference: Dict[str, torch.Tensor] | None = None,
    proximal_mu: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        images = batch["images"].to(device)
        targets = batch["targets"]
        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss, logs = criterion(pred, targets)
        if proximal_reference is not None and proximal_mu > 0:
            prox = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if param.requires_grad and name in proximal_reference:
                    prox = prox + torch.sum((param - proximal_reference[name].to(device)) ** 2)
            loss = loss + 0.5 * proximal_mu * prox
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        count += 1
    return {"loss": total_loss / max(count, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    anchors: int,
    num_classes: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    all_predictions = []
    all_targets = []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        images = batch["images"].to(device)
        targets = batch["targets"]
        pred = model(images)
        loss, _ = criterion(pred, targets)
        detections = decode_predictions(
            pred,
            anchors=anchors,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        all_predictions.extend([d.detach().cpu() for d in detections])
        all_targets.extend([t.detach().cpu() for t in targets])
        total_loss += float(loss.detach().cpu())
        count += 1
    metrics = detection_metrics(all_predictions, all_targets, iou_threshold=iou_threshold, num_classes=num_classes)
    metrics["loss"] = total_loss / max(count, 1)
    return metrics


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return True
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
