from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.losses.detection_loss import DetectionLoss
from src.models.decode import decode_predictions
from src.models.tiny_detector import TinyStudentBehaviorDetector


def main() -> None:
    model = TinyStudentBehaviorDetector(num_classes=20, anchors=3, dropout=0.10)
    x = torch.randn(2, 3, 320, 320)
    y = model(x)
    assert tuple(y.shape) == (2, 75, 40, 40), tuple(y.shape)
    targets = [torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32), torch.zeros((0, 5), dtype=torch.float32)]
    loss, logs = DetectionLoss()(y, targets)
    assert torch.isfinite(loss), logs
    detections = decode_predictions(y, confidence_threshold=0.99)
    assert len(detections) == 2
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
