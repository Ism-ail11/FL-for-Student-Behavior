from __future__ import annotations

from typing import List, Tuple

import torch

from src.utils.boxes import clip_boxes, nms, xywh_to_xyxy


def reshape_predictions(pred: torch.Tensor, anchors: int, num_classes: int) -> torch.Tensor:
    b, _, h, w = pred.shape
    pred = pred.view(b, anchors, 5 + num_classes, h, w)
    return pred.permute(0, 1, 3, 4, 2).contiguous()


def decode_predictions(
    pred: torch.Tensor,
    anchors: int = 3,
    num_classes: int = 20,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    max_detections: int = 100,
) -> List[torch.Tensor]:
    """Return detections per image as [x1, y1, x2, y2, score, class_id], normalized."""
    p = reshape_predictions(pred, anchors, num_classes)
    bsz, a, h, w, _ = p.shape
    device = p.device
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    grid_x = grid_x.view(1, 1, h, w).float()
    grid_y = grid_y.view(1, 1, h, w).float()

    xy = torch.sigmoid(p[..., 0:2])
    wh = torch.sigmoid(p[..., 2:4])
    obj = torch.sigmoid(p[..., 4])
    cls = torch.sigmoid(p[..., 5:])
    scores, labels = (obj.unsqueeze(-1) * cls).max(dim=-1)

    cx = (grid_x + xy[..., 0]) / float(w)
    cy = (grid_y + xy[..., 1]) / float(h)
    boxes_xywh = torch.stack([cx, cy, wh[..., 0], wh[..., 1]], dim=-1).reshape(bsz, -1, 4)
    scores = scores.reshape(bsz, -1)
    labels = labels.reshape(bsz, -1)
    boxes = clip_boxes(xywh_to_xyxy(boxes_xywh.reshape(-1, 4))).reshape(bsz, -1, 4)

    outputs: List[torch.Tensor] = []
    for i in range(bsz):
        mask = scores[i] >= confidence_threshold
        if not mask.any():
            outputs.append(torch.zeros((0, 6), dtype=torch.float32, device=device))
            continue
        b = boxes[i][mask]
        s = scores[i][mask]
        c = labels[i][mask]
        keep_all = []
        for cls_id in c.unique():
            cls_mask = c == cls_id
            keep = nms(b[cls_mask], s[cls_mask], iou_threshold)
            original_idx = torch.nonzero(cls_mask, as_tuple=False).flatten()[keep]
            keep_all.append(original_idx)
        keep_idx = torch.cat(keep_all) if keep_all else torch.empty((0,), dtype=torch.long, device=device)
        keep_idx = keep_idx[s[keep_idx].argsort(descending=True)[:max_detections]]
        outputs.append(torch.cat([b[keep_idx], s[keep_idx, None], c[keep_idx, None].float()], dim=1))
    return outputs
