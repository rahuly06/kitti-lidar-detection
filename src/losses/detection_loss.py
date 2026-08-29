"""Losses for the center-based BEV detector."""

import torch
import torch.nn as nn


PREDICTION_KEYS = ("heatmap", "offset", "size", "rotation")
TARGET_KEYS = (*PREDICTION_KEYS, "regression_mask")


def focal_heatmap_loss(logits, target, alpha=2.0, beta=4.0):
    """Compute a CenterNet-style focal heatmap loss."""
    if logits.shape != target.shape:
        raise ValueError(f"heatmap shapes must match, got {logits.shape} and {target.shape}")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("heatmap tensors must have shape (B, 1, H, W)")
    if not torch.is_floating_point(logits) or not torch.is_floating_point(target):
        raise TypeError("heatmap tensors must be floating point")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError("heatmap tensors must be finite")
    if torch.any((target < 0) | (target > 1)):
        raise ValueError("heatmap targets must lie in [0, 1]")

    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1 - 1e-4)
    positive_mask = (target == 1).to(logits.dtype)
    negative_mask = (target < 1).to(logits.dtype)
    negative_weights = (1 - target) ** beta
    positive_loss = torch.log(pred) * ((1 - pred) ** alpha) * positive_mask
    negative_loss = torch.log(1 - pred) * (pred**alpha) * negative_weights * negative_mask
    num_positive = positive_mask.sum()
    if num_positive.item() > 0:
        return -(positive_loss.sum() + negative_loss.sum()) / num_positive
    return -negative_loss.sum()


class DetectionLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, offset_weight=1.0, size_weight=1.0, rotation_weight=1.0):
        super().__init__()
        weights = (heatmap_weight, offset_weight, size_weight, rotation_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.size_weight = size_weight
        self.rotation_weight = rotation_weight

    @staticmethod
    def masked_l1_loss(prediction, target, mask):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("regression prediction and target shapes must match (B, C, H, W)")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("regression_mask must have shape (B, 1, H, W)")
        if prediction.shape[0] != mask.shape[0] or prediction.shape[2:] != mask.shape[2:]:
            raise ValueError("regression mask dimensions must match predictions")
        if not all(torch.isfinite(tensor).all() for tensor in (prediction, target, mask)):
            raise ValueError("regression tensors must be finite")
        active_values = mask.sum() * prediction.shape[1]
        return (torch.abs(prediction - target) * mask).sum() / active_values.clamp(min=1.0)

    def forward(self, predictions, targets):
        missing_predictions = set(PREDICTION_KEYS).difference(predictions)
        missing_targets = set(TARGET_KEYS).difference(targets)
        if missing_predictions or missing_targets:
            raise KeyError(
                f"missing prediction keys {sorted(missing_predictions)}; "
                f"missing target keys {sorted(missing_targets)}"
            )
        heatmap_loss = focal_heatmap_loss(predictions["heatmap"], targets["heatmap"])
        mask = targets["regression_mask"]
        offset_loss = self.masked_l1_loss(predictions["offset"], targets["offset"], mask)
        size_loss = self.masked_l1_loss(predictions["size"], targets["size"], mask)
        rotation_loss = self.masked_l1_loss(predictions["rotation"], targets["rotation"], mask)
        total_loss = (
            self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.size_weight * size_loss
            + self.rotation_weight * rotation_loss
        )
        return {"total": total_loss, "heatmap": heatmap_loss, "offset": offset_loss, "size": size_loss, "rotation": rotation_loss}
