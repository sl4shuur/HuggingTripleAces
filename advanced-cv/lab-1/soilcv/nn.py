"""PyTorch curve head, official EMD loss, and explicit model policies."""

from contextlib import contextmanager
import math
import torch
from torch import nn
from .curves import LOG_WIDTHS


class MassCurveHead(nn.Module):
    """11 mass bins -> cumulative percentages. Optional linear feature projection."""

    def __init__(self, in_features=None):
        super().__init__()
        self.projection = (
            nn.Identity() if in_features is None else nn.Linear(in_features, 11)
        )

    def forward(self, features):
        logits = self.projection(features)
        if logits.ndim != 2 or logits.shape[1] != 11:
            raise ValueError("The curve head needs [batch, 11] logits.")
        curve = (100 * logits.float().softmax(dim=1).cumsum(dim=1)).clamp(max=100)
        return torch.cat([curve[:, :-1], torch.full_like(curve[:, -1:], 100)], dim=1)


class EMDLoss(nn.Module):
    """Same percent/log10 units as NumPy EMD. Scaling is explicit."""

    def __init__(self, reduction="mean", scale=1.0):
        super().__init__()
        if (
            reduction not in {"none", "mean", "sum"}
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Set a valid reduction and positive loss scale.")
        self.reduction, self.scale = reduction, scale
        self.register_buffer(
            "widths", torch.tensor(LOG_WIDTHS.copy(), dtype=torch.float32)
        )

    def forward(self, pred, true):
        if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 11:
            raise ValueError("Loss inputs must have the same [batch, 11] shape.")
        loss = ((pred.float() - true.float()).abs()[:, :-1] * self.widths).sum(
            dim=1
        ) * self.scale
        if self.reduction == "none":
            return loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


def set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
    return module


def freeze_bn_stats(module):
    """Call after model.train(); leaves affine parameter gradients unchanged."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
    return module


def make_param_groups(
    model, *, lr, weight_decay=0.0, layer_ids=None, layer_decay=1.0, no_weight_decay=()
):
    """No decay on bias/1D parameters. Layer IDs must be supplied for LLRD."""
    if lr <= 0 or weight_decay < 0 or not 0 < layer_decay <= 1:
        raise ValueError("Set valid learning rate, decay, and layer decay.")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("The model has no trainable parameters.")
    if layer_decay != 1 and layer_ids is None:
        raise ValueError(
            "Provide explicit parameter-name layer_ids for layer-wise LR decay."
        )
    if layer_ids is not None and any(name not in layer_ids for name, _ in named):
        raise ValueError("Each trainable parameter needs a layer ID.")
    if layer_ids is not None and any(
        not isinstance(layer_ids[name], int) or layer_ids[name] < 0 for name, _ in named
    ):
        raise ValueError("Layer IDs must be non-negative integers.")
    if set(no_weight_decay) - set(dict(model.named_parameters())):
        raise ValueError("Unknown no_weight_decay parameter names.")
    max_layer = max(layer_ids[name] for name, _ in named) if layer_ids else 0
    groups = {}
    for name, parameter in named:
        layer = layer_ids[name] if layer_ids else 0
        decay = (
            0.0
            if parameter.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay
            else weight_decay
        )
        learning_rate = lr * layer_decay ** (max_layer - layer)
        groups.setdefault((learning_rate, decay), []).append(parameter)
    return [
        {"params": params, "lr": rate, "weight_decay": decay}
        for (rate, decay), params in groups.items()
    ]


@contextmanager
def evaluation_mode(model):
    """Restore every module's mode, including deliberately frozen BN layers."""
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield model
    finally:
        for module, training in modes.items():
            module.training = training
