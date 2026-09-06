from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn


def functional_trainable_named_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield only trainable parameters that participate in functional computation.

    Regional ``capacity_reserve`` tensors exist only to preserve exact parameter
    accounting. They are deliberately excluded from optimizer parameter groups.
    """
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("capacity_reserve") or ".capacity_reserve" in name:
            continue
        yield name, parameter


def build_functional_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    if lr <= 0:
        raise ValueError("lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    parameters = [parameter for _, parameter in functional_trainable_named_parameters(model)]
    if not parameters:
        raise ValueError("model exposes no functional trainable parameters")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
