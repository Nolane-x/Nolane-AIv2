from __future__ import annotations

import torch
from torch import nn


def finalize_region_budget(
    module: nn.Module,
    target_parameters: int,
    *,
    device: str | torch.device | None,
    frozen: bool,
) -> None:
    """Fill unused regional capacity explicitly while preserving exact accounting."""
    used = sum(parameter.numel() for parameter in module.parameters())
    reserve = target_parameters - used
    if reserve < 0:
        raise ValueError(
            f"target budget {target_parameters:,} is smaller than functional path {used:,}"
        )

    module.register_parameter(
        "capacity_reserve",
        nn.Parameter(torch.empty(reserve, device=device)),
    )
    device_type = torch.device(device).type if device is not None else "cpu"
    if reserve and device_type != "meta":
        nn.init.zeros_(module.capacity_reserve)

    actual = sum(parameter.numel() for parameter in module.parameters())
    if actual != target_parameters:
        raise RuntimeError(
            f"region count mismatch: expected {target_parameters:,}, got {actual:,}"
        )

    if frozen:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


class BudgetedResidualRegion(nn.Module):
    """Generic residual fallback plus an explicit capacity reserve."""

    def __init__(
        self,
        d_model: int,
        target_parameters: int,
        *,
        device: str | torch.device | None = None,
        frozen: bool = False,
        functional: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.target_parameters = target_parameters
        self.functional = functional

        if functional:
            self.norm = nn.LayerNorm(d_model, device=device)
            self.fc1 = nn.Linear(d_model, d_model, device=device)
            self.fc2 = nn.Linear(d_model, d_model, device=device)
            self.gate = nn.Parameter(torch.zeros((), device=device))
        else:
            self.norm = None
            self.fc1 = None
            self.fc2 = None
            self.register_parameter("gate", None)

        finalize_region_budget(
            self,
            target_parameters,
            device=device,
            frozen=frozen,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.functional:
            return x
        assert self.norm is not None and self.fc1 is not None and self.fc2 is not None
        residual = self.fc2(torch.nn.functional.silu(self.fc1(self.norm(x))))
        return x + torch.sigmoid(self.gate) * residual
