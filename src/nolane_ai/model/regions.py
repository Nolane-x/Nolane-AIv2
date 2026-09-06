from __future__ import annotations

import torch
from torch import nn


class BudgetedResidualRegion(nn.Module):
    """A small functional residual path plus an explicit unallocated capacity reserve.

    The reserve is intentionally visible. It keeps parameter accounting faithful while
    avoiding the false claim that every V0.16 research subsystem has already earned a
    full neural implementation before Stage-A evidence exists.
    """

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

        used = sum(p.numel() for p in self.parameters())
        reserve = target_parameters - used
        if reserve < 0:
            raise ValueError(
                f"target budget {target_parameters:,} is smaller than functional path {used:,}"
            )
        self.capacity_reserve = nn.Parameter(torch.empty(reserve, device=device))
        if reserve and device != "meta":
            nn.init.zeros_(self.capacity_reserve)

        actual = sum(p.numel() for p in self.parameters())
        if actual != target_parameters:
            raise RuntimeError(f"region count mismatch: expected {target_parameters}, got {actual}")

        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.functional:
            return x
        assert self.norm is not None and self.fc1 is not None and self.fc2 is not None
        residual = self.fc2(torch.nn.functional.silu(self.fc1(self.norm(x))))
        return x + torch.sigmoid(self.gate) * residual
