from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.v017_recursive import (
    ActiveCapacityExchange,
    V017CoreBackbone,
    V017Geometry,
    build_nrs_core_10m,
    build_simple_recurrent_10m,
    count_trainable_parameters,
)


class Exp301ArmId(str, Enum):
    A_FIXED = "A_FIXED"
    B_LOOP_SIMPLE = "B_LOOP_SIMPLE"
    C_NRS_CORE = "C_NRS_CORE"


@dataclass(frozen=True, slots=True)
class Exp301ArmReceipt:
    arm_id: Exp301ArmId
    loop_budget: int | None
    depth: int
    d_model: int
    n_heads: int
    head_dim: int
    d_ff: int
    capacity_bottleneck: int
    capacity_tail_parameters: int
    weight_tied_across_depth: bool
    loop_conditioning: bool
    stateless_restarts: bool
    latent_state_carry: bool
    trainable_parameters: int


@dataclass(slots=True)
class CompiledExp301Arm:
    arm_id: Exp301ArmId
    model: nn.Module
    receipt: Exp301ArmReceipt


def _restart_code(
    restart_index: int,
    *,
    d_model: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if restart_index < 0:
        raise ValueError("restart_index must be non-negative")
    dimensions = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    inverse_frequency = torch.exp(-math.log(10_000.0) * dimensions / d_model)
    angles = (restart_index + 1) * inverse_frequency
    code = torch.empty(d_model, device=device, dtype=torch.float32)
    code[0::2] = angles.sin()
    code[1::2] = angles.cos()
    return code.to(dtype=dtype)


class V017FixedRestartLM(nn.Module):
    """Fixed three-layer rival with repeated stateless compute restarts.

    Every restart begins again from the original token embedding. The same
    fixed-depth model is reused, but no hidden/latent state crosses restart
    boundaries. A parameter-free sinusoidal restart code provides a fair way
    for the fixed rival to exploit extra compute without adding resident
    parameters or becoming stateful recurrence.
    """

    def __init__(self, *, device=None) -> None:
        super().__init__()
        self.geometry = V017Geometry()
        self.backbone = V017CoreBackbone(self.geometry, device=device)
        self.capacity_exchange = ActiveCapacityExchange(
            d_model=self.geometry.d_model,
            bottleneck=981,
            tail_parameters=192,
            device=device,
        )

    @property
    def output_weight(self) -> torch.nn.Parameter:
        return self.backbone.token_embedding.weight

    def forward(self, tokens: torch.Tensor, *, restarts: int) -> torch.Tensor:
        if restarts not in (1, 2, 4, 8, 12, 16):
            raise ValueError("restarts must be one of (1, 2, 4, 8, 12, 16)")

        base = self.backbone.embed(tokens)
        restart_states: list[torch.Tensor] = []
        for restart_index in range(restarts):
            code = _restart_code(
                restart_index,
                d_model=self.geometry.d_model,
                device=base.device,
                dtype=base.dtype,
            )
            current = base + code.view(1, 1, -1)
            current = self.backbone.forward_block(current)
            current = self.capacity_exchange(current)
            restart_states.append(current)

        aggregated = torch.stack(restart_states, dim=0).mean(dim=0)
        aggregated = self.backbone.finalize(aggregated)
        return F.linear(aggregated, self.output_weight)


_FROZEN_EFFORT_MULTIPLIERS = (1, 2, 4, 8, 12, 16)


def _receipt_for_fixed(*, loop_budget: int, model: V017FixedRestartLM) -> Exp301ArmReceipt:
    geometry = model.geometry
    return Exp301ArmReceipt(
        arm_id=Exp301ArmId.A_FIXED,
        loop_budget=loop_budget,
        depth=geometry.shared_layers,
        d_model=geometry.d_model,
        n_heads=geometry.n_heads,
        head_dim=geometry.head_dim,
        d_ff=geometry.d_ff,
        capacity_bottleneck=981,
        capacity_tail_parameters=192,
        weight_tied_across_depth=False,
        loop_conditioning=False,
        stateless_restarts=True,
        latent_state_carry=False,
        trainable_parameters=count_trainable_parameters(model),
    )


def compile_fixed_frontier_point(*, loop_budget: int, device=None) -> CompiledExp301Arm:
    if loop_budget not in _FROZEN_EFFORT_MULTIPLIERS:
        raise ValueError(f"loop_budget must be one of {_FROZEN_EFFORT_MULTIPLIERS}")

    model = V017FixedRestartLM(device=device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count != 10_000_000:
        raise RuntimeError(
            f"fixed restart arm parameter drift at effort {loop_budget}: {parameter_count:,}"
        )

    return CompiledExp301Arm(
        arm_id=Exp301ArmId.A_FIXED,
        model=model,
        receipt=_receipt_for_fixed(loop_budget=loop_budget, model=model),
    )


def _recurrent_receipt(
    *,
    arm_id: Exp301ArmId,
    model: nn.Module,
    capacity_bottleneck: int,
    capacity_tail_parameters: int,
    loop_conditioning: bool,
) -> Exp301ArmReceipt:
    geometry = model.geometry
    return Exp301ArmReceipt(
        arm_id=arm_id,
        loop_budget=None,
        depth=geometry.shared_layers,
        d_model=geometry.d_model,
        n_heads=geometry.n_heads,
        head_dim=geometry.head_dim,
        d_ff=geometry.d_ff,
        capacity_bottleneck=capacity_bottleneck,
        capacity_tail_parameters=capacity_tail_parameters,
        weight_tied_across_depth=True,
        loop_conditioning=loop_conditioning,
        stateless_restarts=False,
        latent_state_carry=True,
        trainable_parameters=count_trainable_parameters(model),
    )


def build_simple_recurrent_arm(*, device=None) -> CompiledExp301Arm:
    model = build_simple_recurrent_10m(device=device)
    return CompiledExp301Arm(
        arm_id=Exp301ArmId.B_LOOP_SIMPLE,
        model=model,
        receipt=_recurrent_receipt(
            arm_id=Exp301ArmId.B_LOOP_SIMPLE,
            model=model,
            capacity_bottleneck=981,
            capacity_tail_parameters=192,
            loop_conditioning=False,
        ),
    )


def build_nrs_core_arm(*, device=None) -> CompiledExp301Arm:
    model = build_nrs_core_10m(device=device)
    return CompiledExp301Arm(
        arm_id=Exp301ArmId.C_NRS_CORE,
        model=model,
        receipt=_recurrent_receipt(
            arm_id=Exp301ArmId.C_NRS_CORE,
            model=model,
            capacity_bottleneck=973,
            capacity_tail_parameters=192,
            loop_conditioning=True,
        ),
    )
