from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    trainable_parameters: int


@dataclass(slots=True)
class CompiledExp301Arm:
    arm_id: Exp301ArmId
    model: nn.Module
    receipt: Exp301ArmReceipt


class V017FixedDepthLM(nn.Module):
    """Untied fixed-depth rival with tied token embedding/output weights."""

    def __init__(
        self,
        *,
        geometry: V017Geometry,
        capacity_bottleneck: int,
        capacity_tail_parameters: int,
        device=None,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.backbone = V017CoreBackbone(geometry, device=device)
        self.capacity_exchange = ActiveCapacityExchange(
            d_model=geometry.d_model,
            bottleneck=capacity_bottleneck,
            tail_parameters=capacity_tail_parameters,
            device=device,
        )

    @property
    def output_weight(self) -> torch.nn.Parameter:
        return self.backbone.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.backbone.embed(tokens)
        x = self.backbone.forward_block(x)
        x = self.capacity_exchange(x)
        x = self.backbone.finalize(x)
        return F.linear(x, self.output_weight)


# Frozen before challenge materialization. Each point is an independently
# parameter-compiled, untied fixed-depth Transformer with exactly 10M active
# trainable parameters. Depth is 3 * recurrent loop budget so the strongest
# fixed rival can spend comparable sequential block applications without
# receiving weight tying for free.
_FIXED_FRONTIER_GEOMETRY: dict[int, tuple[int, int, int, int, int]] = {
    # loop_budget: (depth, d_model, d_ff, active_capacity_bottleneck, tail)
    1: (3, 448, 1_344, 117, 192),
    2: (6, 320, 1_024, 258, 320),
    4: (12, 192, 768, 5_293, 64),
    8: (24, 192, 384, 673, 64),
    12: (36, 128, 448, 3_314, 0),
    16: (48, 128, 320, 1_382, 0),
}


def _receipt_for_fixed(
    *,
    loop_budget: int,
    geometry: V017Geometry,
    capacity_bottleneck: int,
    capacity_tail_parameters: int,
    trainable_parameters: int,
) -> Exp301ArmReceipt:
    return Exp301ArmReceipt(
        arm_id=Exp301ArmId.A_FIXED,
        loop_budget=loop_budget,
        depth=geometry.shared_layers,
        d_model=geometry.d_model,
        n_heads=geometry.n_heads,
        head_dim=geometry.head_dim,
        d_ff=geometry.d_ff,
        capacity_bottleneck=capacity_bottleneck,
        capacity_tail_parameters=capacity_tail_parameters,
        weight_tied_across_depth=False,
        loop_conditioning=False,
        trainable_parameters=trainable_parameters,
    )


def compile_fixed_frontier_point(*, loop_budget: int, device=None) -> CompiledExp301Arm:
    try:
        depth, d_model, d_ff, bottleneck, tail = _FIXED_FRONTIER_GEOMETRY[loop_budget]
    except KeyError as exc:
        raise ValueError(
            f"loop_budget must be one of {tuple(_FIXED_FRONTIER_GEOMETRY)}"
        ) from exc

    geometry = V017Geometry(
        vocab_size=4_608,
        d_model=d_model,
        n_heads=d_model // 64,
        head_dim=64,
        shared_layers=depth,
        d_ff=d_ff,
        max_loops=16,
    )
    model = V017FixedDepthLM(
        geometry=geometry,
        capacity_bottleneck=bottleneck,
        capacity_tail_parameters=tail,
        device=device,
    )
    parameter_count = count_trainable_parameters(model)
    if parameter_count != 10_000_000:
        raise RuntimeError(
            f"fixed frontier parameter drift at loop budget {loop_budget}: "
            f"{parameter_count:,}"
        )

    return CompiledExp301Arm(
        arm_id=Exp301ArmId.A_FIXED,
        model=model,
        receipt=_receipt_for_fixed(
            loop_budget=loop_budget,
            geometry=geometry,
            capacity_bottleneck=bottleneck,
            capacity_tail_parameters=tail,
            trainable_parameters=parameter_count,
        ),
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
