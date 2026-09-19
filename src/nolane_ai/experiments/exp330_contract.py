from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

EXPERIMENT_ID = "EXP-330"
SCHEMA_VERSION = "EXP330-P02-PARAMETER-GROUP-LOCALIZATION-V1"
FAMILY = "iterative-grid-and-maze"
WORLD_INDICES = (0, 2)
EXPOSURES_PER_WORLD = 32
EFFORT_CYCLE = (1, 2, 4, 8)
PROBE_ROUNDS = (1, 4, 8, 16, 24, 31)
CROSS_DAMAGE_EPS = 1e-6
SELF_IMPROVEMENT_EPS = 1e-6
PARENT_EVIDENCE_DIGEST = "719b0c35b8538ef379a912448d7539bbdf1ef6e856e036a986b7c64b7a648cd7"
PARENT_WORLD_DAMAGE_ROUNDS = (4, 8, 16, 24, 31)
PARENT_WORLDD2_DAMAGE_ROUNDS = (1, 4, 8, 16, 24, 31)
PARENT_FINAL_MODEL_DIGEST = "c128b999b58dd1ea852de77fc346a6632e61cd8dbb08192697f80cf8b53f8f71"
PARENT_FINAL_OPTIMIZER_DIGEST = "86a4afe342c5def87d2f518023e06e863a592c23830b3ecf73496e4a71a23a3f"
PARENT_FINAL_RNG_DIGEST = "e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07"

PARAMETER_GROUPS = (
    "embedding_output",
    "layer0",
    "layer1",
    "layer2",
    "final_norm",
    "capacity_exchange",
)

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}

# Filled after the preregistration JSON is frozen. Tests bind this constant to
# the canonical digest so source and protocol cannot silently diverge.
APPROVED_PREREGISTRATION_DIGEST = "9ca527b7647b46e162b7f7e965f4e4ac74dcf4b1a4b71d4af4ab792331db02bb"


@dataclass(frozen=True, slots=True)
class GroupProbeRecord:
    round_index: int
    effort: int
    parameter_group: str
    gradient_dot: float
    gradient_cosine: float
    world0_gradient_norm: float
    world2_gradient_norm: float
    world0_update_self_delta: float
    world0_update_cross_delta_on_world2: float
    world2_update_self_delta: float
    world2_update_cross_delta_on_world0: float
    world0_direct_damage: bool
    world2_direct_damage: bool
    nonfinite_events: int


def effort_for_round(round_index: int) -> int:
    if not isinstance(round_index, int) or not 0 <= round_index < EXPOSURES_PER_WORLD:
        raise ValueError("round index")
    return EFFORT_CYCLE[round_index % len(EFFORT_CYCLE)]


def parameter_group_for_name(name: str) -> str:
    if name == "backbone.token_embedding.weight":
        return "embedding_output"
    for index in range(3):
        if name.startswith(f"backbone.layers.{index}."):
            return f"layer{index}"
    if name.startswith("backbone.final_norm."):
        return "final_norm"
    if name.startswith("capacity_exchange."):
        return "capacity_exchange"
    raise ValueError(f"unregistered A_FIXED parameter: {name}")


def directed_damage(*, self_delta: float, cross_delta: float) -> bool:
    if not math.isfinite(self_delta) or not math.isfinite(cross_delta):
        raise ValueError("nonfinite directed deltas")
    return self_delta <= -SELF_IMPROVEMENT_EPS and cross_delta >= CROSS_DAMAGE_EPS


def validate_probe(record: GroupProbeRecord) -> None:
    if record.round_index not in PROBE_ROUNDS:
        raise ValueError("probe round")
    if record.effort != effort_for_round(record.round_index):
        raise ValueError("probe effort")
    if record.parameter_group not in PARAMETER_GROUPS:
        raise ValueError("parameter group")
    values = (
        record.gradient_dot,
        record.gradient_cosine,
        record.world0_gradient_norm,
        record.world2_gradient_norm,
        record.world0_update_self_delta,
        record.world0_update_cross_delta_on_world2,
        record.world2_update_self_delta,
        record.world2_update_cross_delta_on_world0,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite probe metric")
    if not -1.000001 <= record.gradient_cosine <= 1.000001:
        raise ValueError("gradient cosine range")
    if record.world0_gradient_norm < 0 or record.world2_gradient_norm < 0:
        raise ValueError("gradient norm range")
    if record.nonfinite_events != 0:
        raise ValueError("probe nonfinite event")
    if record.world0_direct_damage is not directed_damage(
        self_delta=record.world0_update_self_delta,
        cross_delta=record.world0_update_cross_delta_on_world2,
    ):
        raise ValueError("world0 damage mismatch")
    if record.world2_direct_damage is not directed_damage(
        self_delta=record.world2_update_self_delta,
        cross_delta=record.world2_update_cross_delta_on_world0,
    ):
        raise ValueError("world2 damage mismatch")


def expected_probe_keys() -> tuple[tuple[int, str], ...]:
    return tuple((round_index, group) for round_index in PROBE_ROUNDS for group in PARAMETER_GROUPS)


def reduce_group_localization(
    probes: Sequence[GroupProbeRecord],
    *,
    parent_reproduced: bool,
    invalid: bool = False,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if invalid:
        return "INVALID_PARAMETER_GROUP_COURT", (), (), ()
    rows = tuple(probes)
    try:
        if tuple((row.round_index, row.parameter_group) for row in rows) != expected_probe_keys():
            return "INVALID_PARAMETER_GROUP_COURT", (), (), ()
        for row in rows:
            validate_probe(row)
    except (TypeError, ValueError):
        return "INVALID_PARAMETER_GROUP_COURT", (), (), ()
    if not parent_reproduced:
        return "PARENT_REPRODUCTION_MISMATCH", (), (), ()

    world0_groups = tuple(
        group for group in PARAMETER_GROUPS
        if any(row.parameter_group == group and row.world0_direct_damage for row in rows)
    )
    world2_groups = tuple(
        group for group in PARAMETER_GROUPS
        if any(row.parameter_group == group and row.world2_direct_damage for row in rows)
    )
    active_groups = tuple(group for group in PARAMETER_GROUPS if group in set(world0_groups) | set(world2_groups))
    bidirectional = tuple(group for group in PARAMETER_GROUPS if group in world0_groups and group in world2_groups)

    if not active_groups:
        decision = "NO_GROUP_LOCAL_CROSS_DAMAGE"
    elif len(active_groups) == 1 and len(bidirectional) == 1:
        decision = "SINGLE_GROUP_BIDIRECTIONAL_CROSS_DAMAGE"
    elif bidirectional:
        decision = "MULTI_GROUP_CROSS_DAMAGE"
    elif world0_groups and world2_groups:
        decision = "DIRECTIONALLY_SPLIT_GROUP_CROSS_DAMAGE"
    else:
        decision = "ONE_DIRECTION_GROUP_CROSS_DAMAGE"
    return decision, world0_groups, world2_groups, bidirectional


def exact_parent_reproduction(final_reproduction: Mapping[str, object], final_state: Mapping[str, object]) -> bool:
    return (
        final_reproduction.get("world_token_accuracies") == [0.5, 1.0]
        and final_reproduction.get("world_full_answer_exact") == [0.0, 1.0]
        and final_reproduction.get("per_world_greedy_exact") == [0.0, 1.0]
        and final_reproduction.get("main_nonfinite_events") == 0
        and final_reproduction.get("invalid_reason") is None
        and final_state.get("model_state_digest") == PARENT_FINAL_MODEL_DIGEST
        and final_state.get("optimizer_state_digest") == PARENT_FINAL_OPTIMIZER_DIGEST
        and final_state.get("rng_state_digest") == PARENT_FINAL_RNG_DIGEST
    )
