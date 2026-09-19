from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

EXPERIMENT_ID = "EXP-334"
SCHEMA_VERSION = "EXP334-HIGHER-ORDER-CONFLICT-SUBSPACE-RESCUE-V1"
FAMILY = "iterative-grid-and-maze"
GROUP_MEMBERS = {
    "T012": (0, 1, 2),
    "T013": (0, 1, 3),
    "T023": (0, 2, 3),
    "T123": (1, 2, 3),
    "Q0123": (0, 1, 2, 3),
    "Q4567": (4, 5, 6, 7),
    "O01234567": (0, 1, 2, 3, 4, 5, 6, 7),
}
GROUP_IDS = tuple(GROUP_MEMBERS)
PARENT_CONTROL_PASS = {
    "T012": False,
    "T013": True,
    "T023": False,
    "T123": True,
    "Q0123": False,
    "Q4567": True,
    "O01234567": False,
}
MODES = ("CONTROL", "SHAM", "SUBSPACE_PROJECT")
EXPOSURES_PER_WORLD = 32
EFFORT_CYCLE = (1, 2, 4, 8)
GRADIENT_CLIP_NORM = 1.0
TARGET_NORM_SQUARED_FLOOR = 1e-24
PINV_RTOL = 1e-12
TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90
APPROVED_PREREGISTRATION_DIGEST = "cd973c9724b27ad4d881de2211522cb315c9a34275f613395c69b675d8c3fdb3"

PARENT_EXP333_RUN_ID = 35431490998
PARENT_EXP333_ARTIFACT_ID = 10581761422
PARENT_EXP333_ZIP_DIGEST = "49ca57616fa99099dc24d39d0846a5f2fcfe3e55b5eec7156d5bf42c6e0236dd"
PARENT_EXP333_JSON_SHA256 = "50e53e3b45a454b7fbb2c3358f8a7284d60ec4b473c1929bbaa157dafbc4340b"
PARENT_EXP333_EVIDENCE_DIGEST = "232ffe694650bc5b953acee2b03a84e30444d7aec41456b450ace739840e6b10"
PARENT_EXP333_EXECUTION_DIGEST = "b54dcdf76ec2b604f48ee21fb5f2c001db2e480be6d21daa8a3277c3a7758771"

PARENT_EXP327_ARTIFACT_ID = 10578071604
PARENT_EXP327_JSON_SHA256 = "cc608361850a524f6e82c06c8f65a5a7c0cb321a90b2c6d4686238a82233540f"
PARENT_EXP327_EVIDENCE_DIGEST = "a9b9f06cfc05bc2dcca025d90ecbdd65c53b261437fedb178c63b23c3a8ebeb7"

PARENT_EXP326_FAMILY_ARTIFACT_ID = 10574301567
PARENT_EXP326_FAMILY_JSON_SHA256 = "04f8f89f33f3af0145076872bb845d80751c0f3406852309215114647f1b06a7"
PARENT_EXP326_FAMILY_EVIDENCE_DIGEST = "7074a809c3d68ebe59ca6297c9fc401d1783467c0cb42ff64a271eb4d5523780"

RECONSTRUCTION_ARTIFACT_ID = 10547681681

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}

@dataclass(frozen=True, slots=True)
class GroupArmResult:
    group_id: str
    mode: str
    total_optimizer_updates: int
    world_token_accuracies: tuple[float, ...]
    world_full_answer_exact: tuple[float, ...]
    model_state_digest: str
    optimizer_state_digest: str
    rng_state_digest: str
    nonfinite_events: int
    negative_target_count: int
    projection_update_count: int
    projected_target_count: int

def effort_for_exposure(exposure_index: int) -> int:
    if not isinstance(exposure_index, int) or not 0 <= exposure_index < EXPOSURES_PER_WORLD:
        raise ValueError("exposure index")
    return EFFORT_CYCLE[exposure_index % len(EFFORT_CYCLE)]

def training_schedule(group_id: str) -> tuple[tuple[int, int, int], ...]:
    if group_id not in GROUP_MEMBERS:
        raise ValueError("group")
    return tuple(
        (world, exposure, effort_for_exposure(exposure))
        for exposure in range(EXPOSURES_PER_WORLD)
        for world in GROUP_MEMBERS[group_id]
    )

def validate_arm(result: GroupArmResult) -> None:
    if result.group_id not in GROUP_IDS or result.mode not in MODES:
        raise ValueError("arm identity")
    expected_updates = len(GROUP_MEMBERS[result.group_id]) * EXPOSURES_PER_WORLD
    if result.total_optimizer_updates != expected_updates:
        raise ValueError("update geometry")
    n = len(GROUP_MEMBERS[result.group_id])
    if len(result.world_token_accuracies) != n or len(result.world_full_answer_exact) != n:
        raise ValueError("metric geometry")
    for value in (*result.world_token_accuracies, *result.world_full_answer_exact):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("metric")
    for value in (result.model_state_digest, result.optimizer_state_digest, result.rng_state_digest):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("digest")
    max_targets = expected_updates * max(0, n - 1)
    if result.nonfinite_events < 0:
        raise ValueError("nonfinite")
    if not 0 <= result.negative_target_count <= max_targets:
        raise ValueError("negative target count")
    if not 0 <= result.projected_target_count <= result.negative_target_count:
        raise ValueError("projected target count")
    if not 0 <= result.projection_update_count <= expected_updates:
        raise ValueError("projection update count")
    if result.mode != "SUBSPACE_PROJECT" and (result.projected_target_count or result.projection_update_count):
        raise ValueError("unexpected projection")

def arm_pass(result: GroupArmResult) -> bool:
    validate_arm(result)
    return (
        result.nonfinite_events == 0
        and all(
            token >= TOKEN_FLOOR and exact >= FULL_EXACT_FLOOR
            for token, exact in zip(result.world_token_accuracies, result.world_full_answer_exact)
        )
    )

def sham_equivalent(control: GroupArmResult, sham: GroupArmResult) -> bool:
    validate_arm(control)
    validate_arm(sham)
    return (
        control.group_id == sham.group_id
        and control.mode == "CONTROL"
        and sham.mode == "SHAM"
        and sham.nonfinite_events == 0
        and control.world_token_accuracies == sham.world_token_accuracies
        and control.world_full_answer_exact == sham.world_full_answer_exact
        and control.model_state_digest == sham.model_state_digest
        and control.optimizer_state_digest == sham.optimizer_state_digest
        and control.rng_state_digest == sham.rng_state_digest
    )

def expected_keys() -> set[str]:
    return {f"{group}:{mode}" for group in GROUP_IDS for mode in MODES}

def reduce_higher_order(
    results: Mapping[str, GroupArmResult],
    *,
    parent_reproduced: bool,
    invalid: bool = False,
):
    if invalid or set(results) != expected_keys():
        return "INVALID_HIGHER_ORDER_COURT", {}, (), (), (), (), ()
    try:
        for value in results.values():
            validate_arm(value)
    except (TypeError, ValueError):
        return "INVALID_HIGHER_ORDER_COURT", {}, (), (), (), (), ()
    if any(value.nonfinite_events for value in results.values()):
        return "INVALID_HIGHER_ORDER_COURT", {}, (), (), (), (), ()

    passed = {key: arm_pass(value) for key, value in results.items()}
    if not parent_reproduced:
        return "PARENT_HIGHER_ORDER_REPRODUCTION_MISMATCH", passed, (), (), (), (), ()
    if any(not sham_equivalent(results[f"{group}:CONTROL"], results[f"{group}:SHAM"]) for group in GROUP_IDS):
        return "SHAM_HIGHER_ORDER_MISMATCH", passed, (), (), (), (), ()

    baseline_failures = tuple(group for group in GROUP_IDS if not passed[f"{group}:CONTROL"])
    rescued = tuple(group for group in baseline_failures if passed[f"{group}:SUBSPACE_PROJECT"])
    unresolved = tuple(group for group in baseline_failures if not passed[f"{group}:SUBSPACE_PROJECT"])
    regressions = tuple(
        group for group in GROUP_IDS
        if passed[f"{group}:CONTROL"] and not passed[f"{group}:SUBSPACE_PROJECT"]
    )
    no_trigger = tuple(
        group for group in baseline_failures
        if results[f"{group}:SUBSPACE_PROJECT"].projection_update_count == 0
    )
    if no_trigger:
        return "BASELINE_FAILURE_WITHOUT_SUBSPACE_TRIGGER", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    if regressions:
        return "PROJECT_HIGHER_ORDER_REGRESSION_PRESENT", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    if not rescued:
        return "BASELINE_HIGHER_ORDER_FAILURES_NOT_RESCUED", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    if unresolved:
        return "BASELINE_HIGHER_ORDER_FAILURES_PARTIALLY_RESCUED", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    return "HIGHER_ORDER_SUBSPACE_RESCUE_NO_REGRESSION", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
