from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

EXPERIMENT_ID = "EXP-333"
SCHEMA_VERSION = "EXP333-COMPLETE-8WORLD-PAIR-LATTICE-PROJECTION-V1"
FAMILY = "iterative-grid-and-maze"
WORLD_INDICES = tuple(range(8))
PAIR_MEMBERS = {f"P{i}{j}": (i, j) for i in WORLD_INDICES for j in WORLD_INDICES if i < j}
PAIR_IDS = tuple(PAIR_MEMBERS)
PARENT_ANCHOR_PAIR_IDS = ("P01", "P02", "P03", "P12", "P13", "P23")
PARENT_CONTROL_PASS = {"P01": True, "P02": False, "P03": True, "P12": True, "P13": True, "P23": True}
MODES = ("CONTROL", "SHAM", "PROJECT")
EXPOSURES_PER_WORLD = 32
EFFORT_CYCLE = (1, 2, 4, 8)
GRADIENT_CLIP_NORM = 1.0
TARGET_NORM_SQUARED_FLOOR = 1e-24
TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90
APPROVED_PREREGISTRATION_DIGEST = "5d24139cbaf211b25c8533d499cfa7505564d95967b64e99ff8c1acd9d4c8f08"

PARENT_EXP332_RUN_ID = 35427969759
PARENT_EXP332_ARTIFACT_ID = 10580076061
PARENT_EXP332_ZIP_DIGEST = "7bdbb30570a98326af5a98effb7718c2f40d8430a1a2f5a115fb3590a901b141"
PARENT_EXP332_JSON_SHA256 = "02bf65b708e58837aa5b2a61c3af6acff09d000f8916bcda3a27a59ebdcd6355"
PARENT_EXP332_EVIDENCE_DIGEST = "a50e23fe4fe875358b95d060e22905a6e12eb2c27b4666b7477fb2e96aaedd2b"
PARENT_EXP332_EXECUTION_DIGEST = "6fc30fa5eaca26594fa79d3b37f817e67cc89c5570444dbf9b15b8e9823c21bb"
RECONSTRUCTION_ARTIFACT_ID = 10547681681

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}

@dataclass(frozen=True, slots=True)
class PairArmResult:
    pair_id: str
    mode: str
    total_optimizer_updates: int
    world_token_accuracies: tuple[float, float]
    world_full_answer_exact: tuple[float, float]
    model_state_digest: str
    optimizer_state_digest: str
    rng_state_digest: str
    nonfinite_events: int
    negative_dot_count: int
    projection_event_count: int

def effort_for_round(round_index: int) -> int:
    if not isinstance(round_index, int) or not 0 <= round_index < EXPOSURES_PER_WORLD:
        raise ValueError("round index")
    return EFFORT_CYCLE[round_index % len(EFFORT_CYCLE)]

def training_schedule(pair_id: str) -> tuple[tuple[int, int, int], ...]:
    if pair_id not in PAIR_MEMBERS:
        raise ValueError("pair")
    return tuple(
        (world, round_index, effort_for_round(round_index))
        for round_index in range(EXPOSURES_PER_WORLD)
        for world in PAIR_MEMBERS[pair_id]
    )

def validate_arm(result: PairArmResult) -> None:
    if result.pair_id not in PAIR_IDS or result.mode not in MODES or result.total_optimizer_updates != 64:
        raise ValueError("arm identity")
    if len(result.world_token_accuracies) != 2 or len(result.world_full_answer_exact) != 2:
        raise ValueError("metric geometry")
    for value in (*result.world_token_accuracies, *result.world_full_answer_exact):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("metric")
    for value in (result.model_state_digest, result.optimizer_state_digest, result.rng_state_digest):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("digest")
    if result.nonfinite_events < 0 or not 0 <= result.negative_dot_count <= 64 or not 0 <= result.projection_event_count <= 64:
        raise ValueError("count")
    if result.projection_event_count > result.negative_dot_count:
        raise ValueError("projection count")
    if result.mode != "PROJECT" and result.projection_event_count != 0:
        raise ValueError("unexpected projection")
    if result.mode == "CONTROL" and result.negative_dot_count != 0:
        raise ValueError("control measured dots")

def arm_pass(result: PairArmResult) -> bool:
    validate_arm(result)
    return (
        result.nonfinite_events == 0
        and all(
            token >= TOKEN_FLOOR and exact >= FULL_EXACT_FLOOR
            for token, exact in zip(result.world_token_accuracies, result.world_full_answer_exact)
        )
    )

def sham_equivalent(control: PairArmResult, sham: PairArmResult) -> bool:
    validate_arm(control)
    validate_arm(sham)
    return (
        control.pair_id == sham.pair_id
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
    return {f"{pair}:{mode}" for pair in PAIR_IDS for mode in MODES}

def reduce_complete_pair_lattice(
    results: Mapping[str, PairArmResult],
    *,
    parent_anchor_reproduced: bool,
    invalid: bool = False,
) -> tuple[
    str,
    dict[str, bool],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if invalid or set(results) != expected_keys():
        return "INVALID_COMPLETE_PAIR_LATTICE_COURT", {}, (), (), (), (), ()
    try:
        for value in results.values():
            validate_arm(value)
    except (TypeError, ValueError):
        return "INVALID_COMPLETE_PAIR_LATTICE_COURT", {}, (), (), (), (), ()
    if any(value.nonfinite_events for value in results.values()):
        return "INVALID_COMPLETE_PAIR_LATTICE_COURT", {}, (), (), (), (), ()

    passed = {key: arm_pass(value) for key, value in results.items()}
    if not parent_anchor_reproduced:
        return "PARENT_PAIR_ANCHOR_REPRODUCTION_MISMATCH", passed, (), (), (), (), ()
    if any(not sham_equivalent(results[f"{pair}:CONTROL"], results[f"{pair}:SHAM"]) for pair in PAIR_IDS):
        return "SHAM_COMPLETE_PAIR_LATTICE_MISMATCH", passed, (), (), (), (), ()
    if results["P02:PROJECT"].projection_event_count == 0:
        return "P02_PROJECTION_NOT_TRIGGERED", passed, (), (), (), (), ()

    baseline_failures = tuple(pair for pair in PAIR_IDS if not passed[f"{pair}:CONTROL"])
    rescued = tuple(pair for pair in baseline_failures if passed[f"{pair}:PROJECT"])
    unresolved = tuple(pair for pair in baseline_failures if not passed[f"{pair}:PROJECT"])
    regressions = tuple(
        pair for pair in PAIR_IDS
        if passed[f"{pair}:CONTROL"] and not passed[f"{pair}:PROJECT"]
    )
    no_trigger = tuple(
        pair for pair in baseline_failures
        if results[f"{pair}:PROJECT"].projection_event_count == 0
   )

    if no_trigger:
        return "BASELINE_FAILURE_WITHOUT_PROJECTION_TRIGGER", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    if regressions:
        return "PROJECT_PAIR_REGRESSION_PRESENT", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    if not rescued:
        return "BASELINE_PAIR_FAILURES_NOT_RESCUED", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    if unresolved:
        return "BASELINE_PAIR_FAILURES_PARTIALLY_RESCUED", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
    return "COMPLETE_8WORLD_PAIR_LATTICE_RESCUE_NO_REGRESSION", passed, baseline_failures, rescued, unresolved, regressions, no_trigger
