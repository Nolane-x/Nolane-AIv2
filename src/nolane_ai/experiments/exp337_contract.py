from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

EXPERIMENT_ID = "EXP-337"
SCHEMA_VERSION = "EXP337-CNRS-SELF-ROLLIN-SEQUENCE-RECOVERY-V1"

FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
WORLD_INDICES = tuple(range(8))
WORLD_IDS = tuple(f"{family}:{index}" for family in FAMILIES for index in WORLD_INDICES)
ARMS = (
    "CONTROL_PROJECT_GOLD_PREFIX",
    "SHAM_SELF_ROLLIN_MEASURE_PROJECT",
    "SELF_ROLLIN_RECOVERY_PROJECT",
)

STARTING_CUMULATIVE_STEP = 2048
EXPOSURES_PER_WORLD = 32
EXPOSURES_PER_CHUNK = 4
CHUNK_COUNT = 8
UPDATES_PER_CHUNK = 128
TOTAL_SOURCE_UPDATES = 1024
FINAL_CUMULATIVE_STEP = 3072
EFFORT_CYCLE = (1, 2, 4, 8)
TEACHER_FORCED_EFFORT = 4

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
TARGET_NORM_SQUARED_FLOOR = 1e-24
PINV_RTOL = 1e-12
GOLD_LOSS_WEIGHT = 0.5
SELF_ROLLIN_LOSS_WEIGHT = 0.5

TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90
GREEDY_EXACT_FLOOR = 0.90
EOS_CORRECTNESS_FLOOR = 0.95
INVALID_OUTPUT_RATE_MAX = 0.01
LOSS_FRACTION_MAX = 0.25
ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS = 164.55180455597355

APPROVED_PREREGISTRATION_DIGEST = "5172c3f85d15d71e7fc4a16e3f00ae5d9a5031f09a97c1529f8fa6e25822c53a"
PREREGISTRATION_JSON_SHA256 = "95857497dcc0cc264e5e9c76e1f8cb3a15a6b05f1e3bab90795a5f9be7f84423"
PREREGISTRATION_GIT_BLOB_SHA = "37afa42cf66e874c457f31415c5f426d318e9804"
DESIGN_GIT_BLOB_SHA = "1e8097d4fc56686b7bd24c561811540d8d4c5321"
DESIGN_RAW_SHA256 = "2b251900e435f219a8d2cb48ecf486674bbb1831cf99b9e41d0396c8b0ef19bf"
PREREGISTRATION_LOCK_V2_BLOB_SHA = "f200c69090cdb8c0f8e1b7f72b63ce3f227b2bc8"

PARENT_EXP336_RUN_ID = 35481336946
PARENT_EXP336_HEAD_SHA = "08e2caebca4c1b7e4b197f6d2190ef90e94468fd"
PARENT_EXP336_FINAL_DECISION = "CNRS_FULL32_STAGE_A_NOT_ESTABLISHED"
PARENT_EXP336_EVIDENCE_DIGEST = "c89f70f3555ad95892794821eddbcc158a2e7e9d7c4ef4fa02082a00acd6ccf7"
PARENT_EXP336_CLOSURE_DIGEST = "c291b3f8f0af0952ce8bbbda8fd107e29d8bd916e4656f2d13cc2e9b70a53d48"
PARENT_EXP336_SOURCE_TREE_DIGEST = "6485fdc131dbf0358f8e6548b3ec510ce96bb80b8fbaf2858a97d5bade62a911"
PARENT_EXP336_PREREGISTRATION_DIGEST = "491899f5f613b3d80b36689e7882e8c72875fb268a6886e1a40542eb09dc33ce"
PARENT_EXP336_CHUNK7_ARTIFACT_ID = 10601286249
PARENT_EXP336_CHUNK7_ARTIFACT_NAME = "exp336-chunk-7-35481336946"
PARENT_EXP336_CHUNK7_ZIP_SHA256 = "5a12a2e59e399bec993ee1597ddbf4f85546e1d5de0b26635cb1d7def5a9666d"
PARENT_EXP336_CHUNK7_BUNDLE_DIGEST = "c5c72c4919077b0d1647939d4335cea79f579090af27fb2ff0a5e82e91d8cae8"
PARENT_PROJECT_CHECKPOINT_SHA256 = "a9e638f3029e4af05a685dd4cfba096da6107221ca3a31cb26e5422b4d394208"
PARENT_PROJECT_RECEIPT_JSON_SHA256 = "0efb7d92bd3a75676c0a3c7f5326031f9916101fd1809b0fd87c0217cff7a0f1"
PARENT_PROJECT_RECEIPT_DIGEST = "6831373eb293c6c32281e0c94059610af62f11e1da85cf7f80950d1cb415c04f"
PARENT_PROJECT_MODEL_STATE_DIGEST = "e122331caa2e6924c13d4c479574dc8d3e4e1de823adfc02604158ec3ad7fc38"
PARENT_PROJECT_OPTIMIZER_STATE_DIGEST = "44db727abdf9edca155ebd1a3d9e697a4d0b0e1b82e70371e3a77768b72932a7"
PARENT_PROJECT_RNG_STATE_DIGEST = "3d2d8e928c4e09f880efbd2edaac3df47c88bace04c0e270c4ec0639e65bdb95"

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "stage_b_implementation_authorized": False,
    "stage_c_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}

REDUCER_ORDER = (
    "INVALID_EXP337_SELF_ROLLIN_COURT",
    "EXP337_PARENT_AUTHORITY_MISMATCH",
    "EXP337_SHAM_MEASUREMENT_MISMATCH",
    "EXP337_STAGE_A_ESTABLISHED_BOTH",
    "EXP337_STAGE_A_ESTABLISHED_CONTROL_ONLY_RECOVERY_REGRESSION",
    "EXP337_STAGE_A_RESCUED_BY_SELF_ROLLIN_NO_REGRESSION",
    "EXP337_SELF_ROLLIN_NO_CAUSAL_RESCUE",
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def effort_for_exposure(exposure_index: int) -> int:
    if not isinstance(exposure_index, int) or not 0 <= exposure_index < EXPOSURES_PER_WORLD:
        raise ValueError("EXP-337 exposure index")
    return EFFORT_CYCLE[exposure_index % len(EFFORT_CYCLE)]


def chunk_schedule(chunk_index: int) -> tuple[tuple[str, int, int, int], ...]:
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-337 chunk index")
    start = chunk_index * EXPOSURES_PER_CHUNK
    stop = start + EXPOSURES_PER_CHUNK
    rows = tuple(
        (family, index, exposure, effort_for_exposure(exposure))
        for exposure in range(start, stop)
        for family in FAMILIES
        for index in WORLD_INDICES
    )
    if len(rows) != UPDATES_PER_CHUNK:
        raise AssertionError("EXP-337 chunk geometry")
    return rows


def schedule_prefix(cumulative_exposure_per_world: int) -> tuple[tuple[str, int, int, int], ...]:
    if (
        not isinstance(cumulative_exposure_per_world, int)
        or cumulative_exposure_per_world < 0
        or cumulative_exposure_per_world > EXPOSURES_PER_WORLD
        or cumulative_exposure_per_world % EXPOSURES_PER_CHUNK
    ):
        raise ValueError("EXP-337 cumulative exposure geometry")
    return tuple(
        (family, index, exposure, effort_for_exposure(exposure))
        for exposure in range(cumulative_exposure_per_world)
        for family in FAMILIES
        for index in WORLD_INDICES
    )


def data_order_digest(cumulative_exposure_per_world: int) -> str:
    return canonical_digest(
        {
            "schema": "EXP337-DATA-ORDER-V1",
            "starting_cumulative_step": STARTING_CUMULATIVE_STEP,
            "families": list(FAMILIES),
            "world_indices": list(WORLD_INDICES),
            "cumulative_exposure_per_world": cumulative_exposure_per_world,
            "source_updates": [list(row) for row in schedule_prefix(cumulative_exposure_per_world)],
        }
    )


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    arm: str
    chunk_index: int
    cumulative_exposure_per_world: int
    cumulative_source_updates: int
    cumulative_training_step: int
    world_token_accuracies: tuple[float, ...]
    world_full_answer_exact: tuple[float, ...]
    aggregate_answer_only_loss: float
    aggregate_answer_token_accuracy: float
    aggregate_greedy_exact_match: float
    aggregate_eos_correctness: float
    aggregate_invalid_output_rate: float
    aggregate_loss_fraction_of_original_initial: float
    model_state_digest: str
    optimizer_state_digest: str
    rng_state_digest: str
    nonfinite_events: int
    negative_target_count: int
    projection_update_count: int
    projected_target_count: int
    self_rollin_measurement_count: int
    self_rollin_active_update_count: int
    self_rollin_divergent_update_count: int


def _is_hex_digest(value: object, length: int = 64) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_boundary(result: BoundaryResult) -> None:
    if result.arm not in ARMS:
        raise ValueError("EXP-337 arm")
    if not 0 <= result.chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-337 chunk")
    expected_exposure = (result.chunk_index + 1) * EXPOSURES_PER_CHUNK
    expected_updates = expected_exposure * len(WORLD_IDS)
    if result.cumulative_exposure_per_world != expected_exposure:
        raise ValueError("EXP-337 exposure boundary")
    if result.cumulative_source_updates != expected_updates:
        raise ValueError("EXP-337 source update boundary")
    if result.cumulative_training_step != STARTING_CUMULATIVE_STEP + expected_updates:
        raise ValueError("EXP-337 cumulative training step")
    if len(result.world_token_accuracies) != len(WORLD_IDS):
        raise ValueError("EXP-337 token vector")
    if len(result.world_full_answer_exact) != len(WORLD_IDS):
        raise ValueError("EXP-337 exact vector")
    unit_values = (
        *result.world_token_accuracies,
        *result.world_full_answer_exact,
        result.aggregate_answer_token_accuracy,
        result.aggregate_greedy_exact_match,
        result.aggregate_eos_correctness,
        result.aggregate_invalid_output_rate,
    )
    for value in unit_values:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError("EXP-337 bounded metric")
    for value in (
        result.aggregate_answer_only_loss,
        result.aggregate_loss_fraction_of_original_initial,
    ):
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("EXP-337 nonnegative metric")
    for value in (result.model_state_digest, result.optimizer_state_digest, result.rng_state_digest):
        if not _is_hex_digest(value):
            raise ValueError("EXP-337 state digest")
    max_targets = result.cumulative_source_updates * 31
    if result.nonfinite_events < 0:
        raise ValueError("EXP-337 nonfinite")
    if not 0 <= result.negative_target_count <= max_targets:
        raise ValueError("EXP-337 negative target count")
    if not 0 <= result.projected_target_count <= result.negative_target_count:
        raise ValueError("EXP-337 projected target count")
    if not 0 <= result.projection_update_count <= result.cumulative_source_updates:
        raise ValueError("EXP-337 projection update count")
    if not 0 <= result.self_rollin_divergent_update_count <= result.self_rollin_measurement_count:
        raise ValueError("EXP-337 roll-in divergence count")

    if result.arm == "CONTROL_PROJECT_GOLD_PREFIX":
        if result.self_rollin_measurement_count != 0 or result.self_rollin_active_update_count != 0:
            raise ValueError("EXP-337 CONTROL roll-in counters")
    elif result.arm == "SHAM_SELF_ROLLIN_MEASURE_PROJECT":
        if (
            result.self_rollin_measurement_count != result.cumulative_source_updates
            or result.self_rollin_active_update_count != 0
        ):
            raise ValueError("EXP-337 SHAM roll-in counters")
    else:
        if (
            result.self_rollin_measurement_count != result.cumulative_source_updates
            or result.self_rollin_active_update_count != result.cumulative_source_updates
        ):
            raise ValueError("EXP-337 RECOVERY roll-in counters")


def world_pass_vector(result: BoundaryResult) -> tuple[bool, ...]:
    validate_boundary(result)
    return tuple(
        result.nonfinite_events == 0 and token >= TOKEN_FLOOR and exact >= FULL_EXACT_FLOOR
        for token, exact in zip(result.world_token_accuracies, result.world_full_answer_exact)
    )


def aggregate_pass(result: BoundaryResult) -> bool:
    validate_boundary(result)
    return (
        result.nonfinite_events == 0
        and result.aggregate_greedy_exact_match >= GREEDY_EXACT_FLOOR
        and result.aggregate_answer_token_accuracy >= TOKEN_FLOOR
        and result.aggregate_eos_correctness >= EOS_CORRECTNESS_FLOOR
        and result.aggregate_loss_fraction_of_original_initial <= LOSS_FRACTION_MAX
        and result.aggregate_invalid_output_rate <= INVALID_OUTPUT_RATE_MAX
    )


def arm_pass(result: BoundaryResult) -> bool:
    return all(world_pass_vector(result)) and aggregate_pass(result)


def sham_equivalent(control: BoundaryResult, sham: BoundaryResult) -> bool:
    validate_boundary(control)
    validate_boundary(sham)
    return (
        control.arm == "CONTROL_PROJECT_GOLD_PREFIX"
        and sham.arm == "SHAM_SELF_ROLLIN_MEASURE_PROJECT"
        and control.chunk_index == sham.chunk_index
        and control.cumulative_exposure_per_world == sham.cumulative_exposure_per_world
        and control.cumulative_source_updates == sham.cumulative_source_updates
        and control.cumulative_training_step == sham.cumulative_training_step
        and control.world_token_accuracies == sham.world_token_accuracies
        and control.world_full_answer_exact == sham.world_full_answer_exact
        and control.aggregate_answer_only_loss == sham.aggregate_answer_only_loss
        and control.aggregate_answer_token_accuracy == sham.aggregate_answer_token_accuracy
        and control.aggregate_greedy_exact_match == sham.aggregate_greedy_exact_match
        and control.aggregate_eos_correctness == sham.aggregate_eos_correctness
        and control.aggregate_invalid_output_rate == sham.aggregate_invalid_output_rate
        and control.aggregate_loss_fraction_of_original_initial == sham.aggregate_loss_fraction_of_original_initial
        and control.model_state_digest == sham.model_state_digest
        and control.optimizer_state_digest == sham.optimizer_state_digest
        and control.rng_state_digest == sham.rng_state_digest
        and control.nonfinite_events == sham.nonfinite_events == 0
        and control.negative_target_count == sham.negative_target_count
        and control.projection_update_count == sham.projection_update_count
        and control.projected_target_count == sham.projected_target_count
    )


def reduce_full32(
    control: BoundaryResult,
    sham: BoundaryResult,
    recovery: BoundaryResult,
    *,
    parent_authority_valid: bool,
    invalid: bool = False,
) -> tuple[str, dict[str, Any]]:
    try:
        for result in (control, sham, recovery):
            validate_boundary(result)
        if {control.chunk_index, sham.chunk_index, recovery.chunk_index} != {CHUNK_COUNT - 1}:
            invalid = True
        if control.arm != "CONTROL_PROJECT_GOLD_PREFIX":
            invalid = True
        if sham.arm != "SHAM_SELF_ROLLIN_MEASURE_PROJECT":
            invalid = True
        if recovery.arm != "SELF_ROLLIN_RECOVERY_PROJECT":
            invalid = True
    except (TypeError, ValueError):
        invalid = True

    empty = {
        "control_failed_worlds": [],
        "recovery_failed_worlds": [],
        "rescued_control_failures": [],
        "recovery_regressions": [],
        "control_aggregate_pass": False,
        "recovery_aggregate_pass": False,
    }
    if invalid:
        return REDUCER_ORDER[0], empty
    if not parent_authority_valid:
        return REDUCER_ORDER[1], empty
    if not sham_equivalent(control, sham):
        return REDUCER_ORDER[2], empty

    control_vector = world_pass_vector(control)
    recovery_vector = world_pass_vector(recovery)
    control_failed = [WORLD_IDS[i] for i, passed in enumerate(control_vector) if not passed]
    recovery_failed = [WORLD_IDS[i] for i, passed in enumerate(recovery_vector) if not passed]
    rescued = [
        WORLD_IDS[i]
        for i, (c_pass, r_pass) in enumerate(zip(control_vector, recovery_vector))
        if not c_pass and r_pass
    ]
    regressions = [
        WORLD_IDS[i]
        for i, (c_pass, r_pass) in enumerate(zip(control_vector, recovery_vector))
        if c_pass and not r_pass
    ]
    vectors = {
        "control_failed_worlds": control_failed,
        "recovery_failed_worlds": recovery_failed,
        "rescued_control_failures": rescued,
        "recovery_regressions": regressions,
        "control_aggregate_pass": aggregate_pass(control),
        "recovery_aggregate_pass": aggregate_pass(recovery),
    }

    control_ok = arm_pass(control)
    recovery_ok = arm_pass(recovery)
    if control_ok and recovery_ok:
        return REDUCER_ORDER[3], vectors
    if control_ok and not recovery_ok:
        return REDUCER_ORDER[4], vectors
    if not control_ok and recovery_ok and rescued and not regressions:
        return REDUCER_ORDER[5], vectors
    return REDUCER_ORDER[6], vectors
