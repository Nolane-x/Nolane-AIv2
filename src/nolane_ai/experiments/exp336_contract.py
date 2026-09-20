from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

EXPERIMENT_ID = "EXP-336"
SCHEMA_VERSION = "EXP336-CNRS-FULL32-STAGE-A-PROJECTION-INHERITANCE-V1"

FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
WORLD_INDICES = tuple(range(8))
WORLD_IDS = tuple(f"{family}:{index}" for family in FAMILIES for index in WORLD_INDICES)
ARMS = (
    "CONTROL_CNRS_FULL32",
    "SHAM_MEASURE_CNRS_FULL32",
    "SUBSPACE_PROJECT_CNRS_FULL32",
)

STARTING_CUMULATIVE_STEP = 1024
EXPOSURES_PER_WORLD = 32
EXPOSURES_PER_CHUNK = 4
CHUNK_COUNT = 8
UPDATES_PER_CHUNK = 128
TOTAL_SOURCE_UPDATES = 1024
FINAL_CUMULATIVE_STEP = STARTING_CUMULATIVE_STEP + TOTAL_SOURCE_UPDATES
EFFORT_CYCLE = (1, 2, 4, 8)
TEACHER_FORCED_EFFORT = 4

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
TARGET_NORM_SQUARED_FLOOR = 1e-24
PINV_RTOL = 1e-12

TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90
GREEDY_EXACT_FLOOR = 0.90
EOS_CORRECTNESS_FLOOR = 0.95
INVALID_OUTPUT_RATE_MAX = 0.01
LOSS_FRACTION_MAX = 0.25
ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS = 164.55180455597355

APPROVED_PREREGISTRATION_DIGEST = "491899f5f613b3d80b36689e7882e8c72875fb268a6886e1a40542eb09dc33ce"
PREREGISTRATION_JSON_SHA256 = "6090e7423332756998b8199cc77d56b7e40a296b401f63dbcc6cabecf97feeab"

PARENT_EXP319_RUN_ID = 35311529822
PARENT_SELECTION_ARTIFACT_ID = 10533822077
PARENT_SELECTION_ARTIFACT_NAME = "exp319-stage-a-selection"
PARENT_SELECTION_ZIP_DIGEST = "ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559"
PARENT_SELECTION_JSON_SHA256 = "4da236ea06a76df94ab6aa3e8141c6855a97f427f0b8614fc380d5c71e88e34a"
PARENT_SELECTION_AUTHORITY_DIGEST = "39e5e846dbe87f3d5b34e438f0c8864626caa7049148fac934f2c893032e623f"
PARENT_CNRS_SELECTION_DIGEST = "9f29edb7046b9841b87449f8ee2af635619ce16a1229b1da515984913364e1ba"

PARENT_CNRS_ARTIFACT_ID = 10534076546
PARENT_CNRS_ARTIFACT_NAME = "exp319-stage-a-C_NRS_CORE-0.0003-c3"
PARENT_CNRS_ZIP_DIGEST = "d88a9d85584476ec5226321479560a1255f0d015122827d4be75f67a6f337755"
PARENT_CNRS_CHECKPOINT_SHA256 = "bd58607a0e49bb32f45124689d11880a13040afa702bf0524cce115d947f650b"
PARENT_CNRS_RECEIPT_SHA256 = "cea3fb3d0b94e5d7cbec59813cfa3a1611dec0883a14acd6b2d0e46a23db3e4d"
PARENT_CNRS_SUMMARY_SHA256 = "32131c058049a27c24ac215cc04b77d5e1432cbda41fda71f8724f6723dfea23"
PARENT_CNRS_RECEIPT_ARTIFACT_DIGEST = "f89332de841047bd0bc8c509fb5a1d507ade5ed1cc1b14c36f8aa10a469f1ac6"
PARENT_CNRS_MODEL_STATE_DIGEST = "01c2a0f16b3f84dfee1b6db749821cf09897de05c6ceb15e42274abcd5926a76"
PARENT_CNRS_OPTIMIZER_STATE_DIGEST = "d18439109cc9e020153b33fa6c39d38d4d0e05f2262c6a9ee3598df208f1ddf9"
PARENT_CNRS_RNG_STATE_DIGEST = "3d2d8e928c4e09f880efbd2edaac3df47c88bace04c0e270c4ec0639e65bdb95"
PARENT_CNRS_SOURCE_SHA = "2002a42322c7b919c3c9dc3da7d9cb0f546d4431"
PARENT_CNRS_RUN_IDENTITY = "github-run-35311529822-exp319-v1"

PARENT_EXP335_RUN_ID = 35445927525
PARENT_EXP335_SEALED_MARKER_SHA = "57fdf3a38b42b41951639a35dbb3c0d82055846b"
PARENT_EXP335_SCIENTIFIC_SOURCE_SHA = "d29e20e5af3467d16123e224462b37ef9a2ebf9d"
PARENT_EXP335_EXECUTION_DIGEST = "24d4e97ad3c7026881a122cd309d671d9999f3f84b5dcc04a11c468c4409d520"
PARENT_EXP335_FINAL_ARTIFACT_ID = 10590026763
PARENT_EXP335_FINAL_ZIP_DIGEST = "ae4ab3bc0b8ce064ece92655fd578a896fe3ed6f27fffd411b9b14d0fc1729a5"
PARENT_EXP335_FINAL_JSON_SHA256 = "955d400eb4d590fe0399671dfadb37e8140be835451c5d7482a8928ed2686032"
PARENT_EXP335_EVIDENCE_DIGEST = "6fcabeb87971a0cb5d5ef0d14029f5ab248f3e353b21bb55f42ac977a4ffe1af"
PARENT_EXP335_AUDIT_RUN_ID = 35477337239
PARENT_EXP335_AUDIT_ARTIFACT_ID = 10595335243
PARENT_EXP335_AUDIT_ZIP_DIGEST = "0ae527b905d92e51d6ec62553fdaa0098ebde9432f7d0201baa6d773c2782b7d"
PARENT_EXP335_AUDIT_REPORT_SHA256 = "6eec3678b0b00321a83fa69cc45add2111ec6524d6ace1c691dca2a6817900a0"

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
    "INVALID_CNRS_FULL32_STAGE_A",
    "PARENT_AUTHORITY_MISMATCH",
    "SHAM_CNRS_FULL32_MISMATCH",
    "CNRS_FULL32_STAGE_A_ESTABLISHED_BOTH",
    "CNRS_FULL32_STAGE_A_ESTABLISHED_CONTROL_ONLY_PROJECT_REGRESSION",
    "CNRS_BASELINE_FAILURE_WITHOUT_PROJECTOR_TRIGGER",
    "CNRS_FULL32_STAGE_A_RESCUED_BY_INHERITED_PROJECTOR_NO_REGRESSION",
    "CNRS_FULL32_STAGE_A_NOT_ESTABLISHED",
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def effort_for_exposure(exposure_index: int) -> int:
    if not isinstance(exposure_index, int) or not 0 <= exposure_index < EXPOSURES_PER_WORLD:
        raise ValueError("EXP-336 exposure index")
    return EFFORT_CYCLE[exposure_index % len(EFFORT_CYCLE)]


def chunk_schedule(chunk_index: int) -> tuple[tuple[str, int, int, int], ...]:
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-336 chunk index")
    start = chunk_index * EXPOSURES_PER_CHUNK
    stop = start + EXPOSURES_PER_CHUNK
    rows = tuple(
        (family, index, exposure, effort_for_exposure(exposure))
        for exposure in range(start, stop)
        for family in FAMILIES
        for index in WORLD_INDICES
    )
    if len(rows) != UPDATES_PER_CHUNK:
        raise AssertionError("EXP-336 chunk geometry")
    return rows


def schedule_prefix(cumulative_exposure_per_world: int) -> tuple[tuple[str, int, int, int], ...]:
    if (
        not isinstance(cumulative_exposure_per_world, int)
        or cumulative_exposure_per_world < 0
        or cumulative_exposure_per_world > EXPOSURES_PER_WORLD
        or cumulative_exposure_per_world % EXPOSURES_PER_CHUNK
    ):
        raise ValueError("EXP-336 cumulative exposure geometry")
    return tuple(
        (family, index, exposure, effort_for_exposure(exposure))
        for exposure in range(cumulative_exposure_per_world)
        for family in FAMILIES
        for index in WORLD_INDICES
    )


def data_order_digest(cumulative_exposure_per_world: int) -> str:
    rows = schedule_prefix(cumulative_exposure_per_world)
    return canonical_digest(
        {
            "schema": "EXP336-DATA-ORDER-V1",
            "starting_cumulative_step": STARTING_CUMULATIVE_STEP,
            "families": list(FAMILIES),
            "world_indices": list(WORLD_INDICES),
            "cumulative_exposure_per_world": cumulative_exposure_per_world,
            "source_updates": [list(row) for row in rows],
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
        raise ValueError("EXP-336 arm")
    if not 0 <= result.chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-336 chunk")
    expected_exposure = (result.chunk_index + 1) * EXPOSURES_PER_CHUNK
    expected_updates = expected_exposure * len(WORLD_IDS)
    if result.cumulative_exposure_per_world != expected_exposure:
        raise ValueError("EXP-336 exposure boundary")
    if result.cumulative_source_updates != expected_updates:
        raise ValueError("EXP-336 source update boundary")
    if result.cumulative_training_step != STARTING_CUMULATIVE_STEP + expected_updates:
        raise ValueError("EXP-336 cumulative training step")
    if len(result.world_token_accuracies) != len(WORLD_IDS):
        raise ValueError("EXP-336 token vector")
    if len(result.world_full_answer_exact) != len(WORLD_IDS):
        raise ValueError("EXP-336 exact vector")
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
            raise ValueError("EXP-336 bounded metric")
    for value in (
        result.aggregate_answer_only_loss,
        result.aggregate_loss_fraction_of_original_initial,
    ):
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("EXP-336 nonnegative metric")
    for value in (result.model_state_digest, result.optimizer_state_digest, result.rng_state_digest):
        if not _is_hex_digest(value):
            raise ValueError("EXP-336 state digest")
    max_targets = result.cumulative_source_updates * 31
    if result.nonfinite_events < 0:
        raise ValueError("EXP-336 nonfinite")
    if not 0 <= result.negative_target_count <= max_targets:
        raise ValueError("EXP-336 negative target count")
    if not 0 <= result.projected_target_count <= result.negative_target_count:
        raise ValueError("EXP-336 projected target count")
    if not 0 <= result.projection_update_count <= result.cumulative_source_updates:
        raise ValueError("EXP-336 projection update count")
    if result.arm != "SUBSPACE_PROJECT_CNRS_FULL32" and (
        result.projection_update_count != 0 or result.projected_target_count != 0
    ):
        raise ValueError("EXP-336 unexpected projection")


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
        control.arm == "CONTROL_CNRS_FULL32"
        and sham.arm == "SHAM_MEASURE_CNRS_FULL32"
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
    )


def reduce_full32(
    control: BoundaryResult,
    sham: BoundaryResult,
    project: BoundaryResult,
    *,
    parent_authority_valid: bool,
    invalid: bool = False,
) -> tuple[str, dict[str, Any]]:
    try:
        for result in (control, sham, project):
            validate_boundary(result)
        if {control.chunk_index, sham.chunk_index, project.chunk_index} != {CHUNK_COUNT - 1}:
            invalid = True
        if control.arm != "CONTROL_CNRS_FULL32":
            invalid = True
        if sham.arm != "SHAM_MEASURE_CNRS_FULL32":
            invalid = True
        if project.arm != "SUBSPACE_PROJECT_CNRS_FULL32":
            invalid = True
    except (TypeError, ValueError):
        invalid = True

    empty = {
        "control_failed_worlds": [],
        "project_failed_worlds": [],
        "rescued_control_failures": [],
        "project_regressions": [],
        "control_aggregate_pass": False,
        "project_aggregate_pass": False,
        "projection_update_count": 0,
        "projected_target_count": 0,
    }
    if invalid:
        return REDUCER_ORDER[0], empty
    if not parent_authority_valid:
        return REDUCER_ORDER[1], empty
    if not sham_equivalent(control, sham):
        return REDUCER_ORDER[2], empty

    control_vector = world_pass_vector(control)
    project_vector = world_pass_vector(project)
    control_failed = [WORLD_IDS[i] for i, passed in enumerate(control_vector) if not passed]
    project_failed = [WORLD_IDS[i] for i, passed in enumerate(project_vector) if not passed]
    rescued = [
        WORLD_IDS[i]
        for i, (c_pass, p_pass) in enumerate(zip(control_vector, project_vector))
        if not c_pass and p_pass
    ]
    regressions = [
        WORLD_IDS[i]
        for i, (c_pass, p_pass) in enumerate(zip(control_vector, project_vector))
        if c_pass and not p_pass
    ]
    vectors = {
        "control_failed_worlds": control_failed,
        "project_failed_worlds": project_failed,
        "rescued_control_failures": rescued,
        "project_regressions": regressions,
        "control_aggregate_pass": aggregate_pass(control),
        "project_aggregate_pass": aggregate_pass(project),
        "projection_update_count": project.projection_update_count,
        "projected_target_count": project.projected_target_count,
    }

    control_ok = arm_pass(control)
    project_ok = arm_pass(project)
    projector_trigger = project.projection_update_count > 0 and project.projected_target_count > 0

    if control_ok and project_ok:
        return REDUCER_ORDER[3], vectors
    if control_ok and not project_ok:
        return REDUCER_ORDER[4], vectors
    if not control_ok and not project_ok and not projector_trigger:
        return REDUCER_ORDER[5], vectors
    if not control_ok and project_ok and projector_trigger and rescued and not regressions:
        return REDUCER_ORDER[6], vectors
    return REDUCER_ORDER[7], vectors
