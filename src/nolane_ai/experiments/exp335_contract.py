from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

EXPERIMENT_ID = "EXP-335"
SCHEMA_VERSION = "EXP335-FULL32-AFIXED-FOUNDATION-REENTRY-V1"

FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
WORLD_INDICES = tuple(range(8))
WORLD_IDS = tuple(f"{family}:{index}" for family in FAMILIES for index in WORLD_INDICES)
ARMS = ("CONTROL_FULL32", "SHAM_MEASURE_FULL32", "SUBSPACE_PROJECT_FULL32")

EXPOSURES_PER_WORLD = 32
EXPOSURES_PER_CHUNK = 4
CHUNK_COUNT = 8
UPDATES_PER_CHUNK = 128
TOTAL_SOURCE_UPDATES = 1024
EFFORT_CYCLE = (1, 2, 4, 8)
DIAGNOSTIC_EXPOSURES = (8, 16, 24, 32)

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
TARGET_NORM_SQUARED_FLOOR = 1e-24
PINV_RTOL = 1e-12
TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90

APPROVED_PREREGISTRATION_DIGEST = "db73212d75770ae3a3c0b2a3fb6f60544672970cc3ad112063b1b5bb232021cb"
PREREGISTRATION_JSON_SHA256 = "cc9f245fba352e9607400a6316db987253ccc41cbaf00dc8af9cd8cfe5952d88"

PARENT_EXP334_RUN_ID = 35438656413
PARENT_EXP334_ARTIFACT_ID = 10583626513
PARENT_EXP334_ARTIFACT_NAME = "exp334-higher-order-rescue-final-35438656413"
PARENT_EXP334_ZIP_DIGEST = "8339fd376a23b7daeb8f45087eff634893557f37c47a117f308c7f54692b8ee8"
PARENT_EXP334_JSON_SHA256 = "a393bf8fc237e2f2a357666059011e468ba7de06db3f3c62ecc3090f2aa25354"
PARENT_EXP334_EVIDENCE_DIGEST = "4cc1b2fa67610dc0614fcd6e03ac0aa122f5aa2c7f680dce01e2a29382105a05"
PARENT_EXP334_EXECUTION_DIGEST = "074625265749d9a7437bd14c132b7fe6906267904f0985f2b4e2c38ba1240a44"
PARENT_EXP334_SOURCE_SHA = "2791a3127b4cec57d02d0ce25936350cf35b7707"
PARENT_EXP334_MARKER_SHA = "b6e682af602adce57f4f36045452cac3e384fd10"

RECONSTRUCTION_RUN_ID = 35345351869
RECONSTRUCTION_ARTIFACT_ID = 10547681681
RECONSTRUCTION_ARTIFACT_NAME = "exp323r-reconstruction-candidate-35345351869-0"
RECONSTRUCTION_ZIP_DIGEST = "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621"
RECONSTRUCTION_CHECKPOINT_SHA256 = "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
RECONSTRUCTION_RECEIPT_SHA256 = "617ba665149a366d00782df5c4457a6a3d01081fe847aa144465377ee164bba3"

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}

REDUCER_ORDER = (
    "INVALID_FULL32_FOUNDATION_REENTRY",
    "PARENT_AUTHORITY_MISMATCH",
    "SHAM_FULL32_MISMATCH",
    "AFIXED_FULL32_FOUNDATION_REENTERED_BOTH",
    "AFIXED_FULL32_FOUNDATION_REENTERED_CONTROL_ONLY_PROJECT_REGRESSION",
    "BASELINE_FULL32_FAILURE_WITHOUT_SUBSPACE_TRIGGER",
    "AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION",
    "AFIXED_FULL32_FOUNDATION_REENTRY_NOT_ESTABLISHED",
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def effort_for_exposure(exposure_index: int) -> int:
    if not isinstance(exposure_index, int) or not 0 <= exposure_index < EXPOSURES_PER_WORLD:
        raise ValueError("EXP-335 exposure index")
    return EFFORT_CYCLE[exposure_index % len(EFFORT_CYCLE)]


def chunk_schedule(chunk_index: int) -> tuple[tuple[str, int, int, int], ...]:
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-335 chunk index")
    start = chunk_index * EXPOSURES_PER_CHUNK
    stop = start + EXPOSURES_PER_CHUNK
    rows = tuple(
        (family, index, exposure, effort_for_exposure(exposure))
        for exposure in range(start, stop)
        for family in FAMILIES
        for index in WORLD_INDICES
    )
    if len(rows) != UPDATES_PER_CHUNK:
        raise AssertionError("EXP-335 chunk geometry")
    return rows


def schedule_prefix(cumulative_exposure_per_world: int) -> tuple[tuple[str, int, int, int], ...]:
    if (
        not isinstance(cumulative_exposure_per_world, int)
        or cumulative_exposure_per_world < 0
        or cumulative_exposure_per_world > EXPOSURES_PER_WORLD
        or cumulative_exposure_per_world % EXPOSURES_PER_CHUNK
    ):
        raise ValueError("EXP-335 cumulative exposure geometry")
    return tuple(
        (family, index, exposure, effort_for_exposure(exposure))
        for exposure in range(cumulative_exposure_per_world)
        for family in FAMILIES
        for index in WORLD_INDICES
    )


def data_order_digest(cumulative_exposure_per_world: int) -> str:
    rows = schedule_prefix(cumulative_exposure_per_world)
    payload = {
        "schema": "EXP335-DATA-ORDER-V1",
        "families": list(FAMILIES),
        "world_indices": list(WORLD_INDICES),
        "cumulative_exposure_per_world": cumulative_exposure_per_world,
        "source_updates": [list(row) for row in rows],
    }
    return canonical_digest(payload)


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    arm: str
    chunk_index: int
    cumulative_exposure_per_world: int
    cumulative_source_updates: int
    world_token_accuracies: tuple[float, ...]
    world_full_answer_exact: tuple[float, ...]
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
        raise ValueError("EXP-335 arm")
    if not 0 <= result.chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-335 chunk")
    expected_exposure = (result.chunk_index + 1) * EXPOSURES_PER_CHUNK
    expected_updates = expected_exposure * len(WORLD_IDS)
    if result.cumulative_exposure_per_world != expected_exposure:
        raise ValueError("EXP-335 exposure boundary")
    if result.cumulative_source_updates != expected_updates:
        raise ValueError("EXP-335 source update boundary")
    if len(result.world_token_accuracies) != len(WORLD_IDS):
        raise ValueError("EXP-335 token vector")
    if len(result.world_full_answer_exact) != len(WORLD_IDS):
        raise ValueError("EXP-335 exact vector")
    for value in (*result.world_token_accuracies, *result.world_full_answer_exact):
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError("EXP-335 metric")
    for value in (result.model_state_digest, result.optimizer_state_digest, result.rng_state_digest):
        if not _is_hex_digest(value):
            raise ValueError("EXP-335 state digest")
    max_targets = result.cumulative_source_updates * 31
    if result.nonfinite_events < 0:
        raise ValueError("EXP-335 nonfinite")
    if not 0 <= result.negative_target_count <= max_targets:
        raise ValueError("EXP-335 negative target count")
    if not 0 <= result.projected_target_count <= result.negative_target_count:
        raise ValueError("EXP-335 projected target count")
    if not 0 <= result.projection_update_count <= result.cumulative_source_updates:
        raise ValueError("EXP-335 projection update count")
    if result.arm != "SUBSPACE_PROJECT_FULL32" and (
        result.projection_update_count != 0 or result.projected_target_count != 0
    ):
        raise ValueError("EXP-335 unexpected projection")


def world_pass_vector(result: BoundaryResult) -> tuple[bool, ...]:
    validate_boundary(result)
    return tuple(
        result.nonfinite_events == 0 and token >= TOKEN_FLOOR and exact >= FULL_EXACT_FLOOR
        for token, exact in zip(result.world_token_accuracies, result.world_full_answer_exact)
    )


def arm_pass(result: BoundaryResult) -> bool:
    return all(world_pass_vector(result))


def sham_equivalent(control: BoundaryResult, sham: BoundaryResult) -> bool:
    validate_boundary(control)
    validate_boundary(sham)
    return (
        control.arm == "CONTROL_FULL32"
        and sham.arm == "SHAM_MEASURE_FULL32"
        and control.chunk_index == sham.chunk_index
        and control.cumulative_exposure_per_world == sham.cumulative_exposure_per_world
        and control.cumulative_source_updates == sham.cumulative_source_updates
        and control.world_token_accuracies == sham.world_token_accuracies
        and control.world_full_answer_exact == sham.world_full_answer_exact
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
        if control.arm != "CONTROL_FULL32" or sham.arm != "SHAM_MEASURE_FULL32" or project.arm != "SUBSPACE_PROJECT_FULL32":
            invalid = True
    except (TypeError, ValueError):
        invalid = True

    empty = {
        "control_failed_worlds": [],
        "project_failed_worlds": [],
        "rescued_control_failures": [],
        "project_regressions": [],
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
        WORLD_IDS[i] for i, (c_pass, p_pass) in enumerate(zip(control_vector, project_vector))
        if not c_pass and p_pass
    ]
    regressions = [
        WORLD_IDS[i] for i, (c_pass, p_pass) in enumerate(zip(control_vector, project_vector))
        if c_pass and not p_pass
    ]
    vectors = {
        "control_failed_worlds": control_failed,
        "project_failed_worlds": project_failed,
        "rescued_control_failures": rescued,
        "project_regressions": regressions,
        "projection_update_count": project.projection_update_count,
        "projected_target_count": project.projected_target_count,
    }

    control_ok = not control_failed
    project_ok = not project_failed
    if control_ok and project_ok:
        return REDUCER_ORDER[3], vectors
    if control_ok and not project_ok:
        return REDUCER_ORDER[4], vectors
    if control_failed and project.projection_update_count == 0:
        return REDUCER_ORDER[5], vectors
    if not control_ok and project_ok and not regressions:
        return REDUCER_ORDER[6], vectors
    return REDUCER_ORDER[7], vectors
