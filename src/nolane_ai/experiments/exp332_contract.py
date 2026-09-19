from __future__ import annotations
from typing import Mapping

from .exp331_contract import (
    AUTHORIZATION_FLAGS,
    MODES,
    PAIR_IDS,
    PAIR_MEMBERS,
    PARENT_PAIR_PASS,
    PairArmResult,
    arm_pass,
    sham_equivalent,
    validate_arm,
)

EXPERIMENT_ID = "EXP-332"
SCHEMA_VERSION = "EXP332-PORTABLE-PAIR-LATTICE-PROJECTION-CONFIRMATION-V1"
APPROVED_PREREGISTRATION_DIGEST = "0e00c308e2136a19202988aecba678b5bd7cfe2a96e07b25a9f6860065a669d0"

EXP327_RUN_ID = 35413434081
EXP327_ATTEMPT1_ARTIFACT_ID = 10574837015
EXP327_ATTEMPT1_ZIP_DIGEST = "b4ddab033a85762c99451f6532f790fbf4cda9181ff67c1a2093613dd0156d77"
EXP327_ATTEMPT1_JSON_SHA256 = "27f8ef1fbe2bbb2a2a931d9dddd1409a6f3a0b9a8ecb06902f1e46bb07543279"
EXP327_ATTEMPT1_EVIDENCE_DIGEST = "bc9999f3f6e366de8ec41b26ec265514546cf68cb82661295a31b202e3e907e2"
EXP327_ATTEMPT2_ARTIFACT_ID = 10578071604
EXP327_ATTEMPT2_ZIP_DIGEST = "a5093c7b332f2a174336f3216ec878b09fc8fa0e5dcda9cd5f7651b796708b95"
EXP327_ATTEMPT2_JSON_SHA256 = "cc608361850a524f6e82c06c8f65a5a7c0cb321a90b2c6d4686238a82233540f"
EXP327_ATTEMPT2_EVIDENCE_DIGEST = "a9b9f06cfc05bc2dcca025d90ecbdd65c53b261437fedb178c63b23c3a8ebeb7"
EXP327_SOURCE_SHA = "9753aa78f768ce6b6a20436f9b6729d9bf1c20af"
EXP327_EXECUTION_DIGEST = "e9e48638779f9c33772659adfe1f6db860e5966ca598958f75a606d149bb0b08"
EXP327_WORKFLOW_SHA256 = "11dbceedfb91135f4863b2f7efcad16948c70705d2a6a625988beca303065f5c"

EXP330_RUN_ID = 35419828278
EXP330_ARTIFACT_ID = 10577706134
EXP330_ZIP_DIGEST = "cadf92521948973bb319e8280934e559526616097c75f74a1fc1a90d02c33ab7"
EXP330_JSON_SHA256 = "8aefca4b8fa4b6f814da219a2e58a4e97bab849a4f3291f95a6b2d19c35f4b2d"
EXP330_EVIDENCE_DIGEST = "6a48ad66fbe9b07a54f1aca0ad8ed5b92759101817543d5fa4c228cddf37fae3"

RECONSTRUCTION_ARTIFACT_ID = 10547681681

def metric_vector_equal(left: object, right: object) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return tuple(left) == tuple(right)

def reduce_portable_pair_lattice(
    results: Mapping[str, PairArmResult],
    *,
    parent_behavior_reproduced: bool,
    witness_divergence_confirmed: bool,
    invalid: bool = False,
) -> tuple[str, dict[str, bool], tuple[str, ...]]:
    expected = {f"{p}:{m}" for p in PAIR_IDS for m in MODES}
    if invalid or not witness_divergence_confirmed or set(results) != expected:
        return "INVALID_PORTABLE_PAIR_LATTICE_COURT", {}, ()
    try:
        for value in results.values():
            validate_arm(value)
    except (TypeError, ValueError):
        return "INVALID_PORTABLE_PAIR_LATTICE_COURT", {}, ()
    if any(value.nonfinite_events for value in results.values()):
        return "INVALID_PORTABLE_PAIR_LATTICE_COURT", {}, ()
    passed = {key: arm_pass(value) for key, value in results.items()}
    if not parent_behavior_reproduced:
        return "PARENT_BEHAVIORAL_REPRODUCTION_MISMATCH", passed, ()
    if any(not sham_equivalent(results[f"{pair}:CONTROL"], results[f"{pair}:SHAM"]) for pair in PAIR_IDS):
        return "SHAM_PAIR_LATTICE_MISMATCH", passed, ()
    if results["P02:PROJECT"].projection_event_count == 0:
        return "P02_PROJECTION_NOT_TRIGGERED", passed, ()
    regressions = tuple(
        pair for pair in PAIR_IDS
        if PARENT_PAIR_PASS[pair] and not passed[f"{pair}:PROJECT"]
    )
    if passed["P02:PROJECT"] and not regressions:
        return "PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION", passed, regressions
    if passed["P02:PROJECT"]:
        return "P02_RESCUE_WITH_PAIR_REGRESSION", passed, regressions
    return "P02_PROJECTION_NO_RESCUE", passed, regressions
