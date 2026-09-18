from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = "EXP321-AFIXED-STAGE-A-FAILURE-LOCALIZATION-V1"
EXPERIMENT_ID = "EXP-321"

EXP319_RUN_ID = 35311529822
EXP319_MARKER_SHA = "124584061616ab0864355219645ed00c1298c575"
EXP319_SOURCE_SHA = "2002a42322c7b919c3c9dc3da7d9cb0f546d4431"
EXP319_FINAL_EVIDENCE_DIGEST = (
    "4be2850f6c1d633af4cb6687e379c0011aaf8281075bea79821d0f2e038ffb07"
)
STAGE_A_SELECTION_ARTIFACT_ID = 10533822077
STAGE_A_SELECTION_ZIP_DIGEST = (
    "ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559"
)
STAGE_A_SELECTION_AUTHORITY_DIGEST = (
    "39e5e846dbe87f3d5b34e438f0c8864626caa7049148fac934f2c893032e623f"
)

SELECTED_CHECKPOINT_ARTIFACT_ID = 10534351390
SELECTED_CHECKPOINT_ZIP_DIGEST = (
    "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3"
)
SELECTED_RECEIPT_ARTIFACT_DIGEST = (
    "d874545fa677569e30849128df337e27823d8aa4c5335adb9cb422968096a280"
)
SELECTED_MODEL_STATE_DIGEST = (
    "18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb"
)
SELECTED_ARM = "A_FIXED"
SELECTED_LEARNING_RATE = 1e-4
SELECTED_STEP = 1024

EFFORT_GRID = (1, 2, 4, 8)
VOCAB_SIZE = 4608
EOS_ID = 3
BYTE_ID_START = 4
BYTE_ID_END_INCLUSIVE = 259
UNUSED_ID_START = 260
UNUSED_ID_END_INCLUSIVE = 4607

DISPOSITIONS = (
    "INVALID_LOCALIZATION",
    "TEACHER_FORCED_FOUNDATION_INSUFFICIENT",
    "ROLLOUT_EXPOSURE_BOTTLENECK",
    "EFFORT_MISMATCH_EVIDENT",
    "UNUSED_VOCAB_COMPETITION_EVIDENT",
    "MIXED_TRAINING_STACK_FAILURE",
    "NO_DOMINANT_LOCALIZED_CAUSE",
)

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}


@dataclass(frozen=True)
class LocalizationThresholds:
    teacher_forced_token_accuracy: float = 0.99
    teacher_forced_full_answer_exact: float = 0.90
    greedy_exact: float = 0.90
    effort_effect_min: float = 0.25
    masked_exact_gain_min: float = 0.25
    masked_wrong_target_recovery_min: float = 0.25
    severe_family_teacher_forced_exact: float = 0.50
    severe_family_greedy_exact: float = 0.25
    rollout_gap_min: float = 0.50


@dataclass(frozen=True)
class ReproductionAnchors:
    effort: int = 4
    teacher_forced_token_accuracy: float = 0.7711267605633803
    greedy_exact: float = 0.28125
    eos_correctness: float = 1.0
    invalid_output_rate: float = 0.0


THRESHOLDS = LocalizationThresholds()
REPRODUCTION_ANCHORS = ReproductionAnchors()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def preregistration_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "role": "post_exp319_gradient_free_failure_localization",
        "scientific_boundary": {
            "exp319_disposition": "TRAINING_STACK_NOT_LEARNABLE",
            "exp319_final_evidence_digest": EXP319_FINAL_EVIDENCE_DIGEST,
            "exp301_rescued": False,
            "exp319_reinterpreted": False,
            "gradient_updates": 0,
            "scale_change": False,
        },
        "source_evidence": {
            "exp319_run_id": EXP319_RUN_ID,
            "exp319_marker_sha": EXP319_MARKER_SHA,
            "exp319_source_sha": EXP319_SOURCE_SHA,
            "stage_a_selection_artifact_id": STAGE_A_SELECTION_ARTIFACT_ID,
            "stage_a_selection_zip_digest": STAGE_A_SELECTION_ZIP_DIGEST,
            "stage_a_selection_authority_digest": STAGE_A_SELECTION_AUTHORITY_DIGEST,
            "selected_checkpoint_artifact_id": SELECTED_CHECKPOINT_ARTIFACT_ID,
            "selected_checkpoint_zip_digest": SELECTED_CHECKPOINT_ZIP_DIGEST,
            "selected_receipt_artifact_digest": SELECTED_RECEIPT_ARTIFACT_DIGEST,
            "selected_model_state_digest": SELECTED_MODEL_STATE_DIGEST,
            "selected_arm": SELECTED_ARM,
            "selected_learning_rate": SELECTED_LEARNING_RATE,
            "selected_step": SELECTED_STEP,
        },
        "tokenizer": {
            "vocab_size": VOCAB_SIZE,
            "eos_id": EOS_ID,
            "byte_id_start": BYTE_ID_START,
            "byte_id_end_inclusive": BYTE_ID_END_INCLUSIVE,
            "unused_id_start": UNUSED_ID_START,
            "unused_id_end_inclusive": UNUSED_ID_END_INCLUSIVE,
        },
        "effort_grid": list(EFFORT_GRID),
        "measurements": [
            "teacher_forced_token_geometry",
            "greedy_rollout",
            "legal_vocab_masked_counterfactual",
            "family_and_answer_length_localization",
        ],
        "thresholds": asdict(THRESHOLDS),
        "reproduction_anchors": asdict(REPRODUCTION_ANCHORS),
        "dispositions": list(DISPOSITIONS),
        "authorizations": dict(AUTHORIZATION_FLAGS),
    }


def preregistration_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(preregistration_payload())).hexdigest()
