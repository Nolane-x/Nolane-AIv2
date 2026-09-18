from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "EXP322-AFIXED-TEACHER-FORCED-LEARNABILITY-INTERVENTION-V1"
EXPERIMENT_ID = "EXP-322"

PARENT_EXP321_RUN_ID = 35331243762
PARENT_EXP321_DISPOSITION = "TEACHER_FORCED_FOUNDATION_INSUFFICIENT"
PARENT_EXP321_EVIDENCE_DIGEST = (
    "5f3bd6cbacef0af4883e33f14bafd56b49a307df88641e837bb9ecdf004d5091"
)
PARENT_EXP321_ARTIFACT_ID = 10541366011
PARENT_CHECKPOINT_ARTIFACT_ID = 10534351390
PARENT_CHECKPOINT_ZIP_DIGEST = (
    "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3"
)
PARENT_RECEIPT_ARTIFACT_DIGEST = (
    "d874545fa677569e30849128df337e27823d8aa4c5335adb9cb422968096a280"
)
PARENT_MODEL_STATE_DIGEST = (
    "18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb"
)

ARMS = ("HOLD_1E4", "DECAY_5E5")
ARM_LEARNING_RATES = {"HOLD_1E4": 1e-4, "DECAY_5E5": 5e-5}
CHECKPOINTS = (1280, 1536, 2048)
STARTING_STEP = 1024
MAX_ADDITIONAL_STEPS = 1024
BASELINE_TOKEN_ACCURACY = 0.7711267605633803
BASELINE_FULL_EXACT = 0.28125
BASELINE_GREEDY_EXACT = 0.28125

DISPOSITIONS = (
    "INVALID_INTERVENTION",
    "BUDGET_INSUFFICIENCY_EVIDENT",
    "LR_SCHEDULE_INSUFFICIENCY_EVIDENT",
    "PARTIAL_CONTINUATION_PROGRESS",
    "NO_REGISTERED_RESCUE",
)

AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}


@dataclass(frozen=True, slots=True)
class Exp322Thresholds:
    teacher_forced_token_accuracy_floor: float = 0.99
    teacher_forced_full_answer_exact_floor: float = 0.90
    partial_token_accuracy_gain_min: float = 0.10
    partial_full_answer_exact_gain_min: float = 0.25
    nonfinite_events_max: int = 0


THRESHOLDS = Exp322Thresholds()


@dataclass(frozen=True, slots=True)
class InterventionSnapshot:
    arm: str
    step: int
    teacher_forced_answer_token_accuracy: float
    teacher_forced_full_answer_exact: float
    greedy_exact: float
    eos_correctness: float
    invalid_output_rate: float
    answer_only_loss: float
    gradient_norm_preclip: float
    parameter_update_norm_ratio: float
    nonfinite_events: int


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
        "role": "post_exp321_teacher_forced_learnability_intervention",
        "scientific_boundary": {
            "parent_exp321_run_id": PARENT_EXP321_RUN_ID,
            "parent_exp321_disposition": PARENT_EXP321_DISPOSITION,
            "parent_exp321_evidence_digest": PARENT_EXP321_EVIDENCE_DIGEST,
            "parent_exp321_artifact_id": PARENT_EXP321_ARTIFACT_ID,
            "parent_exp319_checkpoint_artifact_id": PARENT_CHECKPOINT_ARTIFACT_ID,
            "parent_exp319_checkpoint_zip_digest": PARENT_CHECKPOINT_ZIP_DIGEST,
            "parent_exp319_receipt_artifact_digest": PARENT_RECEIPT_ARTIFACT_DIGEST,
            "parent_exp319_model_state_digest": PARENT_MODEL_STATE_DIGEST,
            "architecture_change": False,
            "tokenizer_change": False,
            "dataset_change": False,
            "objective_change": False,
            "scale_change": False,
        },
        "population": {
            "root": 0,
            "families": 4,
            "examples_per_family": 8,
            "total_examples": 32,
            "population_role": "tuning_sanity_only",
        },
        "model": {
            "arm": "A_FIXED",
            "resident_trainable_parameters": 10_000_000,
            "starting_cumulative_step": STARTING_STEP,
            "device": "cpu",
        },
        "training_invariants": {
            "optimizer": "AdamW",
            "optimizer_state_reused": True,
            "rng_state_reused": True,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "training_effort_cycle": [1, 2, 4, 8],
            "byte_tokenizer_unchanged": True,
            "answer_only_loss_unchanged": True,
            "world_order_unchanged": True,
        },
        "intervention_arms": {
            "HOLD_1E4": {
                "learning_rate": 1e-4,
                "description": (
                    "continue exact selected checkpoint with inherited constant learning rate"
                ),
            },
            "DECAY_5E5": {
                "learning_rate": 5e-5,
                "description": (
                    "continue same checkpoint and optimizer moments after setting all "
                    "AdamW param-group learning rates to 5e-5"
                ),
            },
        },
        "checkpoints": list(CHECKPOINTS),
        "max_additional_optimizer_steps_per_arm": MAX_ADDITIONAL_STEPS,
        "primary_measurements": [
            "teacher_forced_answer_token_accuracy",
            "teacher_forced_full_answer_exact",
        ],
        "secondary_measurements": [
            "greedy_exact",
            "eos_correctness",
            "invalid_output_rate",
            "answer_only_loss",
            "gradient_norm_preclip",
            "parameter_update_norm_ratio",
            "nonfinite_events",
            "family_teacher_forced_token_accuracy",
            "family_teacher_forced_full_answer_exact",
        ],
        "baseline": {
            "teacher_forced_answer_token_accuracy": BASELINE_TOKEN_ACCURACY,
            "teacher_forced_full_answer_exact": BASELINE_FULL_EXACT,
            "greedy_exact": BASELINE_GREEDY_EXACT,
        },
        "thresholds": asdict(THRESHOLDS),
        "dispositions": list(DISPOSITIONS),
        "decision_order": list(DISPOSITIONS),
        "authorization": dict(AUTHORIZATION_FLAGS),
        "nonclaims": [
            "EXP-322 does not rescue or reinterpret EXP-301 or EXP-319",
            "EXP-322 cannot establish architectural capacity limits",
            "EXP-322 cannot establish scaling benefit",
            "EXP-322 cannot authorize EXP-320 or larger model sizes",
            (
                "failure of both registered continuation arms only means no "
                "registered continuation rescue was demonstrated"
            ),
        ],
    }


def preregistration_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(preregistration_payload())).hexdigest()


def _unit(value: float, *, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0,1]")


def validate_snapshot(snapshot: InterventionSnapshot) -> None:
    if snapshot.arm not in ARMS:
        raise ValueError("unknown EXP-322 arm")
    if snapshot.step not in CHECKPOINTS:
        raise ValueError("unknown EXP-322 checkpoint")
    for field in (
        "teacher_forced_answer_token_accuracy",
        "teacher_forced_full_answer_exact",
        "greedy_exact",
        "eos_correctness",
        "invalid_output_rate",
    ):
        _unit(getattr(snapshot, field), field=field)
    for field in (
        "answer_only_loss",
        "gradient_norm_preclip",
        "parameter_update_norm_ratio",
    ):
        value = getattr(snapshot, field)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    if (
        not isinstance(snapshot.nonfinite_events, int)
        or snapshot.nonfinite_events < 0
        or snapshot.nonfinite_events > THRESHOLDS.nonfinite_events_max
    ):
        raise ValueError("nonfinite_events exceeds registered maximum")


def teacher_floor_pass(snapshot: InterventionSnapshot) -> bool:
    return (
        snapshot.teacher_forced_answer_token_accuracy
        >= THRESHOLDS.teacher_forced_token_accuracy_floor
        and snapshot.teacher_forced_full_answer_exact
        >= THRESHOLDS.teacher_forced_full_answer_exact_floor
    )


def _validated_arm_records(
    records: Mapping[str, Sequence[InterventionSnapshot]],
) -> dict[str, tuple[InterventionSnapshot, ...]] | None:
    if set(records) != set(ARMS):
        return None
    materialized: dict[str, tuple[InterventionSnapshot, ...]] = {}
    try:
        for arm in ARMS:
            rows = tuple(records[arm])
            if len(rows) != len(CHECKPOINTS):
                return None
            if tuple(item.step for item in rows) != CHECKPOINTS:
                return None
            if any(item.arm != arm for item in rows):
                return None
            for item in rows:
                validate_snapshot(item)
            materialized[arm] = rows
    except (TypeError, ValueError):
        return None
    return materialized


def reduce_intervention(
    records: Mapping[str, Sequence[InterventionSnapshot]],
) -> str:
    checked = _validated_arm_records(records)
    if checked is None:
        return "INVALID_INTERVENTION"

    hold_pass = any(teacher_floor_pass(item) for item in checked["HOLD_1E4"])
    decay_pass = any(teacher_floor_pass(item) for item in checked["DECAY_5E5"])

    if hold_pass:
        return "BUDGET_INSUFFICIENCY_EVIDENT"
    if decay_pass:
        return "LR_SCHEDULE_INSUFFICIENCY_EVIDENT"

    best_token = max(
        item.teacher_forced_answer_token_accuracy
        for rows in checked.values()
        for item in rows
    )
    best_full = max(
        item.teacher_forced_full_answer_exact
        for rows in checked.values()
        for item in rows
    )
    if (
        best_token - BASELINE_TOKEN_ACCURACY
        >= THRESHOLDS.partial_token_accuracy_gain_min
        or best_full - BASELINE_FULL_EXACT
        >= THRESHOLDS.partial_full_answer_exact_gain_min
    ):
        return "PARTIAL_CONTINUATION_PROGRESS"
    return "NO_REGISTERED_RESCUE"
