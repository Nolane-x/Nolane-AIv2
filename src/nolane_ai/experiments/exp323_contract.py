from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "EXP323-AFIXED-SECOND-DECAY-CONVERGENCE-INTERVENTION-V1"
EXPERIMENT_ID = "EXP-323"

PARENT_EXP322_RUN_ID = 35339004168
PARENT_EXP322_DISPOSITION = "PARTIAL_CONTINUATION_PROGRESS"
PARENT_EXP322_FINAL_ARTIFACT_ID = 10544985252
PARENT_EXP322_FINAL_ARTIFACT_ZIP_DIGEST = (
    "18eb23aa09086aea89bf9ab96e07c52270069b75e8dc276f7610148b46b2e4cc"
)
PARENT_EXP322_EVIDENCE_DIGEST = (
    "1e3ea213718d1612a6370c0fb4124c869733747dc9992ca58345d16dcfb5aee2"
)
PARENT_EXP322_EXECUTION_DIGEST = (
    "551d41405e26a12947869a9aaa94e09c19239a0de1da49571051873e51afe1ed"
)
PARENT_DECAY_ARM_EVIDENCE_DIGEST = (
    "78c4c44ae168a4a186ceb8040e239501804263e9b45737afdee25fb465c038ce"
)
PARENT_DECAY_MODEL_STATE_DIGEST = (
    "4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc"
)
PARENT_DECAY_OPTIMIZER_STATE_DIGEST = (
    "9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79"
)
PARENT_DECAY_RNG_STATE_DIGEST = (
    "e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07"
)
PARENT_EXP322_MARKER_COMMIT_SHA = "4a7f079a97f2efeeb4682abe05246e893d30c4ad"
PARENT_EXP322_SOURCE_COMMIT_SHA = "270a33508d72fa753318a891127d50677f7c4b98"

SOURCE_CHECKPOINT_RUN_ID = 35311529822
SOURCE_CHECKPOINT_ARTIFACT_ID = 10534351390
SOURCE_CHECKPOINT_NAME = "exp319-stage-a-A_FIXED-0.0001-c3"
SOURCE_CHECKPOINT_ZIP_DIGEST = (
    "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3"
)
SOURCE_RECEIPT_ARTIFACT_DIGEST = (
    "d874545fa677569e30849128df337e27823d8aa4c5335adb9cb422968096a280"
)
SOURCE_MODEL_STATE_DIGEST = (
    "18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb"
)

ARMS = ("HOLD_5E5", "DECAY_2P5E5")
ARM_LEARNING_RATES = {"HOLD_5E5": 5e-5, "DECAY_2P5E5": 2.5e-5}
CHECKPOINTS = (2304, 2560, 3072)
STARTING_STEP = 2048
MAX_ADDITIONAL_STEPS = 1024

BASELINE_TOKEN_ACCURACY = 0.9577464788732394
BASELINE_FULL_EXACT = 0.6875
BASELINE_GREEDY_EXACT = 0.6875
BASELINE_ANSWER_ONLY_LOSS = 0.4435557168891316

DISPOSITIONS = (
    "INVALID_INTERVENTION",
    "REDUCED_LR_BUDGET_SUFFICIENT",
    "SECOND_DECAY_SUFFICIENT",
    "CONTINUATION_PROGRESS",
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
class Exp323Thresholds:
    teacher_forced_token_accuracy_floor: float = 0.99
    teacher_forced_full_answer_exact_floor: float = 0.90
    partial_token_accuracy_gain_min: float = 0.02
    partial_full_answer_exact_gain_min: float = 0.125
    nonfinite_events_max: int = 0


THRESHOLDS = Exp323Thresholds()


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
    return {"schema_version":"EXP323-AFIXED-SECOND-DECAY-CONVERGENCE-INTERVENTION-V1","experiment_id":"EXP-323","role":"post_exp322_second_decay_convergence_intervention","scientific_boundary":{"parent_exp322_run_id":35339004168,"parent_exp322_disposition":"PARTIAL_CONTINUATION_PROGRESS","parent_exp322_final_artifact_id":10544985252,"parent_exp322_final_artifact_zip_digest":"18eb23aa09086aea89bf9ab96e07c52270069b75e8dc276f7610148b46b2e4cc","parent_exp322_evidence_digest":"1e3ea213718d1612a6370c0fb4124c869733747dc9992ca58345d16dcfb5aee2","parent_exp322_execution_digest":"551d41405e26a12947869a9aaa94e09c19239a0de1da49571051873e51afe1ed","parent_exp322_decay_arm_evidence_digest":"78c4c44ae168a4a186ceb8040e239501804263e9b45737afdee25fb465c038ce","parent_exp322_decay_model_state_digest":"4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc","parent_exp322_decay_optimizer_state_digest":"9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79","parent_exp322_decay_rng_state_digest":"e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07","parent_exp322_marker_commit_sha":"4a7f079a97f2efeeb4682abe05246e893d30c4ad","parent_exp322_source_commit_sha":"270a33508d72fa753318a891127d50677f7c4b98","architecture_change":False,"tokenizer_change":False,"dataset_change":False,"objective_change":False,"scale_change":False},"reconstruction_authority":{"source_checkpoint_run_id":35311529822,"source_checkpoint_artifact_id":10534351390,"source_checkpoint_name":"exp319-stage-a-A_FIXED-0.0001-c3","source_checkpoint_zip_digest":"ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3","source_receipt_artifact_digest":"d874545fa677569e30849128df337e27823d8aa4c5335adb9cb422968096a280","source_model_state_digest":"18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb","replay_arm":"DECAY_5E5","replay_start_step":1024,"replay_end_step":2048,"replay_learning_rate_decimal":"0.00005","exact_replay_required":True},"population":{"root":0,"families":4,"examples_per_family":8,"total_examples":32,"population_role":"tuning_sanity_only"},"model":{"arm":"A_FIXED","resident_trainable_parameters":10000000,"starting_cumulative_step":2048,"device":"cpu"},"training_invariants":{"optimizer":"AdamW","optimizer_state_reused":True,"rng_state_reused":True,"weight_decay_decimal":"0.01","gradient_clip_norm_decimal":"1.0","training_effort_cycle":[1,2,4,8],"byte_tokenizer_unchanged":True,"answer_only_loss_unchanged":True,"world_order_unchanged":True},"intervention_arms":{"HOLD_5E5":{"learning_rate_decimal":"0.00005","description":"continue the exactly reconstructed EXP-322 DECAY state at the inherited reduced learning rate"},"DECAY_2P5E5":{"learning_rate_decimal":"0.000025","description":"clone the same reconstructed state and halve every AdamW param-group learning rate to 2.5e-5 before the first post-2048 update"}},"checkpoints":[2304,2560,3072],"max_additional_optimizer_steps_per_arm":1024,"primary_measurements":["teacher_forced_answer_token_accuracy","teacher_forced_full_answer_exact"],"secondary_measurements":["greedy_exact","eos_correctness","invalid_output_rate","answer_only_loss","gradient_norm_preclip","parameter_update_norm_ratio","nonfinite_events","family_teacher_forced_token_accuracy","family_teacher_forced_full_answer_exact"],"baseline":{"teacher_forced_answer_token_accuracy":0.9577464788732394,"teacher_forced_full_answer_exact":0.6875,"greedy_exact":0.6875,"answer_only_loss":0.4435557168891316},"thresholds":{"teacher_forced_token_accuracy_floor":0.99,"teacher_forced_full_answer_exact_floor":0.9,"partial_token_accuracy_gain_min":0.02,"partial_full_answer_exact_gain_min":0.125,"nonfinite_events_max":0},"dispositions":["INVALID_INTERVENTION","REDUCED_LR_BUDGET_SUFFICIENT","SECOND_DECAY_SUFFICIENT","CONTINUATION_PROGRESS","NO_REGISTERED_RESCUE"],"decision_order":["INVALID_INTERVENTION","REDUCED_LR_BUDGET_SUFFICIENT","SECOND_DECAY_SUFFICIENT","CONTINUATION_PROGRESS","NO_REGISTERED_RESCUE"],"authorization":{"exp302_implementation_authorized":False,"exp320_implementation_authorized":False,"scale_authorized":False,"authorized_30m":False,"authorized_100m":False},"nonclaims":["EXP-323 does not reinterpret EXP-301 EXP-319 EXP-321 or EXP-322","EXP-323 cannot establish architectural capacity limits","EXP-323 cannot establish scaling benefit","EXP-323 cannot authorize EXP-320 or larger model sizes","failure of both registered continuation arms only means no registered post-2048 continuation rescue was demonstrated"]}


def preregistration_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(preregistration_payload())).hexdigest()


def _unit(value: float, *, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0,1]")


def validate_snapshot(snapshot: InterventionSnapshot) -> None:
    if snapshot.arm not in ARMS:
        raise ValueError("unknown EXP-323 arm")
    if snapshot.step not in CHECKPOINTS:
        raise ValueError("unknown EXP-323 checkpoint")
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

    hold_pass = any(teacher_floor_pass(item) for item in checked["HOLD_5E5"])
    decay_pass = any(teacher_floor_pass(item) for item in checked["DECAY_2P5E5"])

    if hold_pass:
        return "REDUCED_LR_BUDGET_SUFFICIENT"
    if decay_pass:
        return "SECOND_DECAY_SUFFICIENT"

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
        return "CONTINUATION_PROGRESS"
    return "NO_REGISTERED_RESCUE"
