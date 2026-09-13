from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from nolane_ai.experiments.exp301_compute import (
    COMPUTE_LEDGER_VERSION,
    account_arm_flops,
    match_common_compute,
)
from nolane_ai.experiments.exp301_worlds import Exp301WorldInstance, verify_world_answer


EXP301_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EXP301_EFFORTS = (1, 2, 4, 8, 12, 16)
PRIMARY_TRAINED_EFFORTS = (1, 2, 4, 8)
UNSEEN_DEPTH_EFFORTS = (12, 16)
RESIDENT_PARAMETERS = 10_000_000
PREDICTION_SCHEMA = "exp301-prediction-commitment-v1"


@dataclass(frozen=True, slots=True)
class PredictionCommitment:
    schema: str
    arm_id: str
    root: int
    effort_multiplier: int
    content_id: str
    candidate_answer: str
    resident_parameters: int
    accounted_flops: int
    compute_ledger_version: str
    compute_match_status: str
    prediction_digest: str


@dataclass(frozen=True, slots=True)
class Exp301EvaluationRow:
    arm_id: str
    family: str
    root: int
    effort_multiplier: int
    content_id: str
    prediction_digest: str
    verified_success: bool
    invalid_output: bool
    resident_parameters: int
    accounted_flops: int
    compute_match_status: str
    unseen_depth_diagnostic: bool
    primary_family_gain_eligible: bool


@dataclass(frozen=True, slots=True)
class AggregateCell:
    arm_id: str
    family: str
    root: int
    effort_multiplier: int
    n: int
    verified_success: float
    invalid_output_rate: float
    accounted_flops: float


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _prediction_payload(commitment: PredictionCommitment) -> dict[str, object]:
    payload = asdict(commitment)
    payload.pop("prediction_digest", None)
    return payload


def validate_prediction_commitment(commitment: PredictionCommitment) -> None:
    if commitment.schema != PREDICTION_SCHEMA:
        raise ValueError("prediction commitment schema mismatch")
    if commitment.arm_id not in EXP301_ARMS:
        raise ValueError("prediction commitment arm mismatch")
    if commitment.effort_multiplier not in EXP301_EFFORTS:
        raise ValueError("prediction commitment effort mismatch")
    if commitment.resident_parameters != RESIDENT_PARAMETERS:
        raise ValueError("prediction commitment resident parameter mismatch")
    if commitment.compute_ledger_version != COMPUTE_LEDGER_VERSION:
        raise ValueError("prediction commitment compute ledger mismatch")
    expected = _canonical_digest(_prediction_payload(commitment))
    if commitment.prediction_digest != expected:
        raise ValueError(
            f"prediction commitment digest mismatch: expected {expected}, got {commitment.prediction_digest}"
        )


def commit_prediction(
    instance: Exp301WorldInstance,
    *,
    arm_id: str,
    root: int,
    effort_multiplier: int,
    candidate_answer: str,
) -> PredictionCommitment:
    if arm_id not in EXP301_ARMS:
        raise ValueError(f"arm must be one of {EXP301_ARMS}")
    if root != instance.root:
        raise ValueError("prediction root must match world root")
    if effort_multiplier not in EXP301_EFFORTS:
        raise ValueError(f"effort must be one of {EXP301_EFFORTS}")
    if not isinstance(candidate_answer, str):
        raise ValueError("candidate_answer must be a string")

    sequence_length = max(1, len(instance.model_input.encode("utf-8")))
    arm_receipt = account_arm_flops(
        arm_id,
        effort_multiplier=effort_multiplier,
        sequence_length=sequence_length,
    )
    match = match_common_compute(
        effort_multiplier=effort_multiplier,
        sequence_length=sequence_length,
    )
    without_digest = {
        "schema": PREDICTION_SCHEMA,
        "arm_id": arm_id,
        "root": root,
        "effort_multiplier": effort_multiplier,
        "content_id": instance.content_id,
        "candidate_answer": candidate_answer,
        "resident_parameters": RESIDENT_PARAMETERS,
        "accounted_flops": arm_receipt.total_flops,
        "compute_ledger_version": COMPUTE_LEDGER_VERSION,
        "compute_match_status": match.status,
    }
    return PredictionCommitment(
        **without_digest,
        prediction_digest=_canonical_digest(without_digest),
    )


def score_committed_prediction(
    commitment: PredictionCommitment,
    instance: Exp301WorldInstance,
) -> Exp301EvaluationRow:
    # The commitment is validated before the verifier is called. This preserves
    # the causal boundary: answer first, ground truth/verifier second.
    validate_prediction_commitment(commitment)
    if commitment.content_id != instance.content_id:
        raise ValueError("prediction content_id does not match world instance")
    if commitment.root != instance.root:
        raise ValueError("prediction root does not match world instance")

    invalid_output = commitment.candidate_answer.strip() == ""
    verified = False if invalid_output else verify_world_answer(
        instance,
        commitment.candidate_answer,
    )
    unseen = commitment.effort_multiplier in UNSEEN_DEPTH_EFFORTS
    return Exp301EvaluationRow(
        arm_id=commitment.arm_id,
        family=instance.family,
        root=commitment.root,
        effort_multiplier=commitment.effort_multiplier,
        content_id=instance.content_id,
        prediction_digest=commitment.prediction_digest,
        verified_success=verified,
        invalid_output=invalid_output,
        resident_parameters=commitment.resident_parameters,
        accounted_flops=commitment.accounted_flops,
        compute_match_status=commitment.compute_match_status,
        unseen_depth_diagnostic=unseen,
        primary_family_gain_eligible=not unseen,
    )


def aggregate_verified_success(
    rows: Iterable[Exp301EvaluationRow],
) -> dict[tuple[str, str, int, int], AggregateCell]:
    grouped: dict[tuple[str, str, int, int], list[Exp301EvaluationRow]] = {}
    for row in rows:
        key = (row.arm_id, row.family, row.root, row.effort_multiplier)
        grouped.setdefault(key, []).append(row)

    result: dict[tuple[str, str, int, int], AggregateCell] = {}
    for key, group in grouped.items():
        n = len(group)
        arm_id, family, root, effort = key
        result[key] = AggregateCell(
            arm_id=arm_id,
            family=family,
            root=root,
            effort_multiplier=effort,
            n=n,
            verified_success=sum(row.verified_success for row in group) / n,
            invalid_output_rate=sum(row.invalid_output for row in group) / n,
            accounted_flops=sum(row.accounted_flops for row in group) / n,
        )
    return result


def _mean_rate(rows: list[Exp301EvaluationRow], field: str) -> float | None:
    if not rows:
        return None
    return sum(float(getattr(row, field)) for row in rows) / len(rows)


def evaluate_protected_floors(
    rows: Iterable[Exp301EvaluationRow],
) -> dict[str, bool]:
    materialized = list(rows)
    resident_parameter_identity = bool(materialized) and all(
        row.resident_parameters == RESIDENT_PARAMETERS for row in materialized
    )
    compute_match = bool(materialized) and all(
        row.compute_match_status == "VALID_COMPUTE_MATCH" for row in materialized
    )

    a_language = [
        row for row in materialized
        if row.arm_id == "A_FIXED" and row.family == "language-sequence-control"
    ]
    c_language = [
        row for row in materialized
        if row.arm_id == "C_NRS_CORE" and row.family == "language-sequence-control"
    ]
    a_language_success = _mean_rate(a_language, "verified_success")
    c_language_success = _mean_rate(c_language, "verified_success")
    language_control = (
        a_language_success is not None
        and c_language_success is not None
        and (a_language_success - c_language_success) <= 0.02 + 1e-12
    )

    a_rows = [row for row in materialized if row.arm_id == "A_FIXED"]
    c_rows = [row for row in materialized if row.arm_id == "C_NRS_CORE"]
    a_invalid = _mean_rate(a_rows, "invalid_output")
    c_invalid = _mean_rate(c_rows, "invalid_output")
    invalid_output = (
        a_invalid is not None
        and c_invalid is not None
        and (c_invalid - a_invalid) <= 0.005 + 1e-12
    )

    floors = {
        "resident_parameter_identity": resident_parameter_identity,
        "compute_match": compute_match,
        "language_control": language_control,
        "invalid_output": invalid_output,
    }
    floors["all_clear"] = all(floors.values())
    return floors
