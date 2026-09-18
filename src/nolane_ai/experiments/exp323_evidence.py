from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Mapping

from .exp323_contract import (
    ARM_LEARNING_RATES,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHECKPOINTS,
    PARENT_DECAY_ARM_EVIDENCE_DIGEST,
    PARENT_DECAY_MODEL_STATE_DIGEST,
    PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
    PARENT_DECAY_RNG_STATE_DIGEST,
    PARENT_EXP322_EVIDENCE_DIGEST,
    PARENT_EXP322_FINAL_ARTIFACT_ID,
    PARENT_EXP322_FINAL_ARTIFACT_ZIP_DIGEST,
    PARENT_EXP322_RUN_ID,
    SOURCE_CHECKPOINT_ARTIFACT_ID,
    SOURCE_CHECKPOINT_ZIP_DIGEST,
    STARTING_STEP,
    InterventionSnapshot,
    canonical_json_bytes,
    reduce_intervention,
    validate_snapshot,
)
from .exp323_identity import (
    Exp323ExecutionIdentity,
    validate_exp323_execution_identity,
)


ARM_EVIDENCE_SCHEMA = "EXP323-ARM-EVIDENCE-V1"
FINAL_EVIDENCE_SCHEMA = "EXP323-FINAL-EVIDENCE-V1"


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def parent_authority_payload() -> dict[str, Any]:
    return {
        "exp322_run_id": PARENT_EXP322_RUN_ID,
        "exp322_final_artifact_id": PARENT_EXP322_FINAL_ARTIFACT_ID,
        "exp322_final_artifact_zip_digest": PARENT_EXP322_FINAL_ARTIFACT_ZIP_DIGEST,
        "exp322_evidence_digest": PARENT_EXP322_EVIDENCE_DIGEST,
        "exp322_decay_arm_evidence_digest": PARENT_DECAY_ARM_EVIDENCE_DIGEST,
        "exp322_decay_model_state_digest": PARENT_DECAY_MODEL_STATE_DIGEST,
        "exp322_decay_optimizer_state_digest": PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
        "exp322_decay_rng_state_digest": PARENT_DECAY_RNG_STATE_DIGEST,
        "source_checkpoint_artifact_id": SOURCE_CHECKPOINT_ARTIFACT_ID,
        "source_checkpoint_zip_digest": SOURCE_CHECKPOINT_ZIP_DIGEST,
    }


def expected_reconstruction_payload() -> dict[str, Any]:
    return {
        "verified": True,
        "completed_step": STARTING_STEP,
        "model_state_digest": PARENT_DECAY_MODEL_STATE_DIGEST,
        "optimizer_state_digest": PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
        "rng_state_digest": PARENT_DECAY_RNG_STATE_DIGEST,
        "nonfinite_events": 0,
    }


def build_arm_evidence(
    *,
    arm: str,
    execution_identity: Exp323ExecutionIdentity,
    reconstruction: Mapping[str, Any],
    snapshots: tuple[InterventionSnapshot, ...],
    family_summaries: Mapping[str, Any],
    completed_step: int,
    final_model_state_digest: str,
    final_optimizer_state_digest: str,
    final_rng_state_digest: str,
    invalid_reason: str | None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown EXP-323 arm")
    validate_exp323_execution_identity(execution_identity)
    payload: dict[str, Any] = {
        "schema": ARM_EVIDENCE_SCHEMA,
        "arm": arm,
        "learning_rate": ARM_LEARNING_RATES[arm],
        "starting_step": STARTING_STEP,
        "parent": parent_authority_payload(),
        "execution_identity": asdict(execution_identity),
        "reconstruction": dict(reconstruction),
        "snapshots": [asdict(item) for item in snapshots],
        "family_summaries": dict(family_summaries),
        "completed_step": completed_step,
        "final_state": {
            "model_state_digest": final_model_state_digest,
            "optimizer_state_digest": final_optimizer_state_digest,
            "rng_state_digest": final_rng_state_digest,
        },
        "invalid_reason": invalid_reason,
        **AUTHORIZATION_FLAGS,
    }
    validate_arm_evidence(payload)
    payload["arm_evidence_digest"] = _digest(payload)
    return payload


def validate_arm_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != ARM_EVIDENCE_SCHEMA:
        raise ValueError("EXP-323 arm evidence schema mismatch")
    arm = payload.get("arm")
    if arm not in ARMS:
        raise ValueError("EXP-323 arm evidence arm mismatch")
    if payload.get("learning_rate") != ARM_LEARNING_RATES[arm]:
        raise ValueError("EXP-323 arm learning-rate mismatch")
    if payload.get("starting_step") != STARTING_STEP:
        raise ValueError("EXP-323 arm starting-step mismatch")
    if payload.get("parent") != parent_authority_payload():
        raise ValueError("EXP-323 parent authority mismatch")
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-323 reconstruction authority mismatch")

    identity_raw = payload.get("execution_identity")
    if not isinstance(identity_raw, Mapping):
        raise ValueError("EXP-323 execution identity must be an object")
    try:
        identity = Exp323ExecutionIdentity(**identity_raw)
        validate_exp323_execution_identity(identity)
    except (TypeError, ValueError) as exc:
        raise ValueError("EXP-323 execution identity is invalid") from exc

    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-323 forbidden authorization drift: {key}")

    final_state = payload.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("EXP-323 final state must be an object")
    for key in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
        value = final_state.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"EXP-323 malformed final-state digest: {key}")

    snapshots_raw = payload.get("snapshots")
    if not isinstance(snapshots_raw, list):
        raise ValueError("EXP-323 snapshots must be an array")
    snapshots = tuple(InterventionSnapshot(**item) for item in snapshots_raw)
    for item in snapshots:
        validate_snapshot(item)
        if item.arm != arm:
            raise ValueError("EXP-323 snapshot arm mismatch")

    invalid_reason = payload.get("invalid_reason")
    if invalid_reason is None:
        if tuple(item.step for item in snapshots) != CHECKPOINTS:
            raise ValueError("valid EXP-323 arm must contain all registered checkpoints")
        if payload.get("completed_step") != CHECKPOINTS[-1]:
            raise ValueError("valid EXP-323 arm must complete step 3072")
    else:
        if not isinstance(invalid_reason, str) or not invalid_reason:
            raise ValueError("invalid_reason must be null or a non-empty string")
        completed = payload.get("completed_step")
        if not isinstance(completed, int) or not STARTING_STEP <= completed <= CHECKPOINTS[-1]:
            raise ValueError("invalid EXP-323 completed_step is malformed")


def validate_arm_evidence_digest(payload: Mapping[str, Any]) -> None:
    expected = payload.get("arm_evidence_digest")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("EXP-323 arm evidence digest is malformed")
    materialized = dict(payload)
    materialized.pop("arm_evidence_digest", None)
    validate_arm_evidence(materialized)
    if _digest(materialized) != expected:
        raise ValueError("EXP-323 arm evidence digest mismatch")


def build_final_evidence(
    hold_payload: Mapping[str, Any],
    decay_payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_arm_evidence_digest(hold_payload)
    validate_arm_evidence_digest(decay_payload)
    by_arm = {
        str(hold_payload["arm"]): hold_payload,
        str(decay_payload["arm"]): decay_payload,
    }
    if set(by_arm) != set(ARMS):
        raise ValueError("EXP-323 final evidence requires one artifact per registered arm")
    hold_identity = by_arm["HOLD_5E5"].get("execution_identity")
    decay_identity = by_arm["DECAY_2P5E5"].get("execution_identity")
    if hold_identity != decay_identity:
        raise ValueError("EXP-323 arm execution identities do not match")
    if by_arm["HOLD_5E5"].get("reconstruction") != by_arm["DECAY_2P5E5"].get("reconstruction"):
        raise ValueError("EXP-323 arm reconstruction receipts do not match")

    invalid = any(by_arm[arm].get("invalid_reason") is not None for arm in ARMS)
    records = {
        arm: tuple(InterventionSnapshot(**item) for item in by_arm[arm]["snapshots"])
        for arm in ARMS
    }
    decision = "INVALID_INTERVENTION" if invalid else reduce_intervention(records)

    payload: dict[str, Any] = {
        "schema": FINAL_EVIDENCE_SCHEMA,
        "parent": parent_authority_payload(),
        "execution_identity": hold_identity,
        "reconstruction": by_arm["HOLD_5E5"]["reconstruction"],
        "arm_evidence_digests": {
            arm: by_arm[arm]["arm_evidence_digest"] for arm in ARMS
        },
        "decision": decision,
        "arms": {
            arm: {
                "learning_rate": ARM_LEARNING_RATES[arm],
                "snapshots": by_arm[arm]["snapshots"],
                "family_summaries": by_arm[arm]["family_summaries"],
                "completed_step": by_arm[arm]["completed_step"],
                "final_state": by_arm[arm]["final_state"],
                "invalid_reason": by_arm[arm]["invalid_reason"],
            }
            for arm in ARMS
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    return payload
