from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .exp301_scientific import build_scientific_arm
from .exp319_training import (
    load_checkpoint_bundle,
    model_state_digest,
    optimizer_state_digest,
    rng_state_digest,
)
from .exp319_worlds import materialize_stage_a
from .exp322_contract import canonical_json_bytes as exp322_canonical_json_bytes
from .exp322_runtime import (
    _evaluate as _evaluate_exp322,
    _train_one_step,
    apply_intervention_learning_rate,
    validate_parent_receipt,
)
from .exp323_contract import (
    ARM_LEARNING_RATES,
    ARMS,
    AUTHORIZATION_FLAGS,
    BASELINE_ANSWER_ONLY_LOSS,
    BASELINE_FULL_EXACT,
    BASELINE_GREEDY_EXACT,
    BASELINE_TOKEN_ACCURACY,
    CHECKPOINTS,
    PARENT_DECAY_ARM_EVIDENCE_DIGEST,
    PARENT_DECAY_MODEL_STATE_DIGEST,
    PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
    PARENT_DECAY_RNG_STATE_DIGEST,
    PARENT_EXP322_DISPOSITION,
    PARENT_EXP322_EVIDENCE_DIGEST,
    PARENT_EXP322_EXECUTION_DIGEST,
    STARTING_STEP,
    InterventionSnapshot,
)
from .exp323_evidence import build_arm_evidence, expected_reconstruction_payload
from .exp323_identity import Exp323ExecutionIdentity, validate_exp323_execution_identity


EXPECTED_PARAMETER_COUNT = 10_000_000
REPLAY_START_STEP = 1024
REPLAY_END_STEP = 2048
REPLAY_ARM = "DECAY_5E5"
REPLAY_LEARNING_RATE = 5e-5


@dataclass(slots=True)
class ReconstructedState:
    compiled: object
    optimizer: torch.optim.Optimizer
    reconstruction: dict[str, Any]


def validate_exp322_parent_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP322-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-323 parent evidence schema mismatch")
    if payload.get("decision") != PARENT_EXP322_DISPOSITION:
        raise ValueError("EXP-323 parent disposition mismatch")
    if payload.get("evidence_digest") != PARENT_EXP322_EVIDENCE_DIGEST:
        raise ValueError("EXP-323 parent evidence digest mismatch")

    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    recomputed = hashlib.sha256(exp322_canonical_json_bytes(materialized)).hexdigest()
    if recomputed != PARENT_EXP322_EVIDENCE_DIGEST:
        raise ValueError("EXP-323 parent evidence canonical digest mismatch")

    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("EXP-323 parent execution identity missing")
    if identity.get("exp322_execution_digest") != PARENT_EXP322_EXECUTION_DIGEST:
        raise ValueError("EXP-323 parent execution identity mismatch")

    arm_digests = payload.get("arm_evidence_digests")
    if not isinstance(arm_digests, Mapping):
        raise ValueError("EXP-323 parent arm evidence digests missing")
    if arm_digests.get("DECAY_5E5") != PARENT_DECAY_ARM_EVIDENCE_DIGEST:
        raise ValueError("EXP-323 parent DECAY arm evidence mismatch")

    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get("DECAY_5E5"), Mapping):
        raise ValueError("EXP-323 parent DECAY arm missing")
    decay = arms["DECAY_5E5"]
    if decay.get("completed_step") != STARTING_STEP or decay.get("invalid_reason") is not None:
        raise ValueError("EXP-323 parent DECAY completion mismatch")
    final_state = decay.get("final_state")
    expected_state = {
        "model_state_digest": PARENT_DECAY_MODEL_STATE_DIGEST,
        "optimizer_state_digest": PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
        "rng_state_digest": PARENT_DECAY_RNG_STATE_DIGEST,
    }
    if final_state != expected_state:
        raise ValueError("EXP-323 parent DECAY final-state mismatch")

    snapshots = decay.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("EXP-323 parent DECAY snapshots missing")
    last = snapshots[-1]
    expected_baseline = {
        "step": STARTING_STEP,
        "teacher_forced_answer_token_accuracy": BASELINE_TOKEN_ACCURACY,
        "teacher_forced_full_answer_exact": BASELINE_FULL_EXACT,
        "greedy_exact": BASELINE_GREEDY_EXACT,
        "answer_only_loss": BASELINE_ANSWER_ONLY_LOSS,
    }
    for key, value in expected_baseline.items():
        if last.get(key) != value:
            raise ValueError(f"EXP-323 parent DECAY baseline mismatch: {key}")

    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-323 parent forbidden authorization drift: {key}")


def validate_reconstruction_digests(
    *,
    model_digest: str,
    optimizer_digest: str,
    rng_digest: str,
    nonfinite_events: int,
) -> dict[str, Any]:
    observed = {
        "verified": True,
        "completed_step": STARTING_STEP,
        "model_state_digest": model_digest,
        "optimizer_state_digest": optimizer_digest,
        "rng_state_digest": rng_digest,
        "nonfinite_events": nonfinite_events,
    }
    if observed != expected_reconstruction_payload():
        raise ValueError("EXP-323 exact step-2048 replay mismatch")
    return observed


def apply_post2048_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
) -> None:
    if arm not in ARMS:
        raise ValueError("unknown EXP-323 arm")
    if not optimizer.param_groups:
        raise ValueError("EXP-323 optimizer has no parameter groups")
    for group in optimizer.param_groups:
        if float(group.get("lr", math.nan)) != REPLAY_LEARNING_RATE:
            raise ValueError("EXP-323 reconstructed optimizer LR is not 5e-5")
        if float(group.get("weight_decay", math.nan)) != 0.01:
            raise ValueError("EXP-323 reconstructed optimizer weight decay drift")
    target = ARM_LEARNING_RATES[arm]
    if arm == "DECAY_2P5E5":
        for group in optimizer.param_groups:
            group["lr"] = target
    for group in optimizer.param_groups:
        if float(group["lr"]) != target:
            raise ValueError("EXP-323 intervention learning rate was not applied exactly")


def reconstruct_exp322_decay_state(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
) -> ReconstructedState:
    bundle = load_checkpoint_bundle(checkpoint_path, receipt_path)
    validate_parent_receipt(bundle.receipt)

    torch.manual_seed(bundle.receipt.model_init_seed)
    compiled = build_scientific_arm("A_FIXED", device="cpu")
    model = getattr(compiled, "model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("EXP-323 resident trainable-parameter count drift")
    model.load_state_dict(bundle.model_state_dict)
    if model_state_digest(model) != bundle.receipt.model_state_digest:
        raise ValueError("EXP-323 source model-state digest mismatch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer.load_state_dict(bundle.optimizer_state_dict)
    if optimizer_state_digest(optimizer) != bundle.receipt.optimizer_state_digest:
        raise ValueError("EXP-323 source optimizer-state digest mismatch")
    apply_intervention_learning_rate(optimizer, arm=REPLAY_ARM)

    torch.set_rng_state(bundle.torch_rng_state)
    if rng_state_digest() != bundle.receipt.rng_state_digest:
        raise ValueError("EXP-323 source RNG-state digest mismatch")

    worlds = tuple(materialize_stage_a())
    if len(worlds) != 32:
        raise ValueError("EXP-323 requires exactly 32 Stage-A worlds")
    identities = tuple((world.family, world.content_id) for world in worlds)
    if len(set(identities)) != 32:
        raise ValueError("EXP-323 Stage-A world identities must be unique")

    cumulative_nonfinite = 0
    for global_step in range(REPLAY_START_STEP, REPLAY_END_STEP):
        world = worlds[global_step % len(worlds)]
        _, _, _, nonfinite = _train_one_step(
            compiled,
            world,
            optimizer=optimizer,
            global_step=global_step,
        )
        cumulative_nonfinite += nonfinite
        if cumulative_nonfinite:
            raise ValueError("EXP-323 replay encountered non-finite state")

    reconstruction = validate_reconstruction_digests(
        model_digest=model_state_digest(model),
        optimizer_digest=optimizer_state_digest(optimizer),
        rng_digest=rng_state_digest(),
        nonfinite_events=cumulative_nonfinite,
    )
    return ReconstructedState(
        compiled=compiled,
        optimizer=optimizer,
        reconstruction=reconstruction,
    )


def run_intervention_arm(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    parent_exp322_path: str | Path,
    arm: str,
    execution_identity: Exp323ExecutionIdentity,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown EXP-323 arm")
    validate_exp323_execution_identity(execution_identity)

    parent_payload = json.loads(Path(parent_exp322_path).read_text(encoding="utf-8"))
    if not isinstance(parent_payload, Mapping):
        raise ValueError("EXP-323 parent evidence must be a JSON object")
    validate_exp322_parent_evidence(parent_payload)

    reconstructed = reconstruct_exp322_decay_state(
        checkpoint_path=checkpoint_path,
        receipt_path=receipt_path,
    )
    compiled = reconstructed.compiled
    optimizer = reconstructed.optimizer
    apply_post2048_learning_rate(optimizer, arm=arm)

    worlds = tuple(materialize_stage_a())
    snapshots: list[InterventionSnapshot] = []
    family_summaries: dict[str, Any] = {}
    cumulative_nonfinite = 0
    last_grad = 0.0
    last_update = 0.0
    completed_step = STARTING_STEP
    invalid_reason: str | None = None

    for global_step in range(STARTING_STEP, CHECKPOINTS[-1]):
        world = worlds[global_step % len(worlds)]
        _, last_grad, last_update, nonfinite = _train_one_step(
            compiled,
            world,
            optimizer=optimizer,
            global_step=global_step,
        )
        cumulative_nonfinite += nonfinite
        completed_step = global_step + 1
        if cumulative_nonfinite:
            invalid_reason = "NONFINITE_EVENT"
            break
        if completed_step in CHECKPOINTS:
            legacy, family = _evaluate_exp322(
                compiled,
                arm=arm,
                step=completed_step,
                gradient_norm_preclip=last_grad,
                parameter_update_norm_ratio_value=last_update,
                nonfinite_events=cumulative_nonfinite,
            )
            snapshots.append(InterventionSnapshot(**asdict(legacy)))
            family_summaries[str(completed_step)] = family

    model = getattr(compiled, "model")
    return build_arm_evidence(
        arm=arm,
        execution_identity=execution_identity,
        reconstruction=reconstructed.reconstruction,
        snapshots=tuple(snapshots),
        family_summaries=family_summaries,
        completed_step=completed_step,
        final_model_state_digest=model_state_digest(model),
        final_optimizer_state_digest=optimizer_state_digest(optimizer),
        final_rng_state_digest=rng_state_digest(),
        invalid_reason=invalid_reason,
    )
