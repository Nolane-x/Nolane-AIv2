from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .exp301_scientific import build_scientific_arm
from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp319_worlds import materialize_stage_a
from .exp322_runtime import _evaluate as _evaluate_exp322, _train_one_step
from .exp323_contract import (
    ARM_LEARNING_RATES,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHECKPOINTS,
    STARTING_STEP,
    InterventionSnapshot,
    canonical_json_bytes,
    reduce_intervention,
    validate_snapshot,
)
from .exp323_evidence import expected_reconstruction_payload
from .exp323_runtime import apply_post2048_learning_rate, validate_exp322_parent_evidence
from .exp323r_identity import Exp323RExecutionIdentity, SELECTION_LOCK_DIGEST, validate_execution_identity
from .exp323r_reconstruction import validate_materialized_reconstruction


SELECTION_LOCK_PATH = "protocols/v017/exp323r_reconstruction_selection_lock_v1.json"
ARM_SCHEMA = "EXP323R-ARM-EVIDENCE-V1"
FINAL_SCHEMA = "EXP323R-FINAL-EVIDENCE-V1"
EXPECTED_PARAMETER_COUNT = 10_000_000


@dataclass(slots=True)
class LockedState:
    compiled: object
    optimizer: torch.optim.Optimizer
    reconstruction: dict[str, Any]


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_selection_lock(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP323R-RECONSTRUCTION-SELECTION-LOCK-V1":
        raise ValueError("EXP-323R selection lock schema mismatch")
    claimed = payload.get("selection_lock_digest")
    if claimed != SELECTION_LOCK_DIGEST:
        raise ValueError("EXP-323R selection lock digest mismatch")
    materialized = dict(payload)
    materialized.pop("selection_lock_digest", None)
    if _digest(materialized) != claimed:
        raise ValueError("EXP-323R selection lock canonical digest mismatch")
    if payload.get("selection_rule") != "lowest_verified_attempt_index":
        raise ValueError("EXP-323R selection rule mismatch")
    if payload.get("verified_candidate_attempts") != [0, 2] or payload.get("selected_attempt_index") != 0:
        raise ValueError("EXP-323R selected attempt mismatch")
    expected = {
        "selected_artifact_id": 10547681681,
        "selected_artifact_name": "exp323r-reconstruction-candidate-35345351869-0",
        "selected_artifact_zip_digest": "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621",
        "selected_checkpoint_sha256": "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5",
        "selected_receipt_digest": "f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f",
        "selected_receipt_file_sha256": "617ba665149a366d00782df5c4457a6a3d01081fe847aa144465377ee164bba3",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"EXP-323R selection lock field mismatch: {key}")
    alternate = payload.get("alternate_exact_candidate")
    if not isinstance(alternate, Mapping) or alternate.get("checkpoint_sha256") != expected["selected_checkpoint_sha256"]:
        raise ValueError("EXP-323R alternate exact candidate does not match selected checkpoint")
    state = payload.get("authoritative_state")
    if not isinstance(state, Mapping):
        raise ValueError("EXP-323R authoritative state missing")
    reconstruction = expected_reconstruction_payload()
    for key in (
        "completed_step",
        "model_state_digest",
        "optimizer_state_digest",
        "rng_state_digest",
        "nonfinite_events",
    ):
        if state.get(key) != reconstruction[key]:
            raise ValueError(f"EXP-323R authoritative state mismatch: {key}")
    if state.get("post_2048_optimizer_steps") != 0:
        raise ValueError("EXP-323R selection lock contains post-2048 training")
    auth = payload.get("authorization")
    if auth != AUTHORIZATION_FLAGS:
        raise ValueError("EXP-323R selection lock authorization drift")


def load_locked_reconstruction(
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path = SELECTION_LOCK_PATH,
) -> LockedState:
    lock = json.loads(Path(selection_lock_path).read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError("EXP-323R selection lock must be an object")
    validate_selection_lock(lock)
    validate_materialized_reconstruction(checkpoint_path=checkpoint_path, receipt_path=receipt_path)

    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("receipt_digest") != lock["selected_receipt_digest"]:
        raise ValueError("EXP-323R selected receipt digest mismatch")
    if receipt.get("checkpoint_sha256") != lock["selected_checkpoint_sha256"]:
        raise ValueError("EXP-323R selected checkpoint SHA mismatch")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    compiled = build_scientific_arm("A_FIXED", device="cpu")
    model = getattr(compiled, "model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("EXP-323R resident parameter count drift")
    model.load_state_dict(raw["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    optimizer.load_state_dict(raw["optimizer_state_dict"])
    torch.set_rng_state(raw["torch_rng_state"])

    reconstruction = expected_reconstruction_payload()
    observed = {
        "model_state_digest": model_state_digest(model),
        "optimizer_state_digest": optimizer_state_digest(optimizer),
        "rng_state_digest": rng_state_digest(),
    }
    for key in observed:
        if observed[key] != reconstruction[key]:
            raise ValueError(f"EXP-323R loaded state digest mismatch: {key}")
    return LockedState(compiled=compiled, optimizer=optimizer, reconstruction=dict(reconstruction))


def _arm_payload(
    *,
    arm: str,
    identity: Exp323RExecutionIdentity,
    reconstruction: Mapping[str, Any],
    snapshots: Sequence[InterventionSnapshot],
    family_summaries: Mapping[str, Any],
    completed_step: int,
    model_digest: str,
    optimizer_digest: str,
    rng_digest: str,
    invalid_reason: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": ARM_SCHEMA,
        "arm": arm,
        "learning_rate": ARM_LEARNING_RATES[arm],
        "starting_step": STARTING_STEP,
        "execution_identity": asdict(identity),
        "selection_lock_digest": SELECTION_LOCK_DIGEST,
        "reconstruction": dict(reconstruction),
        "snapshots": [asdict(x) for x in snapshots],
        "family_summaries": dict(family_summaries),
        "completed_step": completed_step,
        "final_state": {
            "model_state_digest": model_digest,
            "optimizer_state_digest": optimizer_digest,
            "rng_state_digest": rng_digest,
        },
        "invalid_reason": invalid_reason,
        **AUTHORIZATION_FLAGS,
    }
    payload["arm_evidence_digest"] = _digest(payload)
    return payload


def validate_arm_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != ARM_SCHEMA or payload.get("arm") not in ARMS:
        raise ValueError("EXP-323R arm evidence schema/arm mismatch")
    if payload.get("selection_lock_digest") != SELECTION_LOCK_DIGEST:
        raise ValueError("EXP-323R arm selection lock mismatch")
    identity_raw = payload.get("execution_identity")
    if not isinstance(identity_raw, Mapping):
        raise ValueError("EXP-323R arm execution identity missing")
    identity = Exp323RExecutionIdentity(**identity_raw)
    validate_execution_identity(identity)
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-323R arm reconstruction mismatch")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-323R forbidden authorization drift: {key}")
    rows = tuple(InterventionSnapshot(**x) for x in payload.get("snapshots", ()))
    for row in rows:
        validate_snapshot(row)
        if row.arm != payload["arm"]:
            raise ValueError("EXP-323R snapshot arm mismatch")
    if payload.get("invalid_reason") is None:
        if tuple(x.step for x in rows) != CHECKPOINTS or payload.get("completed_step") != CHECKPOINTS[-1]:
            raise ValueError("EXP-323R valid arm geometry mismatch")
    materialized = dict(payload)
    claimed = materialized.pop("arm_evidence_digest", None)
    if not isinstance(claimed, str) or _digest(materialized) != claimed:
        raise ValueError("EXP-323R arm evidence digest mismatch")


def run_locked_intervention_arm(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
    parent_exp322_path: str | Path,
    arm: str,
    execution_identity: Exp323RExecutionIdentity,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown EXP-323R arm")
    validate_execution_identity(execution_identity)
    parent = json.loads(Path(parent_exp322_path).read_text(encoding="utf-8"))
    if not isinstance(parent, Mapping):
        raise ValueError("EXP-323R parent evidence must be an object")
    validate_exp322_parent_evidence(parent)

    state = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    compiled, optimizer = state.compiled, state.optimizer
    apply_post2048_learning_rate(optimizer, arm=arm)

    worlds = tuple(materialize_stage_a())
    snapshots: list[InterventionSnapshot] = []
    families: dict[str, Any] = {}
    nonfinite = 0
    last_grad = last_update = 0.0
    completed = STARTING_STEP
    invalid_reason: str | None = None
    for global_step in range(STARTING_STEP, CHECKPOINTS[-1]):
        world = worlds[global_step % len(worlds)]
        _, last_grad, last_update, observed_nonfinite = _train_one_step(
            compiled, world, optimizer=optimizer, global_step=global_step
        )
        nonfinite += observed_nonfinite
        completed = global_step + 1
        if nonfinite:
            invalid_reason = "NONFINITE_EVENT"
            break
        if completed in CHECKPOINTS:
            legacy, family = _evaluate_exp322(
                compiled,
                arm=arm,
                step=completed,
                gradient_norm_preclip=last_grad,
                parameter_update_norm_ratio_value=last_update,
                nonfinite_events=nonfinite,
            )
            snapshots.append(InterventionSnapshot(**asdict(legacy)))
            families[str(completed)] = family

    model = getattr(compiled, "model")
    return _arm_payload(
        arm=arm,
        identity=execution_identity,
        reconstruction=state.reconstruction,
        snapshots=snapshots,
        family_summaries=families,
        completed_step=completed,
        model_digest=model_state_digest(model),
        optimizer_digest=optimizer_state_digest(optimizer),
        rng_digest=rng_state_digest(),
        invalid_reason=invalid_reason,
    )


def build_final_evidence(hold: Mapping[str, Any], decay: Mapping[str, Any]) -> dict[str, Any]:
    validate_arm_evidence(hold)
    validate_arm_evidence(decay)
    by_arm = {str(hold["arm"]): hold, str(decay["arm"]): decay}
    if set(by_arm) != set(ARMS):
        raise ValueError("EXP-323R final evidence requires both registered arms")
    if by_arm["HOLD_5E5"]["execution_identity"] != by_arm["DECAY_2P5E5"]["execution_identity"]:
        raise ValueError("EXP-323R arm identities mismatch")
    if by_arm["HOLD_5E5"]["reconstruction"] != by_arm["DECAY_2P5E5"]["reconstruction"]:
        raise ValueError("EXP-323R reconstruction receipts mismatch")

    invalid = any(by_arm[a]["invalid_reason"] is not None for a in ARMS)
    records = {
        a: tuple(InterventionSnapshot(**x) for x in by_arm[a]["snapshots"])
        for a in ARMS
    }
    decision = "INVALID_INTERVENTION" if invalid else reduce_intervention(records)
    payload: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "execution_identity": by_arm["HOLD_5E5"]["execution_identity"],
        "selection_lock_digest": SELECTION_LOCK_DIGEST,
        "reconstruction": by_arm["HOLD_5E5"]["reconstruction"],
        "arm_evidence_digests": {a: by_arm[a]["arm_evidence_digest"] for a in ARMS},
        "decision": decision,
        "arms": {
            a: {
                "learning_rate": by_arm[a]["learning_rate"],
                "snapshots": by_arm[a]["snapshots"],
                "family_summaries": by_arm[a]["family_summaries"],
                "completed_step": by_arm[a]["completed_step"],
                "final_state": by_arm[a]["final_state"],
                "invalid_reason": by_arm[a]["invalid_reason"],
            }
            for a in ARMS
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    return payload
