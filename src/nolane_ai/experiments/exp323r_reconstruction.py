from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp323_contract import (
    AUTHORIZATION_FLAGS,
    PARENT_DECAY_MODEL_STATE_DIGEST,
    PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
    PARENT_DECAY_RNG_STATE_DIGEST,
    PARENT_EXP322_EVIDENCE_DIGEST,
    PARENT_EXP322_EXECUTION_DIGEST,
    PARENT_EXP322_RUN_ID,
)
from .exp323_evidence import expected_reconstruction_payload
from .exp323_runtime import (
    ReconstructedState,
    reconstruct_exp322_decay_state,
    validate_exp322_parent_evidence,
)


RECONSTRUCTION_SCHEMA = "EXP323R-MATERIALIZED-RECONSTRUCTION-V1"
RECEIPT_SCHEMA = "EXP323R-MATERIALIZED-RECONSTRUCTION-RECEIPT-V1"


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_without_digest(
    *,
    checkpoint_sha256: str,
    attempt_index: int,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "attempt_index": attempt_index,
        "parent_exp322_run_id": PARENT_EXP322_RUN_ID,
        "parent_exp322_evidence_digest": PARENT_EXP322_EVIDENCE_DIGEST,
        "parent_exp322_execution_digest": PARENT_EXP322_EXECUTION_DIGEST,
        "reconstruction": expected_reconstruction_payload(),
        "checkpoint_sha256": checkpoint_sha256,
        "post_2048_optimizer_steps": 0,
        **AUTHORIZATION_FLAGS,
    }


def receipt_digest(payload: Mapping[str, Any]) -> str:
    materialized = dict(payload)
    materialized.pop("receipt_digest", None)
    return hashlib.sha256(canonical_json_bytes(materialized)).hexdigest()


def materialize_verified_reconstruction(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    parent_exp322_path: str | Path,
    output_checkpoint_path: str | Path,
    output_receipt_path: str | Path,
    output_diagnostic_path: str | Path,
    attempt_index: int,
) -> bool:
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")

    parent_payload = json.loads(Path(parent_exp322_path).read_text(encoding="utf-8"))
    if not isinstance(parent_payload, Mapping):
        raise ValueError("EXP-323R parent evidence must be a JSON object")
    validate_exp322_parent_evidence(parent_payload)

    diagnostic: dict[str, Any]
    try:
        state = reconstruct_exp322_decay_state(
            checkpoint_path=checkpoint_path,
            receipt_path=receipt_path,
        )
    except ValueError as exc:
        diagnostic = {
            "schema": "EXP323R-RECONSTRUCTION-DIAGNOSTIC-V1",
            "attempt_index": attempt_index,
            "verified": False,
            "reason": str(exc),
            "post_2048_optimizer_steps": 0,
        }
        target = Path(output_diagnostic_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json_bytes(diagnostic) + b"\n")
        return False

    reconstruction = state.reconstruction
    if reconstruction != expected_reconstruction_payload():
        raise AssertionError("verified reconstruction payload drifted after validation")

    model = getattr(state.compiled, "model")
    observed = {
        "model_state_digest": model_state_digest(model),
        "optimizer_state_digest": optimizer_state_digest(state.optimizer),
        "rng_state_digest": rng_state_digest(),
    }
    expected = {
        "model_state_digest": PARENT_DECAY_MODEL_STATE_DIGEST,
        "optimizer_state_digest": PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
        "rng_state_digest": PARENT_DECAY_RNG_STATE_DIGEST,
    }
    if observed != expected:
        raise AssertionError("verified reconstruction state drifted before serialization")

    output_checkpoint = Path(output_checkpoint_path)
    output_receipt = Path(output_receipt_path)
    output_diagnostic = Path(output_diagnostic_path)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_diagnostic.parent.mkdir(parents=True, exist_ok=True)
    if output_checkpoint.exists() or output_receipt.exists():
        raise FileExistsError("EXP-323R reconstruction outputs already exist")

    payload = {
        "schema": RECONSTRUCTION_SCHEMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "reconstruction": reconstruction,
        "post_2048_optimizer_steps": 0,
    }
    with output_checkpoint.open("xb") as handle:
        torch.save(payload, handle)

    checkpoint_sha = sha256_file(output_checkpoint)
    receipt = _receipt_without_digest(
        checkpoint_sha256=checkpoint_sha,
        attempt_index=attempt_index,
    )
    receipt["receipt_digest"] = receipt_digest(receipt)
    output_receipt.write_bytes(canonical_json_bytes(receipt) + b"\n")

    diagnostic = {
        "schema": "EXP323R-RECONSTRUCTION-DIAGNOSTIC-V1",
        "attempt_index": attempt_index,
        "verified": True,
        "checkpoint_sha256": checkpoint_sha,
        "receipt_digest": receipt["receipt_digest"],
        "reconstruction": reconstruction,
        "post_2048_optimizer_steps": 0,
    }
    output_diagnostic.write_bytes(canonical_json_bytes(diagnostic) + b"\n")
    return True


def validate_materialized_reconstruction(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
) -> None:
    receipt_payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if not isinstance(receipt_payload, dict):
        raise ValueError("EXP-323R reconstruction receipt must be an object")
    if receipt_payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("EXP-323R reconstruction receipt schema mismatch")
    if receipt_payload.get("receipt_digest") != receipt_digest(receipt_payload):
        raise ValueError("EXP-323R reconstruction receipt digest mismatch")
    if receipt_payload.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("EXP-323R reconstruction checkpoint SHA mismatch")
    if receipt_payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-323R reconstruction receipt authority mismatch")
    if receipt_payload.get("post_2048_optimizer_steps") != 0:
        raise ValueError("EXP-323R reconstruction receipt contains post-2048 training")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if receipt_payload.get(key) is not expected:
            raise ValueError(f"EXP-323R forbidden authorization drift: {key}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("schema") != RECONSTRUCTION_SCHEMA:
        raise ValueError("EXP-323R reconstruction checkpoint schema mismatch")
    if raw.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-323R checkpoint reconstruction authority mismatch")
    if raw.get("post_2048_optimizer_steps") != 0:
        raise ValueError("EXP-323R checkpoint contains post-2048 training")
    if not isinstance(raw.get("model_state_dict"), Mapping):
        raise ValueError("EXP-323R checkpoint model state malformed")
    if not isinstance(raw.get("optimizer_state_dict"), Mapping):
        raise ValueError("EXP-323R checkpoint optimizer state malformed")
    rng = raw.get("torch_rng_state")
    if not isinstance(rng, torch.Tensor):
        raise ValueError("EXP-323R checkpoint RNG state malformed")
