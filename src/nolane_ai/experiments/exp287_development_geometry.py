from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "NLM-EXP-287-DEVELOPMENT-GEOMETRY-V1"
AUTHORITY_SCOPE = "DEVELOPMENT_ONLY"
CANONICAL_INDICES = (0, 1, 2, 3)

_EXPECTED_GEOMETRY: dict[str, Any] = {
    "root_prefix": "20260913-exp287-learned-conflict-localization-v1-dev",
    "canonical_indices": [0, 1, 2, 3],
    "train_replicates": 64,
    "eval_replicates": 32,
    "eval_start_replicate": 10000,
    "batch_size": 8,
    "d_model": 64,
    "hidden_size": 48,
    "target_parameters": 500000,
    "timesteps": 4,
    "variables": 8,
    "decoys": 3,
    "max_search_steps": 16,
    "noise_std": 0.05,
    "lr": 0.002,
    "weight_decay": 0.0,
    "top_k": 2,
    "capture_threshold": 0.50,
    "precision_threshold": 0.50,
    "solution_rate_floor_delta": -0.005,
}


def load_exp287_development_geometry(
    manifest_path: str | Path,
    digest_path: str | Path,
    *,
    protocol_digest: str,
) -> tuple[dict[str, Any], str]:
    manifest_path = Path(manifest_path)
    digest_path = Path(digest_path)
    raw = manifest_path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = digest_path.read_text(encoding="utf-8").strip()
    if actual_digest != expected_digest:
        raise ValueError(
            f"EXP-287 geometry digest mismatch: expected {expected_digest!r}, got {actual_digest!r}"
        )

    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != "EXP-287":
        raise ValueError("EXP-287 development geometry identity drift")
    if payload.get("authority_scope") != AUTHORITY_SCOPE:
        raise ValueError("EXP-287 development geometry authority scope drift")
    if payload.get("predeclaration") != "BEFORE_HELDOUT_EVALUATION":
        raise ValueError("EXP-287 development geometry predeclaration drift")
    if payload.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-287 development geometry protocol lineage mismatch")
    if payload.get("confirmatory_authority") is not False:
        raise ValueError("EXP-287 geometry cannot claim confirmatory authority")
    if payload.get("challenge_materialized") is not False:
        raise ValueError("EXP-287 geometry cannot materialize challenge entropy")
    if payload.get("scientific_evidence_eligible") is not False:
        raise ValueError("EXP-287 geometry cannot claim scientific evidence eligibility")
    if "beacon" in payload or "challenge_seed" in payload:
        raise ValueError("EXP-287 development geometry cannot contain challenge entropy")

    geometry = payload.get("geometry")
    if geometry != _EXPECTED_GEOMETRY:
        raise ValueError("EXP-287 frozen DEVELOPMENT geometry drift")
    if int(geometry["eval_start_replicate"]) < int(geometry["train_replicates"]):
        raise ValueError("EXP-287 DEVELOPMENT train/eval lineage must be disjoint")

    return dict(geometry), actual_digest
