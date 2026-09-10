from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "NLM-EXP-289-AUTHORITATIVE-DEVELOPMENT-V1"
AUTHORITY_SCOPE = "DEVELOPMENT_PILOT_ONLY"
PURPOSE = "AUTHORITATIVE_DEVELOPMENT_GATE_A"


def load_exp289_development_geometry(
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
            f"EXP-289 geometry digest mismatch: expected {expected_digest!r}, got {actual_digest!r}"
        )

    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != "EXP-289":
        raise ValueError("EXP-289 development geometry identity drift")
    if payload.get("purpose") != PURPOSE:
        raise ValueError("EXP-289 development geometry purpose drift")
    if payload.get("authority_scope") != AUTHORITY_SCOPE:
        raise ValueError("EXP-289 development geometry authority scope drift")
    if payload.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-289 development geometry protocol lineage mismatch")
    if payload.get("predeclaration") != "BEFORE_AUTHORITATIVE_DEVELOPMENT_RUN":
        raise ValueError("EXP-289 development geometry predeclaration boundary drift")
    if payload.get("confirmatory_authority") is not False:
        raise ValueError("EXP-289 development geometry cannot claim confirmatory authority")
    if payload.get("challenge_materialized") is not False:
        raise ValueError("EXP-289 development geometry cannot materialize challenge entropy")
    if payload.get("pilot_reuse_as_confirmatory") is not False:
        raise ValueError("EXP-289 development geometry cannot authorize pilot reuse")
    if "beacon" in payload or "challenge_seed" in payload:
        raise ValueError("EXP-289 development geometry cannot contain confirmatory entropy")

    if payload.get("primary_endpoint") != "repeat_dead_end_rate":
        raise ValueError("EXP-289 development geometry primary endpoint drift")
    if payload.get("mesi_relative_reduction") != 0.25:
        raise ValueError("EXP-289 development geometry MESI drift")
    if payload.get("zero_opportunity_policy") != "retain_raw_episode_exclude_from_rder_denominator":
        raise ValueError("EXP-289 development geometry zero-opportunity policy drift")
    if payload.get("epsilon_denominator_rescue") is not False:
        raise ValueError("EXP-289 development geometry cannot enable epsilon denominator rescue")
    protected = payload.get("protected_endpoints")
    if protected != {
        "valid_state_overprune_rate_ceiling": 0.005,
        "verified_solution_rate_floor_delta": -0.01,
    }:
        raise ValueError("EXP-289 development geometry protected endpoint drift")

    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("EXP-289 development geometry payload is missing")
    if geometry.get("tiny") is not False:
        raise ValueError("EXP-289 authoritative DEVELOPMENT geometry cannot be tiny smoke")
    try:
        train_replicates = int(geometry["train_replicates"])
        eval_replicates = int(geometry["eval_replicates"])
        eval_start_replicate = int(geometry["eval_start_replicate"])
        restarts = int(geometry["restarts"])
        decoys = int(geometry["decoys"])
        variables = int(geometry["variables"])
        max_search_steps = int(geometry["max_search_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EXP-289 development geometry counts are invalid") from exc
    if train_replicates <= 0 or eval_replicates < 32:
        raise ValueError("EXP-289 authoritative DEVELOPMENT geometry requires positive training and at least 32 eval replicates")
    if eval_start_replicate < train_replicates:
        raise ValueError("EXP-289 DEVELOPMENT train/eval lineage must be disjoint")
    if restarts < 2:
        raise ValueError("EXP-289 authoritative DEVELOPMENT geometry requires repeated restart opportunities")
    if decoys <= 0 or decoys > variables - 1:
        raise ValueError("EXP-289 authoritative DEVELOPMENT geometry decoy count is invalid")
    if max_search_steps <= 0:
        raise ValueError("EXP-289 authoritative DEVELOPMENT geometry search budget is invalid")

    return dict(geometry), actual_digest
