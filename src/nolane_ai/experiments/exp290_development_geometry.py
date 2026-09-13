from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "NLM-EXP-290-STRUCTURAL-CLAUSE-TRANSFER-DEVELOPMENT-V1"
AUTHORITY_SCOPE = "DEVELOPMENT_EV_E2_ONLY"
PURPOSE = "STRUCTURAL_CLAUSE_TRANSFER_DEVELOPMENT_COURT_V1"
ROOT_PREFIX = "20260913-exp290-structural-clause-transfer-v1-dev"

_EXPECTED_GEOMETRY: dict[str, Any] = {
    "batch_size": 8,
    "canonical_indices": [0, 1, 2, 3],
    "d_model": 64,
    "decoys": 3,
    "eval_replicates": 32,
    "eval_start_replicate": 40000,
    "hidden_size": 48,
    "lr": 0.002,
    "max_search_steps": 24,
    "noise_std": 0.05,
    "restarts": 4,
    "root_prefix": ROOT_PREFIX,
    "target_parameters": 500000,
    "timesteps": 4,
    "train_replicates": 64,
    "variables": 8,
    "weight_decay": 0.0,
}


def load_exp290_development_geometry(
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
            f"EXP-290 geometry digest mismatch: expected {expected_digest!r}, got {actual_digest!r}"
        )

    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != "EXP-290":
        raise ValueError("EXP-290 development geometry identity drift")
    if payload.get("purpose") != PURPOSE:
        raise ValueError("EXP-290 development geometry purpose drift")
    if payload.get("authority_scope") != AUTHORITY_SCOPE:
        raise ValueError("EXP-290 development geometry authority scope drift")
    if payload.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-290 development geometry protocol lineage mismatch")
    if payload.get("predeclaration") != "BEFORE_AUTHORITATIVE_DEVELOPMENT_RUN":
        raise ValueError("EXP-290 predeclaration boundary drift")

    false_flags = (
        "scientific_evidence_eligible",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "stage_a_protocol_modified",
        "oracle_transfer_mode_deployable",
        "arbitrary_value_symbol_remapping_claimed",
        "cross_domain_transfer_claimed",
        "lemma_generation_claimed",
    )
    for key in false_flags:
        if payload.get(key) is not False:
            raise ValueError(f"EXP-290 boundary flag must remain false: {key}")
    if "beacon" in payload or "challenge_seed" in payload:
        raise ValueError("EXP-290 DEVELOPMENT geometry cannot contain confirmatory entropy")

    expected_surface = {
        "event_order_randomized": True,
        "value_labels_remapped": False,
        "variable_identity_randomized": True,
        "variable_order_randomized": True,
    }
    if payload.get("surface_randomization") != expected_surface:
        raise ValueError("EXP-290 surface-randomization contract drift")
    if payload.get("primary_endpoint") != "source_equivalent_target_dead_end_rate":
        raise ValueError("EXP-290 primary endpoint drift")
    if payload.get("capture_threshold") != 0.5:
        raise ValueError("EXP-290 capture threshold drift")
    if payload.get("protected_endpoints") != {
        "valid_state_overprune_rate_ceiling": 0.005,
        "verified_solution_rate_floor_delta": -0.01,
    }:
        raise ValueError("EXP-290 protected endpoint drift")
    if payload.get("successor_scope_if_recurrent") != "DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY":
        raise ValueError("EXP-290 successor scope drift")
    if payload.get("zero_opportunity_policy") != "retain_raw_episode_exclude_from_transfer_rate_denominator":
        raise ValueError("EXP-290 zero-opportunity policy drift")
    if payload.get("epsilon_denominator_rescue") is not False:
        raise ValueError("EXP-290 cannot enable epsilon denominator rescue")

    geometry = payload.get("geometry")
    if geometry != _EXPECTED_GEOMETRY:
        raise ValueError("EXP-290 frozen DEVELOPMENT geometry drift")
    if set(geometry["canonical_indices"]) != {0, 1, 2, 3}:
        raise ValueError("EXP-290 canonical root set drift")
    if int(geometry["eval_start_replicate"]) < int(geometry["train_replicates"]):
        raise ValueError("EXP-290 train/evaluation lineage must remain disjoint")
    return dict(geometry), actual_digest
