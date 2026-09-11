from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256

from .exp279_augmentation_direct_utility_v3 import PROBE_KINDS, SCHEMA as CELL_SCHEMA


SCHEMA = "NLM-EXP-279-AUGMENTATION-DIRECT-UTILITY-CROSS-CELL-V3"
DECISION_TRAIN_REPLICATES = (60, 120)
ALLOWED_DECISIONS = {
    "ROBUST_LINEAR_DIRECT_UTILITY",
    "ROBUST_RICH_DIRECT_UTILITY",
    "NO_ROBUST_DIRECT_UTILITY",
    "INCONCLUSIVE_SUPPORT",
}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _validate_cell_boundary(train_replicates: int, cell: dict[str, Any]) -> None:
    if cell.get("schema") != CELL_SCHEMA:
        raise ValueError(f"train {train_replicates} V3 cell schema mismatch")
    if cell.get("evidence_level") != "EV-E2" or cell.get("decision") != "UNVERIFIED":
        raise ValueError(f"train {train_replicates} V3 cell evidence boundary mismatch")
    boundary = cell.get("data_boundary") or {}
    if boundary.get("evaluation_rng_stream_used") is not False or boundary.get("evaluation_targets_used") is not False:
        raise ValueError(f"train {train_replicates} evaluation boundary violated")
    for key in (
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        if cell.get(key) is not False:
            raise ValueError(f"train {train_replicates} {key} boundary violated")
    probes = cell.get("probes") or {}
    if set(probes) != set(PROBE_KINDS):
        raise ValueError(f"train {train_replicates} probe family incomplete")


def _cell_support_closed(cell: dict[str, Any]) -> bool:
    return all(bool((cell["probes"][kind].get("aggregate") or {}).get("support_closed")) for kind in PROBE_KINDS)


def _family_passes(cell: dict[str, Any], kind: str) -> bool:
    return bool(cell["probes"][kind].get("economically_routable"))


def classify_exp279_direct_utility_v3_cross_cell(cells: dict[int, dict[str, Any]]) -> dict[str, Any]:
    missing = [replicate for replicate in DECISION_TRAIN_REPLICATES if replicate not in cells]
    if missing:
        raise ValueError("V3 cross-cell decision requires train 60 and train 120")

    decision_cells = {replicate: cells[replicate] for replicate in DECISION_TRAIN_REPLICATES}
    for replicate, cell in decision_cells.items():
        _validate_cell_boundary(replicate, cell)

    support_closed = {str(replicate): _cell_support_closed(cell) for replicate, cell in decision_cells.items()}
    robust_probe_families = [
        kind for kind in PROBE_KINDS if all(_family_passes(cell, kind) for cell in decision_cells.values())
    ]

    if not all(support_closed.values()):
        decision = "INCONCLUSIVE_SUPPORT"
        robust_probe_families = []
    elif "LINEAR_MEAN" in robust_probe_families:
        decision = "ROBUST_LINEAR_DIRECT_UTILITY"
        robust_probe_families = ["LINEAR_MEAN"]
    else:
        robust_rich = [kind for kind in ("MLP_MEAN", "DEEPSETS") if kind in robust_probe_families]
        if robust_rich:
            decision = "ROBUST_RICH_DIRECT_UTILITY"
            robust_probe_families = robust_rich
        else:
            decision = "NO_ROBUST_DIRECT_UTILITY"
            robust_probe_families = []

    successor_design_authorized = decision in {"ROBUST_LINEAR_DIRECT_UTILITY", "ROBUST_RICH_DIRECT_UTILITY"}
    receipts = {
        str(replicate): {
            "artifact_digest": cell.get("artifact_digest"),
            "court_classification": cell.get("court_classification"),
            "probe_passes": {kind: _family_passes(cell, kind) for kind in PROBE_KINDS},
            "support_closed": support_closed[str(replicate)],
        }
        for replicate, cell in decision_cells.items()
    }

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": decision,
        "decision_train_replicates": list(DECISION_TRAIN_REPLICATES),
        "same_probe_family_required_across_decision_cells": True,
        "cell_receipts": receipts,
        "robust_probe_families": robust_probe_families,
        "successor_design_authorized": successor_design_authorized,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "scientific_evidence_eligible": False,
        "analysis_boundary": (
            "augmentation-only cross-cell DEVELOPMENT decision; robust direct utility authorizes "
            "successor design only, not fresh evaluation lineage consumption"
        ),
        "artifact_digest": "",
    }
    if payload["decision"] not in ALLOWED_DECISIONS:
        raise RuntimeError("invalid EXP-279 V3 cross-cell decision")
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload
