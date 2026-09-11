from __future__ import annotations

from typing import Any

from nolane_ai.experiments.exp279_prebranch_observability_v4 import PROBE_KINDS, SCHEMA
from nolane_ai.protocol.evidence import canonical_sha256

CROSS_CELL_SCHEMA = "NLM-EXP-279-PREBRANCH-OBSERVABILITY-CROSS-CELL-V4"


def _validate_cell(train_replicates: int, cell: dict[str, Any]) -> None:
    if cell.get("schema") != SCHEMA:
        raise ValueError(f"train {train_replicates} V4 schema mismatch")
    if cell.get("evidence_level") != "EV-E2" or cell.get("decision") != "UNVERIFIED":
        raise ValueError(f"train {train_replicates} evidence boundary mismatch")

    data_boundary = cell.get("data_boundary") or {}
    if data_boundary.get("evaluation_rng_stream_used") is not False or data_boundary.get(
        "evaluation_targets_used"
    ) is not False:
        raise ValueError(f"train {train_replicates} evaluation boundary violated")

    feature_boundary = cell.get("feature_boundary") or {}
    if feature_boundary.get("branch_gru_used_for_probe_features") is not False:
        raise ValueError(f"train {train_replicates} branch-GRU feature boundary violated")

    for key in (
        "fresh_evaluation_lineage_may_be_reserved",
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        if cell.get(key) is not False:
            raise ValueError(f"train {train_replicates} scientific boundary violated: {key}")

    probes = cell.get("probes") or {}
    if set(probes) != set(PROBE_KINDS):
        raise ValueError(f"train {train_replicates} probe family mismatch")
    for kind in PROBE_KINDS:
        probe = probes[kind]
        aggregate = probe.get("aggregate") or {}
        if "support_closed" not in aggregate or "economically_routable" not in probe:
            raise ValueError(f"train {train_replicates} incomplete {kind} receipt")


def _all_support_closed(cell: dict[str, Any]) -> bool:
    return all(bool(cell["probes"][kind]["aggregate"]["support_closed"]) for kind in PROBE_KINDS)


def classify_exp279_prebranch_observability_v4_cross_cell(
    cells: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if 60 not in cells or 120 not in cells:
        raise ValueError("V4 cross-cell court requires train 60 and train 120 decision cells")

    decision_cells = {60: cells[60], 120: cells[120]}
    for train_replicates, cell in decision_cells.items():
        _validate_cell(train_replicates, cell)

    support_closed = all(_all_support_closed(cell) for cell in decision_cells.values())
    robust_probe_families = [
        kind
        for kind in PROBE_KINDS
        if all(bool(cell["probes"][kind]["economically_routable"]) for cell in decision_cells.values())
    ] if support_closed else []

    if not support_closed:
        decision = "INCONCLUSIVE_SUPPORT"
        selected_family = None
    elif "STATE_DEEPSETS" in robust_probe_families:
        decision = "ROBUST_STATE_ONLY_PREBRANCH_SIGNAL"
        selected_family = "STATE_DEEPSETS"
        robust_probe_families = [selected_family]
    elif "STATE_EVENT_MEAN" in robust_probe_families:
        decision = "ROBUST_EVENT_MEAN_PREBRANCH_SIGNAL"
        selected_family = "STATE_EVENT_MEAN"
        robust_probe_families = [selected_family]
    elif "STATE_EVENT_RESIDUAL" in robust_probe_families:
        decision = "ROBUST_EVENT_RESIDUAL_PREBRANCH_SIGNAL"
        selected_family = "STATE_EVENT_RESIDUAL"
        robust_probe_families = [selected_family]
    else:
        decision = "NO_ROBUST_PREBRANCH_OBSERVABILITY"
        selected_family = None
        robust_probe_families = []

    payload: dict[str, Any] = {
        "schema": CROSS_CELL_SCHEMA,
        "experiment_id": "EXP-279",
        "evidence_level": "EV-E2",
        "scientific_decision": "UNVERIFIED",
        "decision": decision,
        "decision_cells": [60, 120],
        "train15_is_descriptive_only": True,
        "probe_family_order": list(PROBE_KINDS),
        "support_closed": support_closed,
        "robust_probe_families": robust_probe_families,
        "selected_probe_family": selected_family,
        "successor_design_authorized": decision.startswith("ROBUST_"),
        "cell_artifact_digests": {
            str(train_replicates): decision_cells[train_replicates].get("artifact_digest")
            for train_replicates in (60, 120)
        },
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "artifact_digest": "",
    }
    digest_payload = dict(payload)
    digest_payload["artifact_digest"] = ""
    payload["artifact_digest"] = canonical_sha256(digest_payload)
    return payload
