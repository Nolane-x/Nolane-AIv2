from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


CELL_SCHEMA = "NLM-EXP-279-COST-ACCOUNTED-BRANCH-PREVIEW-COURT-V5"
PREVIEW_FAMILIES = (
    "PREFIX1_STATE_LINEAR",
    "PREFIX2_STATE_LINEAR",
    "PREFIX3_STATE_LINEAR",
)
CROSS_CELL_SCHEMA = "NLM-EXP-279-COST-ACCOUNTED-BRANCH-PREVIEW-CROSS-CELL-V5"
_DECISION_STEPS = (60, 120)
_DECISIONS = {
    "PREFIX1_STATE_LINEAR": "ROBUST_PREFIX1_BRANCH_PREVIEW",
    "PREFIX2_STATE_LINEAR": "ROBUST_PREFIX2_BRANCH_PREVIEW",
    "PREFIX3_STATE_LINEAR": "ROBUST_PREFIX3_BRANCH_PREVIEW",
}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _validate_cell(cell: dict[str, Any]) -> None:
    if cell.get("schema") != CELL_SCHEMA:
        raise ValueError("V5 cross-cell input schema mismatch")
    if cell.get("evidence_state") != "EV-E2 / UNVERIFIED":
        raise ValueError("V5 cross-cell evidence state mismatch")
    if cell.get("scientific_evidence_eligible") is not False:
        raise ValueError("V5 cross-cell inputs cannot be scientific-evidence eligible")

    boundary = cell.get("data_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("V5 cross-cell data boundary missing")
    if boundary.get("evaluation_rng_stream_used") is not False:
        raise ValueError("V5 cross-cell forbids evaluation RNG usage")
    if boundary.get("evaluation_targets_used") is not False:
        raise ValueError("V5 cross-cell forbids evaluation targets")

    cost_rule = cell.get("preview_cost_rule")
    if not isinstance(cost_rule, dict):
        raise ValueError("V5 cross-cell preview cost rule missing")
    if cost_rule.get("preview_flops_charged_to_primary_utility") is not True:
        raise ValueError("V5 cross-cell requires preview FLOPs charged to primary utility")
    if cost_rule.get("selector_flops_charged_to_primary_utility") is not True:
        raise ValueError("V5 cross-cell requires selector FLOPs charged to primary utility")

    for key in (
        "fresh_evaluation_lineage_may_be_reserved",
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        if cell.get(key) is not False:
            raise ValueError(f"V5 cross-cell boundary violation: {key}")

    probes = cell.get("probes")
    if not isinstance(probes, dict) or tuple(probes.keys()) != PREVIEW_FAMILIES:
        if not isinstance(probes, dict) or set(probes) != set(PREVIEW_FAMILIES):
            raise ValueError("V5 cross-cell preview family mismatch")
    for family in PREVIEW_FAMILIES:
        probe = probes.get(family)
        aggregate = probe.get("aggregate") if isinstance(probe, dict) else None
        if not isinstance(aggregate, dict):
            raise ValueError(f"V5 cross-cell aggregate missing for {family}")
        if aggregate.get("support_closed") not in (True, False):
            raise ValueError(f"V5 cross-cell support flag missing for {family}")
        if aggregate.get("direct_utility_improved") not in (True, False):
            raise ValueError(f"V5 cross-cell utility flag missing for {family}")


def classify_exp279_cost_accounted_branch_preview_v5_cross_cell(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(cells) != 2:
        raise ValueError("V5 cross-cell court requires exactly train60 and train120")
    for cell in cells:
        _validate_cell(cell)

    by_steps: dict[int, dict[str, Any]] = {}
    for cell in cells:
        steps = cell.get("train_steps")
        if not isinstance(steps, int) or steps in by_steps:
            raise ValueError("V5 cross-cell train_steps must be unique integers")
        by_steps[steps] = cell
    if tuple(sorted(by_steps)) != _DECISION_STEPS:
        raise ValueError("V5 cross-cell decision cells must be exactly train60 and train120")

    support_closed = all(
        bool(by_steps[steps]["probes"][family]["aggregate"]["support_closed"])
        for steps in _DECISION_STEPS
        for family in PREVIEW_FAMILIES
    )

    robust_families = [
        family
        for family in PREVIEW_FAMILIES
        if all(
            bool(by_steps[steps]["probes"][family]["aggregate"]["support_closed"])
            and bool(by_steps[steps]["probes"][family]["aggregate"]["direct_utility_improved"])
            for steps in _DECISION_STEPS
        )
    ]

    if not support_closed:
        decision = "INCONCLUSIVE_SUPPORT"
        successor_design_authorized = False
    elif robust_families:
        decision = _DECISIONS[robust_families[0]]
        successor_design_authorized = True
    else:
        decision = "NO_ROBUST_COST_ACCOUNTED_BRANCH_PREVIEW"
        successor_design_authorized = False

    result: dict[str, Any] = {
        "schema": CROSS_CELL_SCHEMA,
        "cell_schema": CELL_SCHEMA,
        "decision_cells": list(_DECISION_STEPS),
        "preview_families": list(PREVIEW_FAMILIES),
        "support_closed": support_closed,
        "robust_preview_families": robust_families,
        "decision": decision,
        "successor_design_authorized": successor_design_authorized,
        "evidence_state": "EV-E2 / UNVERIFIED",
        "scientific_evidence_eligible": False,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "anti_cherry_pick_rule": (
            "a preview family is robust only when its fold-local, cost-accounted direct utility "
            "improves over STOP in both train60 and train120; cheapest robust prefix wins"
        ),
    }
    result["artifact_digest"] = _artifact_digest(result)
    return result
