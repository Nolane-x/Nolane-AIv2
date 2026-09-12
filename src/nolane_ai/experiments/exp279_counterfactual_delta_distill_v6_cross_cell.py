from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256

from .exp279_counterfactual_delta_distill_v6_primitives import (
    CONTROL_FAMILY,
    PRIMARY_FAMILY,
    PROBE_ROOT_SUFFIX,
    SCHEMA as CELL_SCHEMA,
)


SCHEMA = "NLM-EXP-279-COUNTERFACTUAL-DELTA-DISTILL-CROSS-CELL-V6"
DECISION_TRAIN_REPLICATES = (60, 120)
FROZEN_PROTOCOL_ID = "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
FROZEN_PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
FROZEN_ROOT_SEED = "20260912-exp279-counterfactual-delta-distill-v6-dev"
FROZEN_PROBE_ROOT_SEED = FROZEN_ROOT_SEED + PROBE_ROOT_SUFFIX
ALLOWED_DECISIONS = {
    "ROBUST_CDD_BRANCH_COMPLEMENTARITY",
    "NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY",
    "INCONCLUSIVE_SUPPORT",
}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _require_false(cell: dict[str, Any], key: str, train_replicates: int) -> None:
    if cell.get(key) is not False:
        raise ValueError(f"train {train_replicates} {key} boundary violated")


def _validate_cell_boundary(train_replicates: int, cell: dict[str, Any]) -> None:
    if cell.get("schema") != CELL_SCHEMA:
        raise ValueError(f"train {train_replicates} V6 cell schema mismatch")
    if (
        cell.get("evidence_level") != "EV-E2"
        or cell.get("decision") != "UNVERIFIED"
        or cell.get("scientific_evidence_eligible") is not False
    ):
        raise ValueError(f"train {train_replicates} V6 cell evidence boundary mismatch")
    if cell.get("experiment_id") != "EXP-279":
        raise ValueError(f"train {train_replicates} V6 experiment boundary violated")
    if cell.get("canonical_model_frozen_for_cdd") is not True:
        raise ValueError(f"train {train_replicates} canonical freeze boundary violated")

    canonical = cell.get("canonical_training") or {}
    if canonical.get("replicates") != train_replicates:
        raise ValueError(f"train {train_replicates} replicate mismatch")
    if canonical.get("rng_stream") != "augmentation" or canonical.get(
        "canonical_parameters_frozen_before_cdd"
    ) is not True:
        raise ValueError(f"train {train_replicates} canonical training boundary violated")

    boundary = cell.get("data_boundary") or {}
    required_boundary = {
        "training_rng_stream": "augmentation",
        "probe_rng_stream": "augmentation",
        "probe_root_independent_from_training_root": True,
        "evaluation_rng_stream_used": False,
        "evaluation_targets_used": False,
        "heldout_branch_hidden_used_for_features": False,
        "confirmatory_examples_used": False,
        "external_examples_used": False,
    }
    if any(boundary.get(key) != value for key, value in required_boundary.items()):
        raise ValueError(f"train {train_replicates} data boundary violated")

    teacher = cell.get("teacher_rule") or {}
    if teacher.get("teacher_uses_fit_partition_only") is not True or teacher.get(
        "heldout_teacher_delta_materialized_for_features"
    ) is not False:
        raise ValueError(f"train {train_replicates} teacher boundary violated")

    probe_config = cell.get("probe_config") or {}
    if (
        probe_config.get("primary_family") != PRIMARY_FAMILY
        or probe_config.get("descriptive_control_family") != CONTROL_FAMILY
        or probe_config.get("control_can_authorize_successor") is not False
        or probe_config.get("raw_scores_exported") is not False
        or probe_config.get("raw_scores_compared_across_folds") is not False
    ):
        raise ValueError(f"train {train_replicates} probe boundary violated")

    cost_rule = cell.get("cdd_cost_rule") or {}
    if (
        cost_rule.get("cdd_inference_flops_charged_to_primary_utility") is not True
        or cost_rule.get("cdd_cost_charged_on_stop_and_branch_paths") is not True
        or cost_rule.get("false_positive_harms_counted_directly") is not True
    ):
        raise ValueError(f"train {train_replicates} compute boundary violated")

    for key in (
        "fresh_evaluation_lineage_may_be_reserved",
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        _require_false(cell, key, train_replicates)

    probes = cell.get("probes") or {}
    if set(probes) != {PRIMARY_FAMILY, CONTROL_FAMILY}:
        raise ValueError(f"train {train_replicates} probe family boundary violated")
    primary = probes[PRIMARY_FAMILY]
    control = probes[CONTROL_FAMILY]
    if primary.get("raw_scores_compared_across_folds") is not False:
        raise ValueError(f"train {train_replicates} primary score boundary violated")
    if control.get("authorizes_successor") is not False or control.get(
        "raw_scores_compared_across_folds"
    ) is not False:
        raise ValueError(f"train {train_replicates} control boundary violated")


def _validate_frozen_configuration(train_replicates: int, cell: dict[str, Any]) -> None:
    if cell.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: protocol_id")
    if cell.get("protocol_digest") != FROZEN_PROTOCOL_DIGEST:
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: protocol_digest")
    if cell.get("root_seed") != FROZEN_ROOT_SEED:
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: root_seed")

    boundary = cell.get("data_boundary") or {}
    if boundary.get("probe_root_seed") != FROZEN_PROBE_ROOT_SEED:
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: probe_root_seed")

    route = cell.get("route_config") or {}
    if route.get("canonical_threshold") != 0.5 or route.get("threshold_tuned") is not False:
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: route_config")

    world = cell.get("world_geometry") or {}
    expected_world = {
        "batch_size": 8,
        "timesteps": 4,
        "variables": 6,
        "constraints": 3,
        "d_model": 64,
        "noise_std": 0.05,
    }
    if any(world.get(key) != value for key, value in expected_world.items()):
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: world_geometry")

    arm = cell.get("arm_geometry") or {}
    expected_arm = {"hidden_size": 48, "target_parameters": 500000}
    if any(arm.get(key) != value for key, value in expected_arm.items()):
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: arm_geometry")

    probe = cell.get("probe_config") or {}
    expected_probe = {
        "replicates": 198,
        "folds": 3,
        "distill_steps": 200,
        "distill_lr": 0.002,
        "selector_steps": 200,
        "selector_lr": 0.01,
    }
    if any(probe.get(key) != value for key, value in expected_probe.items()):
        raise ValueError(f"train {train_replicates} frozen configuration mismatch: probe_config")


def _validate_shared_provenance(cells: dict[int, dict[str, Any]]) -> None:
    first = cells[DECISION_TRAIN_REPLICATES[0]]
    for replicate in DECISION_TRAIN_REPLICATES[1:]:
        cell = cells[replicate]
        for key in ("protocol_id", "protocol_digest", "code_digest", "root_seed"):
            if cell.get(key) != first.get(key):
                raise ValueError(f"train {replicate} V6 provenance mismatch for {key}")
        first_boundary = first.get("data_boundary") or {}
        boundary = cell.get("data_boundary") or {}
        if boundary.get("probe_root_seed") != first_boundary.get("probe_root_seed"):
            raise ValueError(f"train {replicate} V6 provenance mismatch for probe_root_seed")


def _fit_support_closed(cell: dict[str, Any]) -> bool:
    folds = (cell["probes"][PRIMARY_FAMILY] or {}).get("folds") or []
    return bool(folds) and all(bool(row.get("fit_supported")) for row in folds)


def _heldout_support_closed(cell: dict[str, Any]) -> bool:
    aggregate = (cell["probes"][PRIMARY_FAMILY] or {}).get("aggregate") or {}
    return bool(aggregate.get("support_closed"))


def _direct_utility_improved(cell: dict[str, Any], family: str) -> bool:
    aggregate = (cell["probes"][family] or {}).get("aggregate") or {}
    return bool(aggregate.get("direct_utility_improved"))


def classify_exp279_counterfactual_delta_distill_v6_cross_cell(
    cells: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    missing = [replicate for replicate in DECISION_TRAIN_REPLICATES if replicate not in cells]
    if missing:
        raise ValueError("V6 cross-cell decision requires train 60 and train 120")

    decision_cells = {replicate: cells[replicate] for replicate in DECISION_TRAIN_REPLICATES}
    for replicate, cell in decision_cells.items():
        _validate_cell_boundary(replicate, cell)
        _validate_frozen_configuration(replicate, cell)
    _validate_shared_provenance(decision_cells)

    receipts: dict[str, dict[str, Any]] = {}
    support_closed: dict[str, bool] = {}
    for replicate, cell in decision_cells.items():
        fit_support = _fit_support_closed(cell)
        heldout_support = _heldout_support_closed(cell)
        support = fit_support and heldout_support
        support_closed[str(replicate)] = support
        receipts[str(replicate)] = {
            "artifact_digest": cell.get("artifact_digest"),
            "fit_support_closed": fit_support,
            "heldout_support_closed": heldout_support,
            "support_closed": support,
            "cdd_direct_utility_improved": _direct_utility_improved(cell, PRIMARY_FAMILY),
            "control_direct_utility_improved_descriptive_only": _direct_utility_improved(
                cell, CONTROL_FAMILY
            ),
        }

    if not all(support_closed.values()):
        decision = "INCONCLUSIVE_SUPPORT"
    elif all(
        receipts[str(replicate)]["cdd_direct_utility_improved"]
        for replicate in DECISION_TRAIN_REPLICATES
    ):
        decision = "ROBUST_CDD_BRANCH_COMPLEMENTARITY"
    else:
        decision = "NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY"

    successor_design_authorized = decision == "ROBUST_CDD_BRANCH_COMPLEMENTARITY"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": decision,
        "scientific_evidence_eligible": False,
        "decision_train_replicates": list(DECISION_TRAIN_REPLICATES),
        "primary_family": PRIMARY_FAMILY,
        "descriptive_control_family": CONTROL_FAMILY,
        "control_can_authorize_successor": False,
        "same_primary_family_required_across_decision_cells": True,
        "cell_receipts": receipts,
        "successor_design_authorized": successor_design_authorized,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "analysis_boundary": (
            "augmentation-only DEVELOPMENT cross-cell decision; robust CDD signal authorizes "
            "successor design only and never reserves or consumes fresh evaluation lineage"
        ),
        "artifact_digest": "",
    }
    if decision not in ALLOWED_DECISIONS:
        raise RuntimeError("invalid EXP-279 V6 cross-cell decision")
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload
