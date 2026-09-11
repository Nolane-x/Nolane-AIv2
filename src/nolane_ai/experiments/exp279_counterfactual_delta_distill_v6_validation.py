from __future__ import annotations

from typing import Any

from .exp279_counterfactual_delta_distill_v6_primitives import CONTROL_FAMILY, PRIMARY_FAMILY, SCHEMA


def court_classification(primary: dict[str, Any]) -> str:
    aggregate = primary["aggregate"]
    if not bool(aggregate["support_closed"]):
        return "INCONCLUSIVE_SUPPORT"
    if bool(aggregate["direct_utility_improved"]):
        return "CDD_DIRECT_UTILITY_POSITIVE"
    return "CDD_NOT_DIRECTLY_ECONOMIC"


def validate_artifact(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("V6 schema mismatch")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("V6 evidence boundary mismatch")
    if payload.get("scientific_evidence_eligible") is not False:
        errors.append("V6 cannot be scientific-evidence eligible")
    if payload.get("canonical_model_frozen_for_cdd") is not True:
        errors.append("V6 canonical model must be frozen")
    for key in (
        "fresh_evaluation_lineage_may_be_reserved",
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        if payload.get(key) is not False:
            errors.append(f"V6 {key} must remain false")

    boundary = payload.get("data_boundary") or {}
    if boundary.get("training_rng_stream") != "augmentation" or boundary.get("probe_rng_stream") != "augmentation":
        errors.append("V6 streams must be augmentation-only")
    if boundary.get("evaluation_rng_stream_used") is not False or boundary.get("evaluation_targets_used") is not False:
        errors.append("V6 evaluation boundary violated")
    if boundary.get("heldout_branch_hidden_used_for_features") is not False:
        errors.append("V6 heldout branch hidden leaked into features")
    if boundary.get("probe_root_seed") == payload.get("root_seed"):
        errors.append("V6 probe root must be independent")

    teacher = payload.get("teacher_rule") or {}
    if teacher.get("teacher_uses_fit_partition_only") is not True:
        errors.append("V6 teacher must be fit-only")
    if teacher.get("teacher_training_flops_excluded_from_deployment_utility") is not True:
        errors.append("V6 teacher compute boundary changed")
    if teacher.get("teacher_target") != "mean_detached_branch_state_minus_stop_state_over_variables":
        errors.append("V6 teacher target changed")

    cost = payload.get("cdd_cost_rule") or {}
    if cost.get("cdd_inference_flops_charged_to_primary_utility") is not True:
        errors.append("V6 CDD inference FLOPs must be charged")
    if cost.get("metric") != "direct_counterfactual_verified_solution_per_total_accounted_flop":
        errors.append("V6 primary metric changed")

    config = payload.get("probe_config") or {}
    if config.get("primary_family") != PRIMARY_FAMILY or config.get("descriptive_control_family") != CONTROL_FAMILY:
        errors.append("V6 probe family identity changed")
    if config.get("folds") != 3:
        errors.append("V6 folds changed")
    if config.get("raw_scores_compared_across_folds") is not False or config.get("raw_scores_exported") is not False:
        errors.append("V6 raw score boundary violated")
    if config.get("fold_route_k_rule") != "heldout_fold_true_rescue_count":
        errors.append("V6 route cardinality changed")

    probes = payload.get("probes") or {}
    if set(probes) != {PRIMARY_FAMILY, CONTROL_FAMILY}:
        errors.append("V6 probe receipts incomplete")
    else:
        primary = probes[PRIMARY_FAMILY]
        expected_authorization = bool(primary.get("aggregate", {}).get("direct_utility_improved"))
        if primary.get("authorizes_successor") is not expected_authorization:
            errors.append("V6 primary authorization receipt mismatch")
        if probes[CONTROL_FAMILY].get("authorizes_successor") is not False:
            errors.append("V6 control cannot authorize successor")
        for kind, probe in probes.items():
            if probe.get("raw_scores_compared_across_folds") is not False:
                errors.append(f"V6 {kind} crossed fold score scales")
            for fold in probe.get("folds") or []:
                if fold.get("raw_scores_exported") is not False:
                    errors.append(f"V6 {kind} exported raw scores")

    allowed = {"CDD_DIRECT_UTILITY_POSITIVE", "CDD_NOT_DIRECTLY_ECONOMIC", "INCONCLUSIVE_SUPPORT"}
    classification = payload.get("court_classification")
    if classification not in allowed:
        errors.append("V6 classification invalid")
    elif PRIMARY_FAMILY in probes and classification != court_classification(probes[PRIMARY_FAMILY]):
        errors.append("V6 classification receipt mismatch")
    return errors
