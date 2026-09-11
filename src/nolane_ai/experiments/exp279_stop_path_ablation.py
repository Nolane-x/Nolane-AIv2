from __future__ import annotations

import math
from typing import Any

ARTIFACT_SCHEMA = "NLM-EXP-279-PAIRED-DEV-EVAL-V1"
RECEIPT_SCHEMA = "NLM-EXP-279-STOP-PATH-ABLATION-DEV-V1"
SWEEP_SCHEMA = "NLM-EXP-279-STOP-PATH-SWEEP-DEV-V1"
PRIMARY_METRIC = "verified_utility_per_accounted_flop_on_structure_dense_stratum"
_TOLERANCE = 1e-12


def _finite_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"EXP-279 stop-path ablation {label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"EXP-279 stop-path ablation {label} must be finite")
    return number


def _constant_positive(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"EXP-279 stop-path ablation {label} lineage is empty")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"EXP-279 stop-path ablation {label} must be positive")
    reference = values[0]
    if any(abs(value - reference) > _TOLERANCE for value in values[1:]):
        raise ValueError(f"EXP-279 stop-path ablation {label} must be fixed across replicates")
    return reference


def _close(left: float, right: float, *, tolerance: float = _TOLERANCE) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _classification(*, accuracy_factor: float, compute_factor: float, observed_factor: float) -> str:
    if observed_factor <= 1.0 + _TOLERANCE:
        return "NO_POSITIVE_STOP_PATH_ADVANTAGE"
    if compute_factor > 1.0 + _TOLERANCE and accuracy_factor <= 1.0 + _TOLERANCE:
        return "COMPUTE_DOMINANT_STOP_PATH_ADVANTAGE"
    if compute_factor > 1.0 + _TOLERANCE and accuracy_factor > 1.0 + _TOLERANCE:
        return "COMPUTE_AND_ACCURACY_COMBINE"
    return "MIXED_STOP_PATH_EFFECT"


def _require_development_boundary(payload: dict[str, Any]) -> None:
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("EXP-279 stop-path ablation requires a paired DEVELOPMENT artifact")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-279 stop-path ablation requires EV-E2 / UNVERIFIED input")
    if payload.get("challenge_materialized") is not False:
        raise ValueError("EXP-279 stop-path ablation refuses materialized confirmatory challenge data")
    if payload.get("confirmatory_data_consumed") is not False:
        raise ValueError("EXP-279 stop-path ablation refuses confirmatory data consumption")


def build_exp279_stop_path_ablation(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("EXP-279 stop-path ablation input must be a JSON object")
    _require_development_boundary(payload)

    training = payload.get("training") or {}
    evaluation = payload.get("evaluation") or {}
    rows = list(evaluation.get("per_replicate") or [])
    if not rows:
        raise ValueError("EXP-279 stop-path ablation requires non-empty evaluation lineage")
    eval_count = int(evaluation.get("replicates", 0) or 0)
    if eval_count != len(rows):
        raise ValueError("EXP-279 stop-path ablation evaluation count mismatch")

    propagation_costs: list[float] = []
    hybrid_costs: list[float] = []
    propagation_solution_rates: list[float] = []
    hybrid_solution_rates: list[float] = []
    propagation_utilities: list[float] = []
    hybrid_utilities: list[float] = []
    route_fractions: list[float] = []

    for index, row in enumerate(rows):
        receipt = row.get("hybrid_route_receipt") or {}
        routed = int(receipt.get("routed_episodes", -1))
        route_fraction = _finite_float(
            receipt.get("route_fraction"), label=f"row {index} route fraction"
        )
        mask = list(receipt.get("branch_route_mask") or [])
        if routed != 0 or abs(route_fraction) > _TOLERANCE or any(bool(item) for item in mask):
            raise ValueError(
                "EXP-279 stop-path ablation requires zero hybrid routing in every evaluation replicate"
            )
        route_fractions.append(route_fraction)

        propagation = row.get("propagation_only") or {}
        hybrid = row.get("hybrid") or {}
        propagation_cost = _finite_float(
            propagation.get("accounted_flops_per_episode"),
            label=f"row {index} propagation accounted FLOPs",
        )
        hybrid_cost = _finite_float(
            hybrid.get("accounted_flops_per_episode"),
            label=f"row {index} hybrid accounted FLOPs",
        )
        propagation_solution = _finite_float(
            propagation.get("verified_solution_rate"),
            label=f"row {index} propagation solution rate",
        )
        hybrid_solution = _finite_float(
            hybrid.get("verified_solution_rate"),
            label=f"row {index} hybrid solution rate",
        )
        if propagation_solution < 0.0 or hybrid_solution < 0.0:
            raise ValueError("EXP-279 stop-path ablation solution rates must be non-negative")

        propagation_utility = _finite_float(
            propagation.get(PRIMARY_METRIC),
            label=f"row {index} propagation utility",
        )
        hybrid_utility = _finite_float(
            hybrid.get(PRIMARY_METRIC),
            label=f"row {index} hybrid utility",
        )
        if not _close(propagation_utility, propagation_solution / propagation_cost):
            raise ValueError("EXP-279 stop-path ablation propagation utility/cost identity drift")
        if not _close(hybrid_utility, hybrid_solution / hybrid_cost):
            raise ValueError("EXP-279 stop-path ablation hybrid utility/cost identity drift")

        propagation_costs.append(propagation_cost)
        hybrid_costs.append(hybrid_cost)
        propagation_solution_rates.append(propagation_solution)
        hybrid_solution_rates.append(hybrid_solution)
        propagation_utilities.append(propagation_utility)
        hybrid_utilities.append(hybrid_utility)

    propagation_cost = _constant_positive(
        propagation_costs, label="propagation accounted FLOPs"
    )
    hybrid_stop_cost = _constant_positive(hybrid_costs, label="hybrid stop accounted FLOPs")

    n = len(rows)
    mean_prop_solution = sum(propagation_solution_rates) / n
    mean_hybrid_solution = sum(hybrid_solution_rates) / n
    mean_prop_utility = sum(propagation_utilities) / n
    mean_hybrid_utility = sum(hybrid_utilities) / n
    if mean_prop_solution <= 0.0 or mean_prop_utility <= 0.0:
        raise ValueError(
            "EXP-279 stop-path ablation requires positive propagation baseline solution rate and utility"
        )

    accuracy_factor = mean_hybrid_solution / mean_prop_solution
    compute_factor = propagation_cost / hybrid_stop_cost
    reconstructed_factor = accuracy_factor * compute_factor
    observed_factor = mean_hybrid_utility / mean_prop_utility
    observed_gain = observed_factor - 1.0
    equal_cost_gain = accuracy_factor - 1.0
    compute_only_gain = compute_factor - 1.0
    reconstruction_error = observed_factor - reconstructed_factor
    if not _close(observed_factor, reconstructed_factor):
        raise ValueError("EXP-279 stop-path ablation utility reconstruction does not close")

    aggregate = evaluation.get("aggregate") or {}
    aggregate_expectations = {
        "mean_propagation_solution_rate": mean_prop_solution,
        "mean_hybrid_solution_rate": mean_hybrid_solution,
        "mean_propagation_utility": mean_prop_utility,
        "mean_hybrid_utility": mean_hybrid_utility,
        "hybrid_relative_utility_gain": observed_gain,
    }
    for key, expected in aggregate_expectations.items():
        if key not in aggregate:
            raise ValueError(f"EXP-279 stop-path ablation source aggregate missing {key}")
        observed = _finite_float(aggregate.get(key), label=f"aggregate {key}")
        if not _close(observed, expected):
            raise ValueError(f"EXP-279 stop-path ablation source aggregate {key} drift")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "analysis_scope": "development_posthoc_zero_route_stop_path_cost_decomposition",
        "source_artifact_schema": payload.get("schema"),
        "source_artifact_digest": payload.get("artifact_digest"),
        "protocol_id": payload.get("protocol_id"),
        "protocol_digest": payload.get("protocol_digest"),
        "code_digest": payload.get("code_digest"),
        "training_replicates": int(training.get("replicates", 0) or 0),
        "training_start_replicate": int(training.get("start_replicate", 0) or 0),
        "evaluation_start_replicate": int(evaluation.get("start_replicate", 0) or 0),
        "evaluation_replicates": eval_count,
        "all_hybrid_routes_zero": True,
        "mean_route_fraction": sum(route_fractions) / len(route_fractions),
        "max_route_fraction": max(route_fractions),
        "propagation_accounted_flops_per_episode": propagation_cost,
        "hybrid_stop_accounted_flops_per_episode": hybrid_stop_cost,
        "mean_propagation_solution_rate": mean_prop_solution,
        "mean_hybrid_solution_rate": mean_hybrid_solution,
        "mean_propagation_utility": mean_prop_utility,
        "mean_hybrid_utility": mean_hybrid_utility,
        "accuracy_factor": accuracy_factor,
        "compute_factor": compute_factor,
        "reconstructed_utility_factor": reconstructed_factor,
        "observed_utility_factor": observed_factor,
        "utility_reconstruction_error": reconstruction_error,
        "observed_relative_utility_gain": observed_gain,
        "equal_cost_counterfactual_relative_gain": equal_cost_gain,
        "compute_only_counterfactual_relative_gain": compute_only_gain,
        "mechanism_classification": _classification(
            accuracy_factor=accuracy_factor,
            compute_factor=compute_factor,
            observed_factor=observed_factor,
        ),
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
    }
    errors = validate_exp279_stop_path_ablation(receipt)
    if errors:
        raise RuntimeError("invalid EXP-279 stop-path ablation receipt: " + "; ".join(errors))
    return receipt


def validate_exp279_stop_path_ablation(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append("EXP-279 stop-path ablation schema drift")
    if receipt.get("evidence_level") != "EV-E2" or receipt.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 stop-path ablation evidence/decision boundary drift")
    if receipt.get("scientific_evidence_eligible") is not False:
        errors.append("EXP-279 stop-path ablation must remain scientifically ineligible")
    if receipt.get("all_hybrid_routes_zero") is not True:
        errors.append("EXP-279 stop-path ablation zero-route precondition drift")
    if receipt.get("challenge_materialized") is not False:
        errors.append("EXP-279 stop-path ablation challenge boundary drift")
    if receipt.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-279 stop-path ablation confirmatory boundary drift")
    if receipt.get("promotion_claimed") is not False:
        errors.append("EXP-279 stop-path ablation promotion boundary drift")

    try:
        propagation_cost = float(receipt.get("propagation_accounted_flops_per_episode"))
        hybrid_cost = float(receipt.get("hybrid_stop_accounted_flops_per_episode"))
        accuracy_factor = float(receipt.get("accuracy_factor"))
        compute_factor = float(receipt.get("compute_factor"))
        reconstructed = float(receipt.get("reconstructed_utility_factor"))
        observed = float(receipt.get("observed_utility_factor"))
        observed_gain = float(receipt.get("observed_relative_utility_gain"))
        equal_cost_gain = float(receipt.get("equal_cost_counterfactual_relative_gain"))
        compute_only_gain = float(receipt.get("compute_only_counterfactual_relative_gain"))
        reconstruction_error = float(receipt.get("utility_reconstruction_error"))
    except (TypeError, ValueError):
        errors.append("EXP-279 stop-path ablation numeric receipt fields invalid")
        return errors

    numeric_values = (
        propagation_cost,
        hybrid_cost,
        accuracy_factor,
        compute_factor,
        reconstructed,
        observed,
        observed_gain,
        equal_cost_gain,
        compute_only_gain,
        reconstruction_error,
    )
    if any(not math.isfinite(value) for value in numeric_values):
        errors.append("EXP-279 stop-path ablation numeric receipt fields must be finite")
        return errors
    if propagation_cost <= 0.0 or hybrid_cost <= 0.0 or accuracy_factor < 0.0 or compute_factor <= 0.0:
        errors.append("EXP-279 stop-path ablation cost/factor domain invalid")
    if not _close(reconstructed, accuracy_factor * compute_factor):
        errors.append("EXP-279 stop-path ablation reconstruction factor mismatch")
    if not _close(observed, reconstructed):
        errors.append("EXP-279 stop-path ablation observed/reconstruction mismatch")
    if not _close(observed_gain, observed - 1.0):
        errors.append("EXP-279 stop-path ablation relative gain mismatch")
    if not _close(equal_cost_gain, accuracy_factor - 1.0):
        errors.append("EXP-279 stop-path ablation equal-cost counterfactual mismatch")
    if not _close(compute_only_gain, compute_factor - 1.0):
        errors.append("EXP-279 stop-path ablation compute-only counterfactual mismatch")
    if not _close(reconstruction_error, observed - reconstructed):
        errors.append("EXP-279 stop-path ablation reconstruction error mismatch")
    expected_classification = _classification(
        accuracy_factor=accuracy_factor,
        compute_factor=compute_factor,
        observed_factor=observed,
    )
    if receipt.get("mechanism_classification") != expected_classification:
        errors.append("EXP-279 stop-path ablation mechanism classification mismatch")
    if abs(float(receipt.get("mean_route_fraction", 1.0))) > _TOLERANCE:
        errors.append("EXP-279 stop-path ablation mean route fraction must remain zero")
    if abs(float(receipt.get("max_route_fraction", 1.0))) > _TOLERANCE:
        errors.append("EXP-279 stop-path ablation max route fraction must remain zero")
    return errors


def build_exp279_stop_path_sweep(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("EXP-279 stop-path sweep requires at least one DEVELOPMENT artifact")
    analyses = [build_exp279_stop_path_ablation(payload) for payload in payloads]
    analyses.sort(key=lambda item: int(item["training_replicates"]))

    counts = [int(item["training_replicates"]) for item in analyses]
    if len(set(counts)) != len(counts):
        raise ValueError("EXP-279 stop-path sweep training replicate counts must be unique")
    lineages = {
        (int(item["evaluation_start_replicate"]), int(item["evaluation_replicates"]))
        for item in analyses
    }
    if len(lineages) != 1:
        raise ValueError("EXP-279 stop-path sweep evaluation lineage must match across inputs")
    protocols = {(item.get("protocol_id"), item.get("protocol_digest")) for item in analyses}
    if len(protocols) != 1:
        raise ValueError("EXP-279 stop-path sweep protocol lineage must match across inputs")

    eval_start, eval_count = next(iter(lineages))
    return {
        "schema": SWEEP_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "analysis_scope": "development_posthoc_zero_route_stop_path_cost_decomposition_sweep",
        "training_replicate_counts": counts,
        "evaluation_lineage": {
            "start_replicate": eval_start,
            "replicates": eval_count,
        },
        "protocol_id": analyses[0].get("protocol_id"),
        "protocol_digest": analyses[0].get("protocol_digest"),
        "all_hybrid_routes_zero": all(bool(item["all_hybrid_routes_zero"]) for item in analyses),
        "mechanism_classifications": [item["mechanism_classification"] for item in analyses],
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
        "analyses": analyses,
    }
