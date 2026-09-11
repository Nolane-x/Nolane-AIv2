from __future__ import annotations

from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp279_paired_runner import PRIMARY_METRIC, run_exp279_paired_development
from .exp279_rescue_likelihood_ratio import run_exp279_rescue_likelihood_ratio_development


SCHEMA = "NLM-EXP-279-RESCUE-LIKELIHOOD-RATIO-PAIR-DEV-V1"


def _evaluation_batch_digests(payload: dict[str, Any]) -> list[str]:
    return [str(row["paired_batch_digest"]) for row in payload["evaluation"]["per_replicate"]]


def _route_fraction(payload: dict[str, Any]) -> float:
    rows = payload["evaluation"]["per_replicate"]
    routed = sum(int(row["hybrid_route_receipt"]["routed_episodes"]) for row in rows)
    total = sum(int(row["hybrid_route_receipt"]["batch_size"]) for row in rows)
    return routed / total if total else 0.0


def run_exp279_rescue_likelihood_ratio_pair_development(
    *,
    lineage_role: str,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = run_exp279_paired_development(**kwargs)
    intervention = run_exp279_rescue_likelihood_ratio_development(**kwargs)

    identity = {
        "initial_state_match": baseline["initial_state"] == intervention["initial_state"],
        "training_batch_lineage_match": (
            baseline["training"]["paired_batch_digests"]
            == intervention["training"]["paired_batch_digests"]
        ),
        "evaluation_batch_lineage_match": (
            _evaluation_batch_digests(baseline) == _evaluation_batch_digests(intervention)
        ),
        "pair_audit_digest_match": (
            baseline["resource_match"]["pair_audit_digest"]
            == intervention["resource_match"]["pair_audit_digest"]
        ),
        "propagation_final_state_match": (
            baseline["final_state"]["propagation_only_digest"]
            == intervention["final_state"]["propagation_only_digest"]
        ),
        "branch_final_state_match": (
            baseline["final_state"]["branch_only_digest"]
            == intervention["final_state"]["branch_only_digest"]
        ),
    }
    if not all(identity.values()):
        failed = [key for key, value in identity.items() if value is not True]
        raise RuntimeError("paired likelihood-ratio control identity failed: " + ", ".join(failed))

    baseline_agg = baseline["evaluation"]["aggregate"]
    intervention_agg = intervention["evaluation"]["aggregate"]
    diagnostic = intervention["evaluation"]["routing_discrimination"]
    baseline_utility = float(baseline_agg["mean_hybrid_utility"])
    intervention_utility = float(intervention_agg["mean_hybrid_utility"])
    comparison = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "experiment_id": "EXP-279",
        "protocol_digest": baseline["protocol_digest"],
        "code_digest": baseline["code_digest"],
        "lineage_role": lineage_role,
        "training_replicates": baseline["training"]["replicates"],
        "evaluation_start_replicate": baseline["evaluation"]["start_replicate"],
        "evaluation_replicates": baseline["evaluation"]["replicates"],
        "route_threshold": baseline["route_config"]["threshold"],
        "control_identity": identity,
        "baseline": {
            "artifact_digest": baseline["artifact_digest"],
            "route_fraction": _route_fraction(baseline),
            "mean_hybrid_utility": baseline_utility,
        },
        "likelihood_ratio": {
            "artifact_digest": intervention["artifact_digest"],
            "route_fraction": _route_fraction(intervention),
            "mean_hybrid_utility": intervention_utility,
            "natural_rescue_prevalence": diagnostic["natural_rescue_prevalence"],
            "routed_precision": diagnostic["routed_precision"],
            "routed_episodes": diagnostic["routed_episodes"],
            "routed_rescues": diagnostic["routed_rescues"],
        },
        "delta_likelihood_ratio_minus_baseline": {
            "mean_hybrid_utility": intervention_utility - baseline_utility,
        },
        "primary_metric": PRIMARY_METRIC,
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
        "analysis_boundary": "DEVELOPMENT paired identity court only; cannot promote EXP-279",
        "comparison_digest": "",
    }
    comparison["comparison_digest"] = canonical_sha256(
        {key: value for key, value in comparison.items() if key != "comparison_digest"}
    )
    return baseline, intervention, comparison
