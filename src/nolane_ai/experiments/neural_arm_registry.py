from __future__ import annotations

from dataclasses import asdict
from typing import Any

from nolane_ai.model.audit import ModelAudit
from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1"
TARGET_EXPERIMENTS = ("EXP-277", "EXP-279", "EXP-282", "EXP-286")

_EXPECTED_ARMS: dict[str, tuple[tuple[str, str], ...]] = {
    "EXP-277": (
        ("arcs_branch", "V0.15 ARCS recurrent-depth + branch bank + verifier court"),
        ("oracle_cbrf", "same substrate with ground-truth constraint/factor representation"),
    ),
    "EXP-279": (
        ("propagation_only", "constraint propagation without branch search"),
        ("branch_only", "ARCS branch search without CBRF propagation"),
        ("hybrid", "propagation followed by branch search when residual uncertainty remains"),
    ),
    "EXP-282": (
        ("recurrent_hidden", "matched recurrent state without explicit belief representation"),
        ("explicit_belief", "explicit calibrated belief state over hidden world variables"),
    ),
    "EXP-286": (
        ("chronological_failure", "no conflict core; chronological rollback"),
        ("oracle_conflict_core", "ground-truth conflict core supplied at contradiction"),
    ),
}

_ARM_IMPLEMENTATION: dict[tuple[str, str], dict[str, Any]] = {
    ("EXP-277", "arcs_branch"): {
        "implementation_id": "exp277_matched_arcs_dev_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "NOT_IMPLEMENTED_FULL_PROTOCOL_ARM",
        "required_regions": ["recurrent_deliberation_core", "verifier_proof_counterexample_heads"],
        "blockers": [
            "full V0.15 ARCS branch bank and verifier-court execution path is not implemented as a matched neural arm",
        ],
    },
    ("EXP-277", "oracle_cbrf"): {
        "implementation_id": "exp277_matched_oracle_cbrf_dev_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "ORACLE_NEURAL_ARM_NOT_WIRED",
        "required_regions": ["constraint_belief_fabric", "verifier_proof_counterexample_heads"],
        "blockers": [
            "ground-truth constraint/factor oracle injection is not wired into the exact-16M neural execution path",
        ],
    },
    ("EXP-279", "propagation_only"): {
        "implementation_id": "neural_propagation_only_v1",
        "implementation_status": "DEVELOPMENT_COMPONENT_PRESENT",
        "implementation_tier": "NEURAL_COMPONENT_WITHOUT_CONFIRMATORY_ACCOUNTING",
        "required_regions": ["constraint_belief_fabric"],
        "blockers": [
            "neural accounted-FLOP utility runner and reclaimed-parameter matching against simpler rivals are not frozen",
        ],
    },
    ("EXP-279", "branch_only"): {
        "implementation_id": "neural_branch_only_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "FULL_BRANCH_ARM_NOT_IMPLEMENTED",
        "required_regions": ["recurrent_deliberation_core", "verifier_proof_counterexample_heads"],
        "blockers": ["full neural ARCS branch-search arm is not implemented"],
    },
    ("EXP-279", "hybrid"): {
        "implementation_id": "neural_hybrid_routing_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "MATCHED_HYBRID_ROUTER_NOT_IMPLEMENTED",
        "required_regions": [
            "constraint_belief_fabric",
            "recurrent_deliberation_core",
            "verifier_proof_counterexample_heads",
        ],
        "blockers": [
            "matched neural routing policy with predeclared structure-fit strata and equal accounted-FLOP budget is not implemented",
        ],
    },
    ("EXP-282", "recurrent_hidden"): {
        "implementation_id": "neural_recurrent_hidden_matched_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "MATCHED_BASELINE_NOT_IMPLEMENTED",
        "required_regions": ["recurrent_deliberation_core"],
        "blockers": [
            "matched recurrent-hidden decision head with equal state/controller parameter and compute budget is not implemented",
        ],
    },
    ("EXP-282", "explicit_belief"): {
        "implementation_id": "neural_explicit_belief_dev_v1",
        "implementation_status": "DEVELOPMENT_COMPONENT_PRESENT",
        "implementation_tier": "NEURAL_DEVELOPMENT_COMPONENT",
        "required_regions": ["constraint_belief_fabric"],
        "blockers": [
            "matched recurrent-hidden comparator and equal state/controller parameter/compute accounting are not closed",
        ],
    },
    ("EXP-286", "chronological_failure"): {
        "implementation_id": "exp286_matched_chronological_failure_dev_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "MATCHED_BASELINE_NOT_IMPLEMENTED",
        "required_regions": [],
        "blockers": [
            "matched chronological rollback baseline is not yet admitted by the EXP-286 development pair audit",
        ],
    },
    ("EXP-286", "oracle_conflict_core"): {
        "implementation_id": "exp286_matched_oracle_conflict_core_dev_v1",
        "implementation_status": "DEVELOPMENT_COMPONENT_PRESENT",
        "implementation_tier": "ORACLE_DEVELOPMENT_COMPONENT_WITHOUT_MATCHED_EXECUTION_COURT",
        "required_regions": [],
        "blockers": [
            "oracle conflict-core component exists only as development machinery until matched pair and execution evidence validate it",
        ],
    },
}


def _digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("registry_digest", None)
    return canonical_sha256(clean)


def _protocol_experiment_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("experiment_id")): item
        for item in protocol.get("experiments", [])
        if item.get("experiment_id") in TARGET_EXPERIMENTS
    }


def _validate_protocol_authority(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if protocol.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1":
        raise ValueError("neural arm registry requires frozen Stage-A v1 protocol")
    if protocol.get("status") != "FROZEN_V1":
        raise ValueError("neural arm registry requires FROZEN_V1 protocol status")
    experiments = _protocol_experiment_map(protocol)
    if set(experiments) != set(TARGET_EXPERIMENTS):
        raise ValueError("protocol experiment drift for neural arm registry")
    for experiment_id in TARGET_EXPERIMENTS:
        observed = tuple(
            (str(arm.get("id")), str(arm.get("description")))
            for arm in experiments[experiment_id].get("arms", [])
        )
        if observed != _EXPECTED_ARMS[experiment_id]:
            raise ValueError(f"protocol arm drift for {experiment_id}")
    return experiments


def _validate_exp277_pair_audit(pair_audit: dict[str, Any]) -> None:
    required_true = (
        pair_audit.get("schema") == "NLM-EXP-277-MATCHED-ARMS-DEV-V1",
        pair_audit.get("evidence_level") == "EV-E2",
        pair_audit.get("decision") == "UNVERIFIED",
        pair_audit.get("parameter_match") is True,
        pair_audit.get("functional_parameter_match") is True,
        pair_audit.get("oracle_information_separation") is True,
        pair_audit.get("compute_budget_closed") is True,
    )
    ledger = pair_audit.get("compute_ledger") or {}
    arcs = ledger.get("arcs_branch") or {}
    oracle = ledger.get("oracle_cbrf") or {}
    ceiling = int(pair_audit.get("declared_max_accounted_flops_per_episode", 0) or 0)
    ledger_closed = (
        ledger.get("schema") == "NLM-EXP-277-COMPUTE-LEDGER-V1"
        and arcs.get("hardware_profiler_flops_claimed") is False
        and oracle.get("hardware_profiler_flops_claimed") is False
        and int(arcs.get("accounted_flops_per_episode", 0) or 0) > 0
        and int(oracle.get("accounted_flops_per_episode", 0) or 0) > 0
        and ceiling > 0
        and int(arcs.get("accounted_flops_per_episode", 0) or 0) <= ceiling
        and int(oracle.get("accounted_flops_per_episode", 0) or 0) <= ceiling
    )
    if not all(required_true) or not ledger_closed:
        raise ValueError(
            "EXP-277 matched pair audit does not close frozen parameter/oracle-information/compute resource match"
        )


def _validate_exp279_pair_audit(pair_audit: dict[str, Any]) -> None:
    expected_strata = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
    required_true = (
        pair_audit.get("schema") == "NLM-EXP-279-MATCHED-ROUTING-ARMS-DEV-V1",
        pair_audit.get("evidence_level") == "EV-E2",
        pair_audit.get("decision") == "UNVERIFIED",
        pair_audit.get("parameter_match") is True,
        pair_audit.get("functional_parameter_match") is True,
        pair_audit.get("active_functional_parameter_match") is True,
        pair_audit.get("optimizer_visible_parameter_match") is True,
        pair_audit.get("reclaimed_parameter_assignment_closed") is True,
        pair_audit.get("compute_budget_closed") is True,
        pair_audit.get("inactive_excluded_reclaimed_parameters") == 0,
        pair_audit.get("structure_fit_strata") == expected_strata,
    )
    ceiling = int(pair_audit.get("declared_max_accounted_flops_per_episode", 0) or 0)
    ledger = pair_audit.get("compute_ledger") or {}
    ledger_closed = ceiling > 0 and set(ledger) == {
        "propagation_only",
        "branch_only",
        "hybrid",
    }
    if ledger_closed:
        for arm_id in ("propagation_only", "branch_only", "hybrid"):
            arm_ledger = ledger.get(arm_id) or {}
            if (
                arm_ledger.get("hardware_profiler_flops_claimed") is not False
                or int(arm_ledger.get("max_accounted_flops_per_episode", 0) or 0) <= 0
                or int(arm_ledger.get("max_accounted_flops_per_episode", 0) or 0) > ceiling
            ):
                ledger_closed = False
                break
        hybrid = ledger.get("hybrid") or {}
        stop_cost = int(hybrid.get("stop_accounted_flops_per_episode", 0) or 0)
        branch_cost = int(hybrid.get("branch_accounted_flops_per_episode", 0) or 0)
        if stop_cost <= 0 or branch_cost < stop_cost or branch_cost > ceiling:
            ledger_closed = False
    try:
        threshold = float(pair_audit.get("route_threshold"))
    except (TypeError, ValueError):
        threshold = -1.0
    if not all(required_true) or not ledger_closed or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "EXP-279 matched triplet audit does not close frozen parameter/reclaim/compute/strata resource match"
        )


def _validate_exp282_pair_audit(pair_audit: dict[str, Any]) -> None:
    required_true = (
        pair_audit.get("schema") == "NLM-EXP-282-MATCHED-BELIEF-ARMS-DEV-V1",
        pair_audit.get("evidence_level") == "EV-E2",
        pair_audit.get("decision") == "UNVERIFIED",
        pair_audit.get("parameter_match") is True,
        pair_audit.get("observation_history_match") is True,
        pair_audit.get("primitive_operation_match") is True,
        pair_audit.get("full_accounted_flop_match") is True,
        float(pair_audit.get("relative_accounted_flop_difference", 1.0)) <= 0.05,
    )
    if not all(required_true):
        raise ValueError(
            "EXP-282 matched pair audit does not close frozen parameter/observation/compute resource match"
        )


def _validate_exp286_pair_audit(pair_audit: dict[str, Any]) -> None:
    receipt = pair_audit.get("oracle_information_receipt") or {}
    expected_receipt = {
        "artifact": "ground_truth_conflict_core",
        "ground_truth": True,
        "delivery_event": "after_current_contradiction_only",
        "delivered_to": ["oracle_conflict_core"],
        "withheld_from": ["chronological_failure"],
        "chronological_failure_received_conflict_core": False,
        "future_conflict_core_leakage": False,
        "solution_leakage": False,
    }
    required_true = (
        pair_audit.get("schema") == "NLM-EXP-286-MATCHED-CONFLICT-ARMS-DEV-V1",
        pair_audit.get("evidence_level") == "EV-E2",
        pair_audit.get("decision") == "UNVERIFIED",
        pair_audit.get("parameter_match") is True,
        pair_audit.get("functional_parameter_match") is True,
        pair_audit.get("active_functional_parameter_match") is True,
        pair_audit.get("optimizer_visible_parameter_match") is True,
        pair_audit.get("oracle_information_separation") is True,
        pair_audit.get("compute_budget_closed") is True,
        receipt == expected_receipt,
    )
    ceiling = int(pair_audit.get("declared_max_accounted_flops_per_episode", 0) or 0)
    ledger = pair_audit.get("compute_ledger") or {}
    ledger_closed = ceiling > 0 and set(ledger) == {
        "chronological_failure",
        "oracle_conflict_core",
    }
    if ledger_closed:
        for arm_id in ("chronological_failure", "oracle_conflict_core"):
            arm_ledger = ledger.get(arm_id) or {}
            per_step = int(arm_ledger.get("accounted_flops_per_search_step", 0) or 0)
            maximum = int(arm_ledger.get("max_accounted_flops_per_episode", 0) or 0)
            if (
                arm_ledger.get("hardware_profiler_flops_claimed") is not False
                or per_step <= 0
                or maximum <= 0
                or maximum > ceiling
            ):
                ledger_closed = False
                break
    if not all(required_true) or not ledger_closed:
        raise ValueError(
            "EXP-286 matched pair audit does not close frozen parameter/compute/oracle-information resource match"
        )


def _validate_execution(
    *,
    experiment_id: str,
    execution_artifact: dict[str, Any] | None,
    pair_audit: dict[str, Any] | None,
    protocol_digest: str,
) -> None:
    if execution_artifact is None:
        return
    if pair_audit is None:
        label = "matched triplet audit" if experiment_id == "EXP-279" else "matched pair audit"
        raise ValueError(f"{experiment_id} paired execution artifact requires {label}")
    if (
        experiment_id == "EXP-286"
        and execution_artifact.get("learned_conflict_localizer_validated") not in (None, False)
    ):
        raise ValueError(
            "EXP-286 paired execution artifact cannot claim learned conflict localizer validation"
        )

    if experiment_id == "EXP-277":
        from .exp277_paired_runner import validate_exp277_paired_development as validator
    elif experiment_id == "EXP-279":
        from .exp279_paired_runner import validate_exp279_paired_development as validator
    elif experiment_id == "EXP-282":
        from .exp282_paired_runner import validate_exp282_paired_development as validator
    elif experiment_id == "EXP-286":
        from .exp286_paired_runner import validate_exp286_paired_development as validator
    else:  # pragma: no cover - guarded by TARGET_EXPERIMENTS
        raise ValueError(f"unsupported neural arm registry experiment {experiment_id}")

    execution_errors = validator(execution_artifact)
    if execution_errors:
        raise ValueError(
            f"{experiment_id} paired execution artifact is invalid: " + "; ".join(execution_errors)
        )
    if execution_artifact.get("protocol_digest") != protocol_digest:
        raise ValueError(f"{experiment_id} paired execution artifact protocol digest mismatch")


def _base_arm_payload(
    *,
    experiment_id: str,
    region_counts: dict[str, int],
    implemented: bool,
) -> tuple[dict[str, Any], list[str]]:
    arm_payload: dict[str, Any] = {}
    blockers: list[str] = []
    for arm_id, description in _EXPECTED_ARMS[experiment_id]:
        implementation = dict(_ARM_IMPLEMENTATION[(experiment_id, arm_id)])
        implementation["blockers"] = list(implementation["blockers"])
        if implemented:
            implementation["implementation_status"] = "IMPLEMENTED"
            implementation["implementation_tier"] = "MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM"
            implementation["blockers"] = []
        implementation["protocol_arm_id"] = arm_id
        implementation["protocol_description"] = description
        implementation["required_region_functional_parameters"] = {
            region: region_counts.get(region, 0)
            for region in implementation["required_regions"]
        }
        arm_payload[arm_id] = implementation
        blockers.extend(f"{arm_id}: {item}" for item in implementation["blockers"])
    return arm_payload, blockers


def build_neural_arm_registry(
    *,
    protocol: dict[str, Any],
    protocol_digest: str,
    model_audit: ModelAudit,
    exp277_pair_audit: dict[str, Any] | None = None,
    exp277_execution_artifact: dict[str, Any] | None = None,
    exp279_pair_audit: dict[str, Any] | None = None,
    exp279_execution_artifact: dict[str, Any] | None = None,
    exp282_pair_audit: dict[str, Any] | None = None,
    exp282_execution_artifact: dict[str, Any] | None = None,
    exp286_pair_audit: dict[str, Any] | None = None,
    exp286_execution_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not protocol_digest:
        raise ValueError("protocol_digest is required")
    experiments = _validate_protocol_authority(protocol)

    pair_audits = {
        "EXP-277": exp277_pair_audit,
        "EXP-279": exp279_pair_audit,
        "EXP-282": exp282_pair_audit,
        "EXP-286": exp286_pair_audit,
    }
    executions = {
        "EXP-277": exp277_execution_artifact,
        "EXP-279": exp279_execution_artifact,
        "EXP-282": exp282_execution_artifact,
        "EXP-286": exp286_execution_artifact,
    }

    if exp277_pair_audit is not None:
        _validate_exp277_pair_audit(exp277_pair_audit)
    if exp279_pair_audit is not None:
        _validate_exp279_pair_audit(exp279_pair_audit)
    if exp282_pair_audit is not None:
        _validate_exp282_pair_audit(exp282_pair_audit)
    if exp286_pair_audit is not None:
        _validate_exp286_pair_audit(exp286_pair_audit)

    for experiment_id in TARGET_EXPERIMENTS:
        _validate_execution(
            experiment_id=experiment_id,
            execution_artifact=executions[experiment_id],
            pair_audit=pair_audits[experiment_id],
            protocol_digest=protocol_digest,
        )

    region_counts = {
        name: audit.functional_parameters
        for name, audit in model_audit.regions.items()
    }
    experiment_payload: dict[str, Any] = {}

    for experiment_id in TARGET_EXPERIMENTS:
        spec = experiments[experiment_id]
        pair_audit = pair_audits[experiment_id]
        execution = executions[experiment_id]
        arm_payload, base_blockers = _base_arm_payload(
            experiment_id=experiment_id,
            region_counts=region_counts,
            implemented=pair_audit is not None,
        )

        if pair_audit is None:
            experiment_payload[experiment_id] = {
                "protocol_arm_ids": [arm_id for arm_id, _ in _EXPECTED_ARMS[experiment_id]],
                "resource_match_contract": dict(spec.get("resource_match") or {}),
                "arms": arm_payload,
                "blockers": base_blockers,
                "match_court": "BLOCKED" if base_blockers else "CONFIRMATORY_READY",
            }
            continue

        if experiment_id == "EXP-277":
            status = (
                "PAIRED_STRUCTURE_DENSE_DEV_READY"
                if execution is not None
                else "PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED"
            )
            blockers = [
                "EXP-277: confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution remain open"
                if execution is not None
                else "EXP-277: matched arms are not yet integrated into paired structure-dense evaluator lineage"
            ]
            ledger = pair_audit["compute_ledger"]
            evidence = {
                "parameter_match": True,
                "functional_parameter_match": True,
                "oracle_information_separation": True,
                "compute_budget_closed": True,
                "declared_max_accounted_flops_per_episode": int(
                    pair_audit["declared_max_accounted_flops_per_episode"]
                ),
                "arcs_accounted_flops_per_episode": int(
                    ledger["arcs_branch"]["accounted_flops_per_episode"]
                ),
                "oracle_accounted_flops_per_episode": int(
                    ledger["oracle_cbrf"]["accounted_flops_per_episode"]
                ),
                "pair_audit_digest": canonical_sha256(pair_audit),
            }
        elif experiment_id == "EXP-279":
            status = (
                "PAIRED_ROUTING_DEV_READY"
                if execution is not None
                else "PARAMETER_RECLAIM_COMPUTE_STRATA_CLOSED"
            )
            blockers = [
                "EXP-279: confirmatory sample-size/blocked-analysis freeze, confirmatory-open execution and post-freeze challenge evidence remain open"
                if execution is not None
                else "EXP-279: matched arms are not yet integrated into paired structure-fit evaluator lineage"
            ]
            evidence = {
                "parameter_match": True,
                "functional_parameter_match": True,
                "active_functional_parameter_match": True,
                "optimizer_visible_parameter_match": True,
                "reclaimed_parameter_assignment_closed": True,
                "compute_budget_closed": True,
                "structure_fit_strata": list(pair_audit["structure_fit_strata"]),
                "declared_max_accounted_flops_per_episode": int(
                    pair_audit["declared_max_accounted_flops_per_episode"]
                ),
                "pair_audit_digest": canonical_sha256(pair_audit),
            }
        elif experiment_id == "EXP-282":
            status = (
                "PAIRED_PARTIAL_OBSERVABILITY_DEV_READY"
                if execution is not None
                else "PARAMETER_AND_COMPUTE_MATCH_CLOSED"
            )
            blockers = [
                "EXP-282: confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution remain open"
                if execution is not None
                else "EXP-282: matched arms are not yet integrated into paired partial-observability checkpoint/evaluator lineage"
            ]
            evidence = {
                "parameter_match": True,
                "observation_history_match": True,
                "accounted_flop_match": True,
                "primitive_operation_match": True,
                "relative_accounted_flop_difference": float(
                    pair_audit["relative_accounted_flop_difference"]
                ),
                "pair_audit_digest": canonical_sha256(pair_audit),
            }
        else:
            status = (
                "PAIRED_CONFLICT_HEADROOM_DEV_READY"
                if execution is not None
                else "PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED"
            )
            blockers = (
                [
                    "EXP-286: confirmatory sample-size/paired-log-cost analysis freeze remains open",
                    "EXP-286: confirmatory-open execution remains unrun",
                    "EXP-286: post-freeze challenge evidence remains unavailable",
                    "EXP-286: learned conflict-core localization remains unvalidated",
                ]
                if execution is not None
                else [
                    "EXP-286: matched arms are not yet integrated into paired conflict-headroom DEVELOPMENT execution lineage"
                ]
            )
            evidence = {
                "parameter_match": True,
                "functional_parameter_match": True,
                "active_functional_parameter_match": True,
                "optimizer_visible_parameter_match": True,
                "compute_budget_closed": True,
                "oracle_information_separation": True,
                "declared_max_accounted_flops_per_episode": int(
                    pair_audit["declared_max_accounted_flops_per_episode"]
                ),
                "pair_audit_digest": canonical_sha256(pair_audit),
            }

        item: dict[str, Any] = {
            "protocol_arm_ids": [arm_id for arm_id, _ in _EXPECTED_ARMS[experiment_id]],
            "resource_match_contract": dict(spec.get("resource_match") or {}),
            "resource_match_evidence": evidence,
            "development_match_status": status,
            "arms": arm_payload,
            "blockers": blockers,
            "match_court": "BLOCKED",
        }

        if execution is not None:
            if experiment_id == "EXP-277":
                aggregate = execution["evaluation"]["aggregate"]
                item["paired_execution_evidence"] = {
                    "artifact_digest": execution["artifact_digest"],
                    "code_digest": execution["code_digest"],
                    "training_replicates": int(execution["training"]["replicates"]),
                    "evaluation_replicates": int(execution["evaluation"]["replicates"]),
                    "mean_arcs_utility": float(aggregate["mean_arcs_utility"]),
                    "mean_oracle_utility": float(aggregate["mean_oracle_utility"]),
                    "oracle_relative_utility_gain": float(aggregate["oracle_relative_utility_gain"]),
                    "oracle_minus_arcs_verified_solution_rate": float(
                        aggregate["oracle_minus_arcs_verified_solution_rate"]
                    ),
                }
            elif experiment_id == "EXP-279":
                aggregate = execution["evaluation"]["aggregate"]
                item["paired_execution_evidence"] = {
                    "artifact_digest": execution["artifact_digest"],
                    "code_digest": execution["code_digest"],
                    "training_replicates": int(execution["training"]["replicates"]),
                    "evaluation_replicates": int(execution["evaluation"]["replicates"]),
                    "best_simple_arm": str(aggregate["best_simple_arm"]),
                    "hybrid_relative_utility_gain": float(aggregate["hybrid_relative_utility_gain"]),
                    "hybrid_minus_best_simple_verified_solution_rate": float(
                        aggregate["hybrid_minus_best_simple_verified_solution_rate"]
                    ),
                }
            elif experiment_id == "EXP-282":
                item["paired_execution_evidence"] = {
                    "artifact_digest": execution["artifact_digest"],
                    "code_digest": execution["code_digest"],
                    "training_replicates": int(execution["training"]["replicates"]),
                    "evaluation_replicates": int(execution["evaluation"]["replicates"]),
                    "mean_accuracy_gain": float(
                        execution["evaluation"]["aggregate"]["mean_accuracy_gain"]
                    ),
                    "mean_brier_difference": float(
                        execution["evaluation"]["aggregate"]["mean_brier_difference"]
                    ),
                }
            else:
                aggregate = execution["evaluation"]["aggregate"]
                item["paired_execution_evidence"] = {
                    "artifact_digest": execution["artifact_digest"],
                    "code_digest": execution["code_digest"],
                    "training_replicates": int(execution["training"]["replicates"]),
                    "evaluation_replicates": int(execution["evaluation"]["replicates"]),
                    "descriptive_relative_flop_reduction": float(
                        aggregate["descriptive_relative_flop_reduction"]
                    ),
                    "oracle_minus_chronological_verified_solution_rate": float(
                        aggregate["oracle_minus_chronological_verified_solution_rate"]
                    ),
                    "chronological_censored_episode_count": int(
                        aggregate["chronological_censored_episode_count"]
                    ),
                    "oracle_censored_episode_count": int(
                        aggregate["oracle_censored_episode_count"]
                    ),
                    "learned_conflict_localizer_validated": False,
                }

        experiment_payload[experiment_id] = item

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scope": "stage-a-neural-arm-implementation-and-resource-match-audit",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "protocol_digest": protocol_digest,
        "pilot_total_parameters": model_audit.total_parameters,
        "pilot_functional_parameters": model_audit.functional_parameters,
        "pilot_reserved_parameters": model_audit.reserved_parameters,
        "region_functional_parameters": region_counts,
        "region_audit": {name: asdict(audit) for name, audit in model_audit.regions.items()},
        "experiments": experiment_payload,
        "registry_digest": "",
    }
    payload["registry_digest"] = _digest(payload)
    errors = validate_neural_arm_registry(payload)
    if errors:
        raise RuntimeError("invalid neural arm registry: " + "; ".join(errors))
    return payload


def _validate_development_item(
    errors: list[str],
    *,
    experiment_id: str,
    item: dict[str, Any],
    required_evidence: tuple[str, ...],
    implemented_arms: tuple[str, ...],
    pair_only_status: str,
    execution_status: str,
) -> None:
    if "development_match_status" not in item:
        return
    if item.get("match_court") != "BLOCKED":
        errors.append(f"{experiment_id} development evidence cannot open confirmatory match court")
    evidence = item.get("resource_match_evidence") or {}
    for key in required_evidence:
        if evidence.get(key) is not True:
            errors.append(f"{experiment_id} registry resource evidence {key} is not closed")
    arms = item.get("arms") or {}
    if any(
        (arms.get(arm_id) or {}).get("implementation_status") != "IMPLEMENTED"
        for arm_id in implemented_arms
    ):
        errors.append(f"{experiment_id} development evidence requires all matched neural arms implemented")
    status = item.get("development_match_status")
    has_execution = "paired_execution_evidence" in item
    if has_execution and status != execution_status:
        errors.append(f"{experiment_id} paired execution evidence/status mismatch")
    if not has_execution and status != pair_only_status:
        errors.append(f"{experiment_id} pair-audit evidence/status mismatch")
    if not (item.get("blockers") or []):
        errors.append(f"{experiment_id} development evidence cannot clear confirmatory blockers")


def validate_neural_arm_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid neural arm registry schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("neural arm registry cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("neural arm registry cannot promote a scientific claim")
    if (
        payload.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
        or payload.get("protocol_status") != "FROZEN_V1"
    ):
        errors.append("neural arm registry protocol authority mismatch")
    if int(payload.get("pilot_total_parameters", 0) or 0) != 16_000_000:
        errors.append("neural arm registry requires exact 16M pilot budget")

    experiments = payload.get("experiments") or {}
    if set(experiments) != set(TARGET_EXPERIMENTS):
        errors.append("neural arm registry experiment set mismatch")
    for experiment_id in TARGET_EXPERIMENTS:
        item = experiments.get(experiment_id) or {}
        expected_ids = [arm_id for arm_id, _ in _EXPECTED_ARMS[experiment_id]]
        if item.get("protocol_arm_ids") != expected_ids:
            errors.append(f"{experiment_id} protocol arm IDs mismatch")
        blockers = item.get("blockers") or []
        if item.get("match_court") == "CONFIRMATORY_READY" and blockers:
            errors.append(f"{experiment_id} cannot be confirmatory-ready while blockers remain")
        if item.get("match_court") == "CONFIRMATORY_READY":
            arms = item.get("arms") or {}
            if any(arm.get("implementation_status") != "IMPLEMENTED" for arm in arms.values()):
                errors.append(f"{experiment_id} cannot be confirmatory-ready with incomplete arms")

    _validate_development_item(
        errors,
        experiment_id="EXP-277",
        item=experiments.get("EXP-277") or {},
        required_evidence=(
            "parameter_match",
            "functional_parameter_match",
            "oracle_information_separation",
            "compute_budget_closed",
        ),
        implemented_arms=("arcs_branch", "oracle_cbrf"),
        pair_only_status="PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED",
        execution_status="PAIRED_STRUCTURE_DENSE_DEV_READY",
    )
    _validate_development_item(
        errors,
        experiment_id="EXP-279",
        item=experiments.get("EXP-279") or {},
        required_evidence=(
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "reclaimed_parameter_assignment_closed",
            "compute_budget_closed",
        ),
        implemented_arms=("propagation_only", "branch_only", "hybrid"),
        pair_only_status="PARAMETER_RECLAIM_COMPUTE_STRATA_CLOSED",
        execution_status="PAIRED_ROUTING_DEV_READY",
    )
    exp279 = experiments.get("EXP-279") or {}
    if "development_match_status" in exp279:
        evidence = exp279.get("resource_match_evidence") or {}
        if evidence.get("structure_fit_strata") != [
            "PROPAGATION_FIT",
            "BRANCH_FIT",
            "MIXED_RESIDUAL",
        ]:
            errors.append("EXP-279 registry structure-fit strata drift")

    _validate_development_item(
        errors,
        experiment_id="EXP-282",
        item=experiments.get("EXP-282") or {},
        required_evidence=("parameter_match", "observation_history_match", "accounted_flop_match"),
        implemented_arms=("recurrent_hidden", "explicit_belief"),
        pair_only_status="PARAMETER_AND_COMPUTE_MATCH_CLOSED",
        execution_status="PAIRED_PARTIAL_OBSERVABILITY_DEV_READY",
    )
    _validate_development_item(
        errors,
        experiment_id="EXP-286",
        item=experiments.get("EXP-286") or {},
        required_evidence=(
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "compute_budget_closed",
            "oracle_information_separation",
        ),
        implemented_arms=("chronological_failure", "oracle_conflict_core"),
        pair_only_status="PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED",
        execution_status="PAIRED_CONFLICT_HEADROOM_DEV_READY",
    )
    exp286 = experiments.get("EXP-286") or {}
    paired286 = exp286.get("paired_execution_evidence") or {}
    if paired286 and paired286.get("learned_conflict_localizer_validated") is not False:
        errors.append("EXP-286 development evidence cannot validate a learned conflict localizer")

    if payload.get("registry_digest") not in (None, "") and payload["registry_digest"] != _digest(payload):
        errors.append("neural arm registry digest mismatch")
    return errors
