from __future__ import annotations

from dataclasses import asdict
from typing import Any

from nolane_ai.model.audit import ModelAudit
from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1"
TARGET_EXPERIMENTS = ("EXP-277", "EXP-279", "EXP-282")

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
}

_ARM_IMPLEMENTATION: dict[tuple[str, str], dict[str, Any]] = {
    ("EXP-277", "arcs_branch"): {
        "implementation_id": "neural_arcs_full_v0_15",
        "implementation_status": "BLOCKED",
        "implementation_tier": "NOT_IMPLEMENTED_FULL_PROTOCOL_ARM",
        "required_regions": ["recurrent_deliberation_core", "verifier_proof_counterexample_heads"],
        "blockers": [
            "full V0.15 ARCS branch bank and verifier-court execution path is not implemented as a matched neural arm",
        ],
    },
    ("EXP-277", "oracle_cbrf"): {
        "implementation_id": "neural_oracle_cbrf_v1",
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
        "blockers": [
            "full neural ARCS branch-search arm is not implemented",
        ],
    },
    ("EXP-279", "hybrid"): {
        "implementation_id": "neural_hybrid_routing_v1",
        "implementation_status": "BLOCKED",
        "implementation_tier": "MATCHED_HYBRID_ROUTER_NOT_IMPLEMENTED",
        "required_regions": ["constraint_belief_fabric", "recurrent_deliberation_core", "verifier_proof_counterexample_heads"],
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
        observed = tuple((str(arm.get("id")), str(arm.get("description"))) for arm in experiments[experiment_id].get("arms", []))
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
        raise ValueError("EXP-277 matched pair audit does not close frozen parameter/oracle-information/compute resource match")


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
        raise ValueError("EXP-282 matched pair audit does not close frozen parameter/observation/compute resource match")


def build_neural_arm_registry(
    *,
    protocol: dict[str, Any],
    protocol_digest: str,
    model_audit: ModelAudit,
    exp277_pair_audit: dict[str, Any] | None = None,
    exp277_execution_artifact: dict[str, Any] | None = None,
    exp282_pair_audit: dict[str, Any] | None = None,
    exp282_execution_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not protocol_digest:
        raise ValueError("protocol_digest is required")
    experiments = _validate_protocol_authority(protocol)

    if exp277_pair_audit is not None:
        _validate_exp277_pair_audit(exp277_pair_audit)
    if exp277_execution_artifact is not None:
        if exp277_pair_audit is None:
            raise ValueError("EXP-277 paired execution artifact requires matched pair audit")
        from .exp277_paired_runner import validate_exp277_paired_development

        execution_errors = validate_exp277_paired_development(exp277_execution_artifact)
        if execution_errors:
            raise ValueError("EXP-277 paired execution artifact is invalid: " + "; ".join(execution_errors))
        if exp277_execution_artifact.get("protocol_digest") != protocol_digest:
            raise ValueError("EXP-277 paired execution artifact protocol digest mismatch")

    if exp282_pair_audit is not None:
        _validate_exp282_pair_audit(exp282_pair_audit)
    if exp282_execution_artifact is not None:
        if exp282_pair_audit is None:
            raise ValueError("EXP-282 paired execution artifact requires matched pair audit")
        from .exp282_paired_runner import validate_exp282_paired_development

        execution_errors = validate_exp282_paired_development(exp282_execution_artifact)
        if execution_errors:
            raise ValueError("EXP-282 paired execution artifact is invalid: " + "; ".join(execution_errors))
        if exp282_execution_artifact.get("protocol_digest") != protocol_digest:
            raise ValueError("EXP-282 paired execution artifact protocol digest mismatch")

    region_counts = {
        name: audit.functional_parameters
        for name, audit in model_audit.regions.items()
    }
    experiment_payload: dict[str, Any] = {}
    for experiment_id in TARGET_EXPERIMENTS:
        spec = experiments[experiment_id]
        arm_payload: dict[str, Any] = {}
        blockers: list[str] = []
        for arm_id, description in _EXPECTED_ARMS[experiment_id]:
            implementation = dict(_ARM_IMPLEMENTATION[(experiment_id, arm_id)])
            implementation["blockers"] = list(implementation["blockers"])
            if experiment_id == "EXP-277" and exp277_pair_audit is not None:
                implementation["implementation_status"] = "IMPLEMENTED"
                implementation["implementation_tier"] = "MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM"
                implementation["blockers"] = []
            if experiment_id == "EXP-282" and exp282_pair_audit is not None:
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

        if experiment_id == "EXP-277" and exp277_pair_audit is not None:
            if exp277_execution_artifact is not None:
                blockers = [
                    "EXP-277: confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution remain open"
                ]
                development_match_status = "PAIRED_STRUCTURE_DENSE_DEV_READY"
            else:
                blockers = [
                    "EXP-277: matched arms are not yet integrated into paired structure-dense evaluator lineage"
                ]
                development_match_status = "PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED"
            ledger = exp277_pair_audit["compute_ledger"]
            experiment_payload[experiment_id] = {
                "protocol_arm_ids": [arm_id for arm_id, _ in _EXPECTED_ARMS[experiment_id]],
                "resource_match_contract": dict(spec.get("resource_match") or {}),
                "resource_match_evidence": {
                    "parameter_match": True,
                    "functional_parameter_match": True,
                    "oracle_information_separation": True,
                    "compute_budget_closed": True,
                    "declared_max_accounted_flops_per_episode": int(exp277_pair_audit["declared_max_accounted_flops_per_episode"]),
                    "arcs_accounted_flops_per_episode": int(ledger["arcs_branch"]["accounted_flops_per_episode"]),
                    "oracle_accounted_flops_per_episode": int(ledger["oracle_cbrf"]["accounted_flops_per_episode"]),
                    "pair_audit_digest": canonical_sha256(exp277_pair_audit),
                },
                "development_match_status": development_match_status,
                "arms": arm_payload,
                "blockers": blockers,
                "match_court": "BLOCKED",
            }
            if exp277_execution_artifact is not None:
                aggregate = exp277_execution_artifact["evaluation"]["aggregate"]
                experiment_payload[experiment_id]["paired_execution_evidence"] = {
                    "artifact_digest": exp277_execution_artifact["artifact_digest"],
                    "code_digest": exp277_execution_artifact["code_digest"],
                    "training_replicates": int(exp277_execution_artifact["training"]["replicates"]),
                    "evaluation_replicates": int(exp277_execution_artifact["evaluation"]["replicates"]),
                    "mean_arcs_utility": float(aggregate["mean_arcs_utility"]),
                    "mean_oracle_utility": float(aggregate["mean_oracle_utility"]),
                    "oracle_relative_utility_gain": float(aggregate["oracle_relative_utility_gain"]),
                    "oracle_minus_arcs_verified_solution_rate": float(aggregate["oracle_minus_arcs_verified_solution_rate"]),
                }
        elif experiment_id == "EXP-282" and exp282_pair_audit is not None:
            if exp282_execution_artifact is not None:
                blockers = [
                    "EXP-282: confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution remain open"
                ]
                development_match_status = "PAIRED_PARTIAL_OBSERVABILITY_DEV_READY"
            else:
                blockers = [
                    "EXP-282: matched arms are not yet integrated into paired partial-observability checkpoint/evaluator lineage"
                ]
                development_match_status = "PARAMETER_AND_COMPUTE_MATCH_CLOSED"
            experiment_payload[experiment_id] = {
                "protocol_arm_ids": [arm_id for arm_id, _ in _EXPECTED_ARMS[experiment_id]],
                "resource_match_contract": dict(spec.get("resource_match") or {}),
                "resource_match_evidence": {
                    "parameter_match": True,
                    "observation_history_match": True,
                    "accounted_flop_match": True,
                    "primitive_operation_match": True,
                    "relative_accounted_flop_difference": float(exp282_pair_audit["relative_accounted_flop_difference"]),
                    "pair_audit_digest": canonical_sha256(exp282_pair_audit),
                },
                "development_match_status": development_match_status,
                "arms": arm_payload,
                "blockers": blockers,
                "match_court": "BLOCKED",
            }
            if exp282_execution_artifact is not None:
                experiment_payload[experiment_id]["paired_execution_evidence"] = {
                    "artifact_digest": exp282_execution_artifact["artifact_digest"],
                    "code_digest": exp282_execution_artifact["code_digest"],
                    "training_replicates": int(exp282_execution_artifact["training"]["replicates"]),
                    "evaluation_replicates": int(exp282_execution_artifact["evaluation"]["replicates"]),
                    "mean_accuracy_gain": float(exp282_execution_artifact["evaluation"]["aggregate"]["mean_accuracy_gain"]),
                    "mean_brier_difference": float(exp282_execution_artifact["evaluation"]["aggregate"]["mean_brier_difference"]),
                }
        else:
            experiment_payload[experiment_id] = {
                "protocol_arm_ids": [arm_id for arm_id, _ in _EXPECTED_ARMS[experiment_id]],
                "resource_match_contract": dict(spec.get("resource_match") or {}),
                "arms": arm_payload,
                "blockers": blockers,
                "match_court": "BLOCKED" if blockers else "CONFIRMATORY_READY",
            }

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


def validate_neural_arm_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid neural arm registry schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("neural arm registry cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("neural arm registry cannot promote a scientific claim")
    if payload.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1" or payload.get("protocol_status") != "FROZEN_V1":
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

    exp277 = experiments.get("EXP-277") or {}
    if "development_match_status" in exp277:
        if exp277.get("match_court") != "BLOCKED":
            errors.append("EXP-277 development evidence cannot open confirmatory match court")
        evidence = exp277.get("resource_match_evidence") or {}
        for key in ("parameter_match", "functional_parameter_match", "oracle_information_separation", "compute_budget_closed"):
            if evidence.get(key) is not True:
                errors.append(f"EXP-277 registry resource evidence {key} is not closed")
        arms = exp277.get("arms") or {}
        if any((arms.get(arm_id) or {}).get("implementation_status") != "IMPLEMENTED" for arm_id in ("arcs_branch", "oracle_cbrf")):
            errors.append("EXP-277 development evidence requires both neural arms implemented")
        status = exp277.get("development_match_status")
        has_execution = "paired_execution_evidence" in exp277
        if has_execution and status != "PAIRED_STRUCTURE_DENSE_DEV_READY":
            errors.append("EXP-277 paired execution evidence/status mismatch")
        if not has_execution and status != "PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED":
            errors.append("EXP-277 pair-audit evidence/status mismatch")
        if not (exp277.get("blockers") or []):
            errors.append("EXP-277 development evidence cannot clear confirmatory blockers")

    if payload.get("registry_digest") not in (None, "") and payload["registry_digest"] != _digest(payload):
        errors.append("neural arm registry digest mismatch")
    return errors
