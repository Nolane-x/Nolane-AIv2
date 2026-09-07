from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.experiments.exp297_paired_runner import validate_exp297_execution
from nolane_ai.experiments.neural_arm_registry import validate_neural_arm_registry
from nolane_ai.protocol.evidence import canonical_sha256

EXPECTED_ARMS = (
    (
        "compile_only",
        "formal compile/proof validity without semantic-fidelity court",
    ),
    (
        "fidelity_court",
        "compile-valid formalization plus bidirectional semantic fidelity checks",
    ),
)
PAIR_SCHEMA = "NLM-EXP-297-MATCHED-FIDELITY-ARMS-DEV-V1"
PAIR_ONLY_STATUS = "PARAMETER_COMPUTE_FIDELITY_COURT_CLOSED"
EXECUTION_STATUS = "PAIRED_FIDELITY_COURT_DEV_READY"


def _registry_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("registry_digest", None)
    return canonical_sha256(clean)


def _protocol_exp297(protocol: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in protocol.get("experiments", [])
        if item.get("experiment_id") == "EXP-297"
    ]
    if len(matches) != 1:
        raise ValueError("frozen protocol must contain exactly one EXP-297")
    spec = matches[0]
    observed = tuple(
        (str(arm.get("id")), str(arm.get("description")))
        for arm in spec.get("arms", [])
    )
    if observed != EXPECTED_ARMS:
        raise ValueError("frozen EXP-297 arm contract drift")
    return spec


def _validate_pair(pair: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema": PAIR_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "parameter_match": True,
        "functional_parameter_match": True,
        "active_functional_parameter_match": True,
        "optimizer_visible_parameter_match": True,
        "initialization_match": True,
        "neural_accounted_flops_match": True,
        "label_information_consumed": False,
        "hidden_trap_family_consumed": False,
    }
    for key, value in required.items():
        if pair.get(key) != value:
            errors.append(f"EXP-297 pair audit {key} mismatch")
    ledger = pair.get("compute_ledger") or {}
    if set(ledger) != {"compile_only", "fidelity_court"}:
        errors.append("EXP-297 pair compute ledger arm mismatch")
    else:
        compile_ledger = ledger.get("compile_only") or {}
        fidelity_ledger = ledger.get("fidelity_court") or {}
        compile_flops = int(compile_ledger.get("neural_accounted_flops", 0) or 0)
        fidelity_flops = int(fidelity_ledger.get("neural_accounted_flops", 0) or 0)
        if compile_flops <= 0 or compile_flops != fidelity_flops:
            errors.append("EXP-297 neural accounted FLOPs are not matched")
        if compile_ledger.get("semantic_verification_operations") != 0:
            errors.append("EXP-297 compile-only arm cannot consume semantic verification")
        if fidelity_ledger.get("semantic_verification_operations") != "charged_per_candidate_from_receipt":
            errors.append("EXP-297 fidelity semantic verification cost policy mismatch")
        if (
            compile_ledger.get("hardware_profiler_flops_claimed") is not False
            or fidelity_ledger.get("hardware_profiler_flops_claimed") is not False
        ):
            errors.append("EXP-297 development ledger cannot claim hardware-profiled FLOPs")
    return errors


def extend_neural_arm_registry_with_exp297(
    *,
    base_registry: dict[str, Any],
    protocol: dict[str, Any],
    exp297_pair_audit: dict[str, Any] | None = None,
    exp297_execution_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_errors = validate_neural_arm_registry(base_registry)
    if base_errors:
        raise ValueError("base neural arm registry invalid: " + "; ".join(base_errors))
    spec = _protocol_exp297(protocol)

    pair_errors: list[str] = []
    if exp297_pair_audit is not None:
        pair_errors = _validate_pair(exp297_pair_audit)
        if pair_errors:
            raise ValueError("invalid EXP-297 pair audit: " + "; ".join(pair_errors))

    if exp297_execution_artifact is not None:
        execution_errors = validate_exp297_execution(exp297_execution_artifact)
        if execution_errors:
            raise ValueError("invalid EXP-297 execution: " + "; ".join(execution_errors))
        execution_pair = (
            exp297_execution_artifact.get("resource_match") or {}
        ).get("pair_audit")
        if exp297_pair_audit is None or execution_pair != exp297_pair_audit:
            raise ValueError("EXP-297 execution/pair audit binding mismatch")

    if exp297_execution_artifact is not None:
        status = EXECUTION_STATUS
        blockers = [
            "confirmatory sample size is not frozen/materialized",
            "future-beacon hidden semantic challenge is not materialized",
            "frozen decision rule has not executed",
            "semantic formal authority remains restricted",
        ]
    elif exp297_pair_audit is not None:
        status = PAIR_ONLY_STATUS
        blockers = [
            "paired EXP-297 development execution artifact is not supplied",
            "confirmatory sample size is not frozen/materialized",
            "future-beacon hidden semantic challenge is not materialized",
            "semantic formal authority remains restricted",
        ]
    else:
        status = "BLOCKED"
        blockers = [
            "matched EXP-297 neural pair audit is not supplied",
            "paired EXP-297 development execution artifact is not supplied",
            "semantic formal authority remains restricted",
        ]

    evidence = {
        "parameter_match": bool(exp297_pair_audit and exp297_pair_audit.get("parameter_match") is True),
        "functional_parameter_match": bool(exp297_pair_audit and exp297_pair_audit.get("functional_parameter_match") is True),
        "active_functional_parameter_match": bool(exp297_pair_audit and exp297_pair_audit.get("active_functional_parameter_match") is True),
        "optimizer_visible_parameter_match": bool(exp297_pair_audit and exp297_pair_audit.get("optimizer_visible_parameter_match") is True),
        "initialization_match": bool(exp297_pair_audit and exp297_pair_audit.get("initialization_match") is True),
        "neural_accounted_flops_match": bool(exp297_pair_audit and exp297_pair_audit.get("neural_accounted_flops_match") is True),
        "label_information_separation": bool(exp297_pair_audit and exp297_pair_audit.get("label_information_consumed") is False),
        "hidden_trap_family_separation": bool(exp297_pair_audit and exp297_pair_audit.get("hidden_trap_family_consumed") is False),
        "pair_audit_digest": canonical_sha256(exp297_pair_audit) if exp297_pair_audit is not None else None,
    }

    execution_evidence: dict[str, Any] | None = None
    if exp297_execution_artifact is not None:
        aggregate = exp297_execution_artifact["evaluation"]["aggregate"]
        execution_evidence = {
            "execution_digest": exp297_execution_artifact["artifact_digest"],
            "pair_audit_digest": canonical_sha256(exp297_pair_audit),
            "descriptive_aggregate": deepcopy(aggregate),
            "confirmatory_ready": False,
            "confirmatory_data_consumed": False,
            "challenge_seed_materialized": False,
            "challenge_materialized": False,
            "decision_rule_executed": False,
            "hidden_trap_family_consumed": False,
            "semantic_authority_promoted": False,
        }

    payload = deepcopy(base_registry)
    if "EXP-297" in (payload.get("experiments") or {}):
        raise ValueError("base registry unexpectedly already contains EXP-297")
    payload["base_registry_digest"] = base_registry["registry_digest"]
    payload["experiments"]["EXP-297"] = {
        "protocol_arm_ids": [arm_id for arm_id, _ in EXPECTED_ARMS],
        "resource_match_contract": deepcopy(spec.get("resource_match") or {}),
        "resource_match_evidence": evidence,
        "development_match_status": status,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "arms": {
            "compile_only": {
                "implementation_id": "exp297_matched_compile_only_dev_v1",
                "implementation_status": "DEVELOPMENT_COMPONENT_PRESENT",
                "implementation_tier": "MATCHED_NEURAL_DEVELOPMENT_ARM",
                "required_regions": ["encoding_fidelity_court"],
            },
            "fidelity_court": {
                "implementation_id": "exp297_matched_fidelity_court_dev_v1",
                "implementation_status": "DEVELOPMENT_COMPONENT_PRESENT",
                "implementation_tier": "MATCHED_NEURAL_PLUS_EXACT_WITNESS_DEVELOPMENT_ARM",
                "required_regions": ["encoding_fidelity_court"],
            },
        },
        "paired_execution_evidence": execution_evidence,
        "blockers": blockers,
        "match_court": "BLOCKED",
    }
    payload["registry_digest"] = _registry_digest(payload)
    errors = validate_exp297_neural_arm_registry(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 neural arm registry: " + "; ".join(errors))
    return payload


def validate_exp297_neural_arm_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("registry_digest") != _registry_digest(payload):
        errors.append("EXP-297 extended registry digest mismatch")

    base_digest = payload.get("base_registry_digest")
    base = deepcopy(payload)
    base.pop("base_registry_digest", None)
    experiments = base.get("experiments")
    if not isinstance(experiments, dict) or "EXP-297" not in experiments:
        errors.append("EXP-297 registry experiment missing")
        return errors
    experiments.pop("EXP-297", None)
    base["registry_digest"] = base_digest
    if not isinstance(base_digest, str) or not base_digest:
        errors.append("EXP-297 base registry digest missing")
    else:
        base_clean = deepcopy(base)
        base_clean.pop("registry_digest", None)
        if canonical_sha256(base_clean) != base_digest:
            errors.append("EXP-297 base registry binding mismatch")
        errors.extend(validate_neural_arm_registry(base))

    item = (payload.get("experiments") or {}).get("EXP-297") or {}
    if item.get("protocol_arm_ids") != ["compile_only", "fidelity_court"]:
        errors.append("EXP-297 protocol arm IDs mismatch")
    if item.get("evidence_level") != "EV-E2" or item.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 registry epistemic status mismatch")
    if item.get("match_court") != "BLOCKED":
        errors.append("EXP-297 development evidence cannot open confirmatory match court")

    status = item.get("development_match_status")
    if status not in {"BLOCKED", PAIR_ONLY_STATUS, EXECUTION_STATUS}:
        errors.append("EXP-297 development status invalid")
    evidence = item.get("resource_match_evidence") or {}
    if status in {PAIR_ONLY_STATUS, EXECUTION_STATUS}:
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "initialization_match",
            "neural_accounted_flops_match",
            "label_information_separation",
            "hidden_trap_family_separation",
        ):
            if evidence.get(key) is not True:
                errors.append(f"EXP-297 registry resource evidence {key} is not closed")
        if not evidence.get("pair_audit_digest"):
            errors.append("EXP-297 pair audit digest missing")

    execution = item.get("paired_execution_evidence")
    if status == EXECUTION_STATUS:
        if not isinstance(execution, dict):
            errors.append("EXP-297 paired execution evidence missing")
        else:
            if execution.get("pair_audit_digest") != evidence.get("pair_audit_digest"):
                errors.append("EXP-297 execution/pair digest binding mismatch")
            execution_digest = execution.get("execution_digest")
            if (
                not isinstance(execution_digest, str)
                or len(execution_digest) != 64
                or any(character not in "0123456789abcdef" for character in execution_digest)
            ):
                errors.append("EXP-297 execution digest invalid")
            for flag in (
                "confirmatory_ready",
                "confirmatory_data_consumed",
                "challenge_seed_materialized",
                "challenge_materialized",
                "decision_rule_executed",
                "hidden_trap_family_consumed",
                "semantic_authority_promoted",
            ):
                if execution.get(flag) is not False:
                    errors.append(f"EXP-297 registry forbidden execution flag enabled: {flag}")
            aggregate = execution.get("descriptive_aggregate") or {}
            fidelity = aggregate.get("fidelity_court") or {}
            compile_only = aggregate.get("compile_only") or {}
            for arm in (fidelity, compile_only):
                for metric in (
                    "semantic_fidelity_balanced_accuracy",
                    "wrong_formalization_authority_rate",
                    "faithful_formalization_rejection_rate",
                ):
                    value = arm.get(metric)
                    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                        errors.append(f"EXP-297 descriptive aggregate {metric} invalid")
    elif execution is not None:
        errors.append("EXP-297 execution evidence present without execution-ready development status")

    arms = item.get("arms") or {}
    if set(arms) != {"compile_only", "fidelity_court"}:
        errors.append("EXP-297 registry arm implementation set mismatch")
    return errors
