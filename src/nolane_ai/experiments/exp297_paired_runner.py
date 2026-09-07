from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from nolane_ai.experiments.exp297_fidelity_worlds import generate_fidelity_world
from nolane_ai.experiments.matched_fidelity_arms import (
    audit_matched_fidelity_arms,
    build_matched_fidelity_arms,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.reasoning.fidelity import FidelityCourt, compile_valid

SCHEMA = "NLM-EXP-297-PAIRED-DEV-EVAL-V1"
FALSE_FLAGS = (
    "confirmatory_ready",
    "confirmatory_data_consumed",
    "challenge_seed_materialized",
    "challenge_materialized",
    "decision_rule_executed",
    "hidden_trap_family_consumed",
    "semantic_authority_promoted",
)


def _digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _arm_metrics(rows: list[dict[str, Any]], arm_id: str) -> dict[str, Any]:
    faithful = [row for row in rows if row["is_faithful"]]
    wrong = [row for row in rows if not row["is_faithful"]]
    tp = sum(bool(row["arms"][arm_id]["authority_granted"]) for row in faithful)
    fn = len(faithful) - tp
    fp = sum(bool(row["arms"][arm_id]["authority_granted"]) for row in wrong)
    tn = len(wrong) - fp
    tpr = tp / len(faithful) if faithful else 0.0
    tnr = tn / len(wrong) if wrong else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "faithful_count": len(faithful),
        "wrong_count": len(wrong),
        "semantic_fidelity_balanced_accuracy": 0.5 * (tpr + tnr),
        "wrong_formalization_authority_rate": fp / len(wrong) if wrong else 0.0,
        "faithful_formalization_rejection_rate": fn / len(faithful) if faithful else 0.0,
    }


def _decision_payload(decision, *, semantic_ops: int) -> dict[str, Any]:
    return {
        "authority_granted": bool(decision.authority_granted),
        "fidelity_score": decision.fidelity_score,
        "authority_score": decision.authority_score,
        "verifier_score": decision.verifier_score,
        "receipt_semantics": decision.receipt_semantics,
        "arm_observable_digest": decision.arm_observable_digest,
        "neural_accounted_flops": int(decision.neural_accounted_flops),
        "compile_validation_operations": 1,
        "semantic_verification_operations": int(semantic_ops),
        "total_accounted_cost_proxy": int(
            decision.neural_accounted_flops + 1 + semantic_ops
        ),
        "hardware_profiler_flops_claimed": False,
    }


def run_exp297_paired_development(
    *,
    root_seed: str,
    eval_replicates: int,
    eval_start_replicate: int,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    max_exact_assignments: int,
    protocol_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    if eval_replicates <= 0 or eval_start_replicate < 0:
        raise ValueError("invalid EXP-297 replicate geometry")
    if max_exact_assignments <= 0:
        raise ValueError("max_exact_assignments must be positive")

    model_seed = derive_stream_seed(root_seed, "EXP-297", 0, "model_init")
    torch.manual_seed(model_seed)
    compile_arm, fidelity_arm = build_matched_fidelity_arms(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
    )
    pair_audit = audit_matched_fidelity_arms(compile_arm, fidelity_arm)
    court = FidelityCourt(max_exact_assignments=max_exact_assignments)
    raw: list[dict[str, Any]] = []

    for offset in range(eval_replicates):
        replicate = eval_start_replicate + offset
        environment_seed = derive_stream_seed(
            root_seed,
            "EXP-297",
            replicate,
            "environment",
        )
        batch = generate_fidelity_world(environment_seed)
        for candidate_index, case in enumerate(batch.candidates):
            compiled = compile_valid(case.candidate)
            receipt = court.adjudicate(batch.source, case.candidate)
            compile_decision = compile_arm.decide(
                batch.source,
                case.candidate,
                compile_valid=compiled,
            )
            fidelity_decision = fidelity_arm.decide(
                batch.source,
                case.candidate,
                compile_valid=compiled,
                court_receipt=receipt,
            )
            raw.append(
                {
                    "replicate": replicate,
                    "environment_seed": environment_seed,
                    "candidate_index": candidate_index,
                    "candidate_id": case.candidate_id,
                    "stratum": case.stratum,
                    "is_faithful": case.is_faithful,
                    "source_digest": batch.source_digest,
                    "candidate_digest": case.candidate_digest,
                    "compile_valid": compiled,
                    "court_receipt": receipt.as_dict(),
                    "arm_input_receipt": {
                        "candidate_set_frozen_before_arms": True,
                        "byte_identical_candidate_order": True,
                        "compile_only_candidate_digest": case.candidate_digest,
                        "fidelity_court_candidate_digest": case.candidate_digest,
                        "evaluator_truth_in_causal_path": False,
                        "trap_family_in_causal_path": False,
                    },
                    "arms": {
                        "compile_only": _decision_payload(
                            compile_decision,
                            semantic_ops=0,
                        ),
                        "fidelity_court": _decision_payload(
                            fidelity_decision,
                            semantic_ops=receipt.semantic_verification_operations,
                        ),
                    },
                }
            )

    aggregate = {
        "compile_only": _arm_metrics(raw, "compile_only"),
        "fidelity_court": _arm_metrics(raw, "fidelity_court"),
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": "EXP-297",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_seed": root_seed,
        "model_init_seed": model_seed,
        "config": {
            "eval_replicates": eval_replicates,
            "eval_start_replicate": eval_start_replicate,
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
            "max_exact_assignments": max_exact_assignments,
        },
        "resource_match": {"pair_audit": pair_audit},
        "evaluation": {"raw_candidates": raw, "aggregate": aggregate},
        "measurement_boundary": (
            "EV-E2_ANALYTICAL_NEURAL_FLOPS_PLUS_EXACT_SEMANTIC_OPERATIONS_"
            "NOT_HARDWARE_PROFILED_FLOPS"
        ),
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_seed_materialized": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "hidden_trap_family_consumed": False,
        "semantic_authority_promoted": False,
    }
    payload["artifact_digest"] = _digest(payload)
    return payload


def validate_exp297_execution(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("EXP-297 schema mismatch")
    if (
        payload.get("experiment_id") != "EXP-297"
        or payload.get("evidence_level") != "EV-E2"
        or payload.get("decision") != "UNVERIFIED"
    ):
        errors.append("EXP-297 epistemic identity mismatch")
    for flag in FALSE_FLAGS:
        if payload.get(flag) is not False:
            errors.append(f"EXP-297 forbidden flag enabled: {flag}")
    if payload.get("artifact_digest") != _digest(payload):
        errors.append("EXP-297 artifact digest mismatch")

    config = payload.get("config") or {}
    try:
        eval_replicates = int(config["eval_replicates"])
        eval_start = int(config["eval_start_replicate"])
        max_exact = int(config["max_exact_assignments"])
    except (KeyError, TypeError, ValueError):
        errors.append("EXP-297 invalid config")
        return errors

    rows = (payload.get("evaluation") or {}).get("raw_candidates")
    if not isinstance(rows, list):
        errors.append("EXP-297 raw candidate rows missing")
        return errors

    expected_rows: list[tuple[Any, Any, Any, int, int]] = []
    root_seed = str(payload.get("root_seed", ""))
    court = FidelityCourt(max_exact_assignments=max_exact)
    for offset in range(eval_replicates):
        replicate = eval_start + offset
        environment_seed = derive_stream_seed(
            root_seed,
            "EXP-297",
            replicate,
            "environment",
        )
        batch = generate_fidelity_world(environment_seed)
        for index, case in enumerate(batch.candidates):
            expected_rows.append(
                (
                    batch,
                    case,
                    court.adjudicate(batch.source, case.candidate),
                    replicate,
                    index,
                )
            )

    if len(rows) != len(expected_rows):
        errors.append("EXP-297 candidate row count/order cardinality mismatch")
        return errors

    for row, expected in zip(rows, expected_rows, strict=True):
        batch, case, receipt, replicate, index = expected
        if (
            row.get("replicate") != replicate
            or row.get("environment_seed") != batch.seed
            or row.get("candidate_index") != index
            or row.get("candidate_id") != case.candidate_id
        ):
            errors.append("EXP-297 candidate lineage/order mismatch")
        if (
            row.get("stratum") != case.stratum
            or row.get("is_faithful") is not case.is_faithful
        ):
            errors.append("EXP-297 construction provenance mismatch")
        if (
            row.get("source_digest") != batch.source_digest
            or row.get("candidate_digest") != case.candidate_digest
        ):
            errors.append("EXP-297 source/candidate digest mismatch")

        compiled = compile_valid(case.candidate)
        if row.get("compile_valid") is not compiled:
            errors.append("EXP-297 compile result mismatch")
        if row.get("court_receipt") != receipt.as_dict():
            errors.append("EXP-297 reconstructed witness receipt mismatch")

        arm_input = row.get("arm_input_receipt") or {}
        if arm_input != {
            "candidate_set_frozen_before_arms": True,
            "byte_identical_candidate_order": True,
            "compile_only_candidate_digest": case.candidate_digest,
            "fidelity_court_candidate_digest": case.candidate_digest,
            "evaluator_truth_in_causal_path": False,
            "trap_family_in_causal_path": False,
        }:
            errors.append("EXP-297 arm input identity/leakage receipt mismatch")

        arms = row.get("arms") or {}
        compile_data = arms.get("compile_only") or {}
        fidelity_data = arms.get("fidelity_court") or {}
        if compile_data.get("authority_granted") is not bool(compiled):
            errors.append("EXP-297 compile-only authority mismatch")
        expected_fidelity_authority = bool(
            compiled and receipt.decision == "court_accept"
        )
        if fidelity_data.get("authority_granted") is not expected_fidelity_authority:
            errors.append("EXP-297 fidelity authority mismatch")
        if int(compile_data.get("semantic_verification_operations", -1)) != 0:
            errors.append("EXP-297 compile-only semantic cost mismatch")
        if (
            int(fidelity_data.get("semantic_verification_operations", -1))
            != receipt.semantic_verification_operations
        ):
            errors.append("EXP-297 fidelity semantic cost mismatch")

        for arm_data, semantic_ops in (
            (compile_data, 0),
            (fidelity_data, receipt.semantic_verification_operations),
        ):
            neural = int(arm_data.get("neural_accounted_flops", -1))
            if (
                arm_data.get("hardware_profiler_flops_claimed") is not False
                or int(arm_data.get("compile_validation_operations", -1)) != 1
            ):
                errors.append("EXP-297 cost boundary mismatch")
            if (
                int(arm_data.get("total_accounted_cost_proxy", -1))
                != neural + 1 + semantic_ops
            ):
                errors.append("EXP-297 total accounted cost mismatch")

    expected_aggregate = {
        "compile_only": _arm_metrics(rows, "compile_only"),
        "fidelity_court": _arm_metrics(rows, "fidelity_court"),
    }
    observed_aggregate = (payload.get("evaluation") or {}).get("aggregate")
    if observed_aggregate != expected_aggregate:
        errors.append("EXP-297 aggregate reconstruction mismatch")

    pair = (payload.get("resource_match") or {}).get("pair_audit") or {}
    required_pair = (
        pair.get("schema") == "NLM-EXP-297-MATCHED-FIDELITY-ARMS-DEV-V1",
        pair.get("evidence_level") == "EV-E2",
        pair.get("decision") == "UNVERIFIED",
        pair.get("parameter_match") is True,
        pair.get("functional_parameter_match") is True,
        pair.get("active_functional_parameter_match") is True,
        pair.get("optimizer_visible_parameter_match") is True,
        pair.get("initialization_match") is True,
        pair.get("neural_accounted_flops_match") is True,
        pair.get("label_information_consumed") is False,
        pair.get("hidden_trap_family_consumed") is False,
    )
    if not all(required_pair):
        errors.append("EXP-297 matched neural pair audit invalid")
    return errors
