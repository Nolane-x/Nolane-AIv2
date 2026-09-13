from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Callable

import torch

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralFidelityCourt
from nolane_ai.experiments.exp298_causal_worlds import (
    CausalDiagnosisSemantics,
    generate_exp298_causal_world,
)
from nolane_ai.experiments.exp298_code_worlds import (
    CodeInvariantSemantics,
    generate_exp298_code_world,
)
from nolane_ai.experiments.exp298_language_worlds import (
    GroundedLanguageSemantics,
    generate_exp298_language_world,
)
from nolane_ai.experiments.matched_cross_domain_fidelity_arms import (
    audit_matched_cross_domain_fidelity_arms,
    build_matched_cross_domain_fidelity_arms,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed

SCHEMA = "NLM-EXP-298-CROSS-DOMAIN-FIDELITY-ROOT-V1"
DOMAIN_ORDER = (
    "code_invariant",
    "causal_diagnosis",
    "grounded_language_ambiguity",
)
DOMAIN_OFFSETS = {
    "code_invariant": 0,
    "causal_diagnosis": 100_000,
    "grounded_language_ambiguity": 200_000,
}
FALSE_FLAGS = (
    "scientific_evidence_eligible",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
    "unrestricted_semantic_authority_claimed",
    "open_language_understanding_claimed",
    "causal_discovery_claimed",
    "general_code_reasoning_claimed",
    "exp290_authority_inherited",
    "exp291_296_authority_inherited",
)


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _state_digest(module: torch.nn.Module) -> str:
    digest = sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(bytes(value.untyped_storage()))
    return digest.hexdigest()


def _arm_payload(decision: Any, *, probes: int, domain_ops: int) -> dict[str, Any]:
    return {
        "authority_granted": bool(decision.authority_granted),
        "fidelity_score": float(decision.fidelity_score),
        "authority_score": float(decision.authority_score),
        "verifier_score": float(decision.verifier_score),
        "receipt_semantics": decision.receipt_semantics,
        "arm_observable_digest": decision.arm_observable_digest,
        "neural_accounted_flops": int(decision.neural_accounted_flops),
        "compile_validation_operations": 1,
        "semantic_probe_evaluations": int(probes),
        "semantic_domain_operations": int(domain_ops),
        "total_accounted_cost_proxy": int(
            decision.neural_accounted_flops + 1 + probes + domain_ops
        ),
        "hardware_profiler_flops_claimed": False,
    }


def _arm_metrics(rows: list[dict[str, Any]], arm_id: str) -> dict[str, Any]:
    faithful = [row for row in rows if row["is_faithful"]]
    wrong = [row for row in rows if not row["is_faithful"]]
    tp = sum(bool(row["arms"][arm_id]["authority_granted"]) for row in faithful)
    fp = sum(bool(row["arms"][arm_id]["authority_granted"]) for row in wrong)
    fn = len(faithful) - tp
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


def _domain_factory(domain_id: str) -> tuple[Callable[[int], Any], Callable[[Any], Any]]:
    if domain_id == "code_invariant":
        return generate_exp298_code_world, lambda source: CodeInvariantSemantics(source)
    if domain_id == "causal_diagnosis":
        return generate_exp298_causal_world, lambda source: CausalDiagnosisSemantics(source)
    if domain_id == "grounded_language_ambiguity":
        return generate_exp298_language_world, lambda source: GroundedLanguageSemantics(source)
    raise ValueError(f"unknown EXP-298 domain: {domain_id}")


def _validate_identity_hex(name: str, value: Any, length: int) -> str | None:
    if not isinstance(value, str) or len(value) != length:
        return f"EXP-298 {name} identity invalid"
    try:
        bytes.fromhex(value)
    except ValueError:
        return f"EXP-298 {name} identity invalid"
    return None


def run_exp298_root(
    *,
    root_seed: str,
    canonical_index: int,
    eval_replicates: int,
    eval_start_replicate: int,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    max_exact_probes: int,
    protocol_digest: str,
    geometry_digest: str,
    code_digest: str,
    repository_head: str,
) -> dict[str, Any]:
    if canonical_index not in (0, 1, 2, 3):
        raise ValueError("canonical_index must be one of 0,1,2,3")
    if eval_replicates <= 0 or eval_start_replicate < 0:
        raise ValueError("invalid EXP-298 evaluation geometry")
    if min(d_model, hidden_size, target_parameters, max_exact_probes) <= 0:
        raise ValueError("invalid EXP-298 model geometry")
    if not isinstance(root_seed, str) or not root_seed:
        raise ValueError("root_seed must be non-empty")

    model_seed = derive_stream_seed(root_seed, "EXP-298", canonical_index, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        control, fidelity = build_matched_cross_domain_fidelity_arms(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
        )
    pair_audit = audit_matched_cross_domain_fidelity_arms(control, fidelity)
    state_before = canonical_sha256(
        {"control": _state_digest(control), "fidelity": _state_digest(fidelity)}
    )
    court = BehavioralFidelityCourt(max_exact_probes=max_exact_probes)
    raw: list[dict[str, Any]] = []

    control.eval()
    fidelity.eval()
    with torch.no_grad():
        for domain_id in DOMAIN_ORDER:
            generator, semantics_factory = _domain_factory(domain_id)
            for offset in range(eval_replicates):
                replicate = eval_start_replicate + offset
                seed_replicate = (
                    canonical_index * 1_000_000
                    + DOMAIN_OFFSETS[domain_id]
                    + replicate
                )
                environment_seed = derive_stream_seed(
                    root_seed,
                    "EXP-298",
                    seed_replicate,
                    "environment",
                )
                batch = generator(environment_seed)
                semantics = semantics_factory(batch.source)
                for candidate_index, case in enumerate(batch.candidates):
                    receipt = court.adjudicate(batch.source, case.candidate, semantics)
                    control_decision = control.decide(
                        source_digest=batch.source_digest,
                        candidate_digest=case.candidate_digest,
                        executable=True,
                    )
                    fidelity_decision = fidelity.decide(
                        source_digest=batch.source_digest,
                        candidate_digest=case.candidate_digest,
                        executable=True,
                        court_receipt=receipt,
                    )
                    raw.append(
                        {
                            "domain": domain_id,
                            "replicate": replicate,
                            "environment_seed": environment_seed,
                            "candidate_index": candidate_index,
                            "candidate_id": case.candidate_id,
                            "stratum": case.stratum,
                            "is_faithful": bool(case.is_faithful),
                            "source_digest": batch.source_digest,
                            "candidate_digest": case.candidate_digest,
                            "executable": True,
                            "court_receipt": receipt.as_dict(),
                            "arm_input_receipt": {
                                "candidate_set_frozen_before_arms": True,
                                "byte_identical_candidate_order": True,
                                "source_digest_match": True,
                                "candidate_digest_match": True,
                                "evaluator_truth_in_causal_path": False,
                                "trap_family_in_causal_path": False,
                            },
                            "arms": {
                                "compile_only_control": _arm_payload(
                                    control_decision,
                                    probes=0,
                                    domain_ops=0,
                                ),
                                "cross_domain_fidelity_fabric": _arm_payload(
                                    fidelity_decision,
                                    probes=receipt.probes_evaluated,
                                    domain_ops=receipt.semantic_domain_operations,
                                ),
                            },
                        }
                    )

    state_after = canonical_sha256(
        {"control": _state_digest(control), "fidelity": _state_digest(fidelity)}
    )
    domains: dict[str, Any] = {}
    domain_passes: list[bool] = []
    for domain_id in DOMAIN_ORDER:
        rows = [row for row in raw if row["domain"] == domain_id]
        control_metrics = _arm_metrics(rows, "compile_only_control")
        fidelity_metrics = _arm_metrics(rows, "cross_domain_fidelity_fabric")
        gain = (
            fidelity_metrics["semantic_fidelity_balanced_accuracy"]
            - control_metrics["semantic_fidelity_balanced_accuracy"]
        )
        passed = bool(
            gain >= 0.10
            and fidelity_metrics["wrong_formalization_authority_rate"] <= 0.05
            and fidelity_metrics["faithful_formalization_rejection_rate"] <= 0.10
        )
        domains[domain_id] = {
            "candidate_count": len(rows),
            "compile_only_control": control_metrics,
            "cross_domain_fidelity_fabric": fidelity_metrics,
            "balanced_accuracy_gain": gain,
            "mesi_absolute_gain": 0.10,
            "wrong_authority_ceiling": 0.05,
            "faithful_rejection_ceiling": 0.10,
            "pass": passed,
        }
        domain_passes.append(passed)

    worst_gain = min(
        float(domains[domain_id]["balanced_accuracy_gain"])
        for domain_id in DOMAIN_ORDER
    )
    pair_ok = all(
        bool(pair_audit.get(key))
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "initialization_match",
            "neural_accounted_flops_match",
            "receipt_access_is_only_causal_intervention",
        )
    )
    established = bool(
        all(domain_passes)
        and pair_ok
        and state_before == state_after
        and len(raw) == len(DOMAIN_ORDER) * eval_replicates * 16
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": "EXP-298",
        "evidence_level": "EV-E2",
        "decision": (
            "CROSS_DOMAIN_FIDELITY_TRANSFER_ESTABLISHED"
            if established
            else "CROSS_DOMAIN_FIDELITY_TRANSFER_NOT_ESTABLISHED"
        ),
        "canonical_index": canonical_index,
        "root_seed": root_seed,
        "model_init_seed": model_seed,
        "repository_head": repository_head,
        "protocol_digest": protocol_digest,
        "geometry_digest": geometry_digest,
        "code_digest": code_digest,
        "config": {
            "eval_replicates": eval_replicates,
            "eval_start_replicate": eval_start_replicate,
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
            "max_exact_probes": max_exact_probes,
            "domains": list(DOMAIN_ORDER),
            "candidates_per_replicate": 16,
        },
        "resource_match": {"pair_audit": pair_audit},
        "model_state_digest_before_evaluation": state_before,
        "model_state_digest_after_evaluation": state_after,
        "model_state_unchanged_during_evaluation": state_before == state_after,
        "evaluation_candidate_count": len(raw),
        "heldout_domain_count": len(DOMAIN_ORDER),
        "evaluation": {
            "raw_candidates": raw,
            "domains": domains,
            "worst_domain_fidelity_gain": worst_gain,
        },
        "successor_design_authorized": False,
        "authorization_scope": "NONE",
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "unrestricted_semantic_authority_claimed": False,
        "open_language_understanding_claimed": False,
        "causal_discovery_claimed": False,
        "general_code_reasoning_claimed": False,
        "exp290_authority_inherited": False,
        "exp291_296_authority_inherited": False,
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def validate_exp298_root(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("EXP-298 root schema mismatch")
    if payload.get("experiment_id") != "EXP-298" or payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-298 root epistemic identity mismatch")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-298 root artifact digest mismatch")
    for flag in FALSE_FLAGS:
        if payload.get(flag) is not False:
            errors.append(f"EXP-298 forbidden flag enabled: {flag}")
    if payload.get("successor_design_authorized") is not False or payload.get("authorization_scope") != "NONE":
        errors.append("EXP-298 root successor authority must remain closed")

    for name, length in (
        ("repository_head", 40),
        ("protocol_digest", 64),
        ("geometry_digest", 64),
        ("code_digest", 64),
    ):
        error = _validate_identity_hex(name, payload.get(name), length)
        if error:
            errors.append(error)

    config = payload.get("config") or {}
    try:
        kwargs = {
            "root_seed": payload["root_seed"],
            "canonical_index": int(payload["canonical_index"]),
            "eval_replicates": int(config["eval_replicates"]),
            "eval_start_replicate": int(config["eval_start_replicate"]),
            "d_model": int(config["d_model"]),
            "hidden_size": int(config["hidden_size"]),
            "target_parameters": int(config["target_parameters"]),
            "max_exact_probes": int(config["max_exact_probes"]),
            "protocol_digest": payload["protocol_digest"],
            "geometry_digest": payload["geometry_digest"],
            "code_digest": payload["code_digest"],
            "repository_head": payload["repository_head"],
        }
    except (KeyError, TypeError, ValueError):
        errors.append("EXP-298 root reconstruction config invalid")
        return errors

    if config.get("domains") != list(DOMAIN_ORDER) or config.get("candidates_per_replicate") != 16:
        errors.append("EXP-298 root frozen domain/cardinality mismatch")
    try:
        expected = run_exp298_root(**kwargs)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        errors.append(f"EXP-298 root reconstruction failed: {type(exc).__name__}")
        return errors
    if payload != expected:
        errors.append("EXP-298 root reconstruction mismatch")
    return errors
