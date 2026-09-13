from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch
from torch.nn import functional as F

from nolane_ai.experiments.exp298_behavioral_fidelity import (
    BehavioralFidelityCourt,
    BehavioralFidelityReceipt,
)
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
from nolane_ai.experiments.exp299_structural_encoding import (
    encode_exp299_pair,
    structural_encoder_contract,
)
from nolane_ai.experiments.matched_native_fidelity_arms import (
    NativeFidelityDecision,
    audit_matched_native_fidelity_arms,
    build_matched_native_fidelity_arms,
    native_arm_state_digest,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import (
    build_functional_optimizer,
    functional_trainable_named_parameters,
)

SCHEMA = "NLM-EXP-299-NATIVE-FIDELITY-ROOT-V1"
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
    "durable_lifelong_internalization_claimed",
    "exp300_execution_authorized",
)


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _domain_factory(domain_id: str) -> tuple[Callable[[int], Any], Callable[[Any], Any]]:
    if domain_id == "code_invariant":
        return generate_exp298_code_world, lambda source: CodeInvariantSemantics(source)
    if domain_id == "causal_diagnosis":
        return generate_exp298_causal_world, lambda source: CausalDiagnosisSemantics(source)
    if domain_id == "grounded_language_ambiguity":
        return generate_exp298_language_world, lambda source: GroundedLanguageSemantics(source)
    raise ValueError(f"unknown EXP-299 domain: {domain_id}")


def _phase_environment_seed(
    root_seed: str,
    *,
    phase: str,
    canonical_index: int,
    domain_id: str,
    replicate: int,
) -> int:
    if phase not in {"fit", "eval"}:
        raise ValueError("EXP-299 phase must be fit or eval")
    composite_replicate = (
        canonical_index * 1_000_000 + DOMAIN_OFFSETS[domain_id] + int(replicate)
    )
    return derive_stream_seed(
        f"{root_seed}|{phase}",
        "EXP-299",
        composite_replicate,
        "environment",
    )


def _candidate_instance_id(
    *,
    phase: str,
    domain_id: str,
    replicate: int,
    environment_seed: int,
    candidate_id: str,
    candidate_digest: str,
) -> str:
    return canonical_sha256(
        {
            "phase": phase,
            "domain": domain_id,
            "replicate": int(replicate),
            "environment_seed": int(environment_seed),
            "candidate_id": candidate_id,
            "candidate_digest": candidate_digest,
        }
    )


def _teacher_target(
    receipt: BehavioralFidelityReceipt,
    *,
    total_probes: int,
) -> tuple[float, ...]:
    witness = receipt.witness or {}
    direction = witness.get("direction")
    denominator = max(int(total_probes), 1)
    return (
        float(receipt.decision == "court_accept"),
        float(receipt.decision == "court_reject"),
        float(receipt.decision == "court_inconclusive"),
        float(direction == "source_to_candidate"),
        float(direction == "candidate_to_source"),
        float(receipt.complete_probe_space),
        min(float(receipt.probes_evaluated) / float(denominator), 1.0),
    )


def _aggregate_state_digest(control: torch.nn.Module, teacher: torch.nn.Module) -> str:
    return canonical_sha256(
        {
            "binary_supervision_control": native_arm_state_digest(control),
            "court_teacher_then_native": native_arm_state_digest(teacher),
        }
    )


def _arm_payload(decision: NativeFidelityDecision) -> dict[str, Any]:
    return {
        "authority_granted": bool(decision.authority_granted),
        "authority_score": float(decision.authority_score),
        "fidelity_score": float(decision.fidelity_score),
        "teacher_prediction": [float(value) for value in decision.teacher_prediction],
        "neural_accounted_flops": int(decision.neural_accounted_flops),
        "native_inference": bool(decision.native_inference),
        "teacher_scaffold_consumed": bool(decision.teacher_scaffold_consumed),
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
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "faithful_count": len(faithful),
        "wrong_count": len(wrong),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "wrong_authority_rate": fp / len(wrong) if wrong else 0.0,
        "faithful_rejection_rate": fn / len(faithful) if faithful else 0.0,
    }


def _summarize_domains(
    raw: list[dict[str, Any]],
    *,
    native_ba_floor: float,
    gain_mesi: float,
    wrong_authority_ceiling: float,
    faithful_rejection_ceiling: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain_id in DOMAIN_ORDER:
        rows = [row for row in raw if row["domain"] == domain_id]
        control = _arm_metrics(rows, "binary_supervision_control")
        teacher = _arm_metrics(rows, "court_teacher_then_native")
        gain = teacher["balanced_accuracy"] - control["balanced_accuracy"]
        passed = bool(
            teacher["balanced_accuracy"] >= native_ba_floor
            and gain >= gain_mesi
            and teacher["wrong_authority_rate"] <= wrong_authority_ceiling
            and teacher["faithful_rejection_rate"] <= faithful_rejection_ceiling
        )
        result[domain_id] = {
            "candidate_count": len(rows),
            "binary_supervision_control": control,
            "court_teacher_then_native": teacher,
            "native_balanced_accuracy_gain": gain,
            "native_ba_floor": native_ba_floor,
            "gain_mesi": gain_mesi,
            "wrong_authority_ceiling": wrong_authority_ceiling,
            "faithful_rejection_ceiling": faithful_rejection_ceiling,
            "pass": passed,
        }
    return result


def _validate_identity_hex(name: str, value: Any, length: int) -> str | None:
    if not isinstance(value, str) or len(value) != length:
        return f"EXP-299 {name} identity invalid"
    try:
        bytes.fromhex(value)
    except ValueError:
        return f"EXP-299 {name} identity invalid"
    return None


def run_exp299_root(
    *,
    root_seed: str,
    canonical_index: int,
    fit_replicates: int,
    fit_start_replicate: int,
    eval_replicates: int,
    eval_start_replicate: int,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    authority_threshold: float,
    max_exact_probes: int,
    protocol_digest: str,
    geometry_digest: str,
    code_digest: str,
    repository_head: str,
) -> dict[str, Any]:
    if canonical_index not in (0, 1, 2, 3):
        raise ValueError("canonical_index must be one of 0,1,2,3")
    if not isinstance(root_seed, str) or not root_seed:
        raise ValueError("root_seed must be non-empty")
    if fit_replicates <= 0 or eval_replicates <= 0:
        raise ValueError("EXP-299 fit/eval replicates must be positive")
    if fit_start_replicate < 0 or eval_start_replicate < 0:
        raise ValueError("EXP-299 replicate starts must be non-negative")
    if min(d_model, hidden_size, target_parameters, batch_size, max_exact_probes) <= 0:
        raise ValueError("invalid EXP-299 geometry")
    if learning_rate <= 0 or weight_decay < 0 or gradient_clip_norm <= 0:
        raise ValueError("invalid EXP-299 optimizer geometry")
    if not 0.0 <= authority_threshold <= 1.0:
        raise ValueError("authority_threshold must be in [0,1]")

    native_ba_floor = 0.70
    gain_mesi = 0.10
    wrong_authority_ceiling = 0.05
    faithful_rejection_ceiling = 0.10

    model_seed = derive_stream_seed(root_seed, "EXP-299", canonical_index, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        control, teacher = build_matched_native_fidelity_arms(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
        )
    pair_audit = audit_matched_native_fidelity_arms(control, teacher)
    model_state_digest_at_initialization = _aggregate_state_digest(control, teacher)

    court = BehavioralFidelityCourt(max_exact_probes=max_exact_probes)
    encoder_contract = structural_encoder_contract()
    encoder_digest = canonical_sha256(encoder_contract)

    fit_examples: list[dict[str, Any]] = []
    fit_instance_ids: list[str] = []
    fit_semantic_probe_evaluations = 0
    fit_semantic_domain_operations = 0
    fit_inconclusive_count = 0

    for domain_id in DOMAIN_ORDER:
        generator, semantics_factory = _domain_factory(domain_id)
        for offset in range(fit_replicates):
            replicate = fit_start_replicate + offset
            environment_seed = _phase_environment_seed(
                root_seed,
                phase="fit",
                canonical_index=canonical_index,
                domain_id=domain_id,
                replicate=replicate,
            )
            batch = generator(environment_seed)
            semantics = semantics_factory(batch.source)
            total_probes = len(semantics.probes())
            for candidate_index, case in enumerate(batch.candidates):
                receipt = court.adjudicate(batch.source, case.candidate, semantics)
                fit_semantic_probe_evaluations += int(receipt.probes_evaluated)
                fit_semantic_domain_operations += int(receipt.semantic_domain_operations)
                instance_id = _candidate_instance_id(
                    phase="fit",
                    domain_id=domain_id,
                    replicate=replicate,
                    environment_seed=environment_seed,
                    candidate_id=case.candidate_id,
                    candidate_digest=case.candidate_digest,
                )
                fit_instance_ids.append(instance_id)
                if receipt.decision == "court_inconclusive":
                    fit_inconclusive_count += 1
                    continue
                pair_features = encode_exp299_pair(
                    domain_id,
                    batch.source,
                    case.candidate,
                )
                fit_examples.append(
                    {
                        "instance_id": instance_id,
                        "domain": domain_id,
                        "replicate": replicate,
                        "candidate_index": candidate_index,
                        "pair_features": pair_features,
                        "pair_digest": canonical_sha256(list(pair_features)),
                        "primary_target": float(receipt.decision == "court_accept"),
                        "teacher_target": _teacher_target(
                            receipt,
                            total_probes=total_probes,
                        ),
                        "court_receipt_digest": receipt.receipt_digest,
                    }
                )

    if not fit_examples:
        raise RuntimeError("EXP-299 fit partition produced no trainable examples")

    control_optimizer = build_functional_optimizer(
        control,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    teacher_optimizer = build_functional_optimizer(
        teacher,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    control_parameters = [
        parameter for _, parameter in functional_trainable_named_parameters(control)
    ]
    teacher_parameters = [
        parameter for _, parameter in functional_trainable_named_parameters(teacher)
    ]

    control.train()
    teacher.train()
    fit_steps = 0
    final_control_loss = 0.0
    final_teacher_loss = 0.0
    for start in range(0, len(fit_examples), batch_size):
        examples = fit_examples[start : start + batch_size]
        pair_tensor = torch.tensor(
            [example["pair_features"] for example in examples],
            dtype=torch.float32,
        )
        primary_target = torch.tensor(
            [example["primary_target"] for example in examples],
            dtype=torch.float32,
        )
        teacher_target = torch.tensor(
            [example["teacher_target"] for example in examples],
            dtype=torch.float32,
        )

        control_optimizer.zero_grad(set_to_none=True)
        control_output = control.forward_pair(pair_tensor)
        control_primary = 0.5 * (
            F.binary_cross_entropy_with_logits(
                control_output.authority_logit,
                primary_target,
            )
            + F.binary_cross_entropy_with_logits(
                control_output.fidelity_logit,
                primary_target,
            )
        )
        control_auxiliary = F.mse_loss(
            control_output.teacher_prediction,
            teacher_target,
        )
        control_loss = (
            control_primary
            + control.auxiliary_loss_weight * control_auxiliary
        )
        control_loss.backward()
        torch.nn.utils.clip_grad_norm_(control_parameters, gradient_clip_norm)
        control_optimizer.step()

        teacher_optimizer.zero_grad(set_to_none=True)
        teacher_output = teacher.forward_pair(pair_tensor)
        teacher_primary = 0.5 * (
            F.binary_cross_entropy_with_logits(
                teacher_output.authority_logit,
                primary_target,
            )
            + F.binary_cross_entropy_with_logits(
                teacher_output.fidelity_logit,
                primary_target,
            )
        )
        teacher_auxiliary = F.mse_loss(
            teacher_output.teacher_prediction,
            teacher_target,
        )
        teacher_loss = (
            teacher_primary
            + teacher.auxiliary_loss_weight * teacher_auxiliary
        )
        teacher_loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher_parameters, gradient_clip_norm)
        teacher_optimizer.step()

        fit_steps += 1
        final_control_loss = float(control_loss.detach().cpu().item())
        final_teacher_loss = float(teacher_loss.detach().cpu().item())

    model_state_digest_after_fit = _aggregate_state_digest(control, teacher)
    model_state_digest_before_evaluation = model_state_digest_after_fit

    raw: list[dict[str, Any]] = []
    eval_instance_ids: list[str] = []
    eval_inconclusive_count = 0
    eval_semantic_probe_evaluations = 0
    eval_semantic_domain_operations = 0

    control.eval()
    teacher.eval()
    for domain_id in DOMAIN_ORDER:
        generator, semantics_factory = _domain_factory(domain_id)
        for offset in range(eval_replicates):
            replicate = eval_start_replicate + offset
            environment_seed = _phase_environment_seed(
                root_seed,
                phase="eval",
                canonical_index=canonical_index,
                domain_id=domain_id,
                replicate=replicate,
            )
            batch = generator(environment_seed)
            semantics = semantics_factory(batch.source)
            for candidate_index, case in enumerate(batch.candidates):
                pair_features = encode_exp299_pair(
                    domain_id,
                    batch.source,
                    case.candidate,
                )
                pair_digest = canonical_sha256(list(pair_features))

                # Critical ordering: both native predictions are committed before
                # exact held-out truth is materialized by the evaluator court.
                control_decision = control.decide(
                    pair_features,
                    executable=True,
                    threshold=authority_threshold,
                )
                teacher_decision = teacher.decide(
                    pair_features,
                    executable=True,
                    threshold=authority_threshold,
                )
                prediction_commitment = {
                    "binary_supervision_control": _arm_payload(control_decision),
                    "court_teacher_then_native": _arm_payload(teacher_decision),
                }
                prediction_commitment_digest = canonical_sha256(prediction_commitment)

                receipt = court.adjudicate(batch.source, case.candidate, semantics)
                eval_semantic_probe_evaluations += int(receipt.probes_evaluated)
                eval_semantic_domain_operations += int(receipt.semantic_domain_operations)
                if receipt.decision == "court_inconclusive":
                    eval_inconclusive_count += 1

                instance_id = _candidate_instance_id(
                    phase="eval",
                    domain_id=domain_id,
                    replicate=replicate,
                    environment_seed=environment_seed,
                    candidate_id=case.candidate_id,
                    candidate_digest=case.candidate_digest,
                )
                eval_instance_ids.append(instance_id)
                raw.append(
                    {
                        "domain": domain_id,
                        "replicate": replicate,
                        "environment_seed": environment_seed,
                        "candidate_index": candidate_index,
                        "candidate_instance_id": instance_id,
                        "candidate_id": case.candidate_id,
                        "source_digest": batch.source_digest,
                        "candidate_digest": case.candidate_digest,
                        "stratum": case.stratum,
                        "is_faithful": bool(receipt.decision == "court_accept"),
                        "model_causal_input": {
                            "structural_pair_digest": pair_digest,
                            "executable": True,
                        },
                        "prediction_commitment_digest": prediction_commitment_digest,
                        "prediction_committed_before_court": True,
                        "court_receipt": receipt.as_dict(),
                        "arms": prediction_commitment,
                    }
                )

    model_state_digest_after_evaluation = _aggregate_state_digest(control, teacher)
    model_state_unchanged = (
        model_state_digest_before_evaluation == model_state_digest_after_evaluation
    )

    fit_partition_digest = canonical_sha256(fit_instance_ids)
    eval_partition_digest = canonical_sha256(eval_instance_ids)
    fit_eval_disjoint = set(fit_instance_ids).isdisjoint(set(eval_instance_ids))
    raw_digest = canonical_sha256(raw)

    domains = _summarize_domains(
        raw,
        native_ba_floor=native_ba_floor,
        gain_mesi=gain_mesi,
        wrong_authority_ceiling=wrong_authority_ceiling,
        faithful_rejection_ceiling=faithful_rejection_ceiling,
    )
    worst_gain = min(
        float(domains[domain_id]["native_balanced_accuracy_gain"])
        for domain_id in DOMAIN_ORDER
    )
    pair_ok = all(
        bool(pair_audit.get(key))
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "optimizer_visible_parameter_match",
            "initialization_match",
            "neural_accounted_flops_match",
            "forward_path_match",
        )
    )
    established = bool(
        all(bool(domains[domain_id]["pass"]) for domain_id in DOMAIN_ORDER)
        and pair_ok
        and fit_eval_disjoint
        and fit_inconclusive_count == 0
        and eval_inconclusive_count == 0
        and model_state_unchanged
        and len(raw) == len(DOMAIN_ORDER) * eval_replicates * 16
        and encoder_contract.get("evaluator_information_consumed") is False
    )

    forward_flops = int(control._neural_accounted_flops())
    fit_trainable_count = len(fit_examples)
    evaluation_count = len(raw)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": "EXP-299",
        "evidence_level": "EV-E2",
        "decision": (
            "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL"
            if established
            else "NATIVE_FIDELITY_NOT_ESTABLISHED"
        ),
        "canonical_index": canonical_index,
        "root_seed": root_seed,
        "model_init_seed": model_seed,
        "repository_head": repository_head,
        "protocol_digest": protocol_digest,
        "geometry_digest": geometry_digest,
        "code_digest": code_digest,
        "config": {
            "fit_replicates": fit_replicates,
            "fit_start_replicate": fit_start_replicate,
            "eval_replicates": eval_replicates,
            "eval_start_replicate": eval_start_replicate,
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "authority_threshold": authority_threshold,
            "max_exact_probes": max_exact_probes,
            "domains": list(DOMAIN_ORDER),
            "candidates_per_replicate": 16,
            "native_ba_floor": native_ba_floor,
            "gain_mesi": gain_mesi,
            "wrong_authority_ceiling": wrong_authority_ceiling,
            "faithful_rejection_ceiling": faithful_rejection_ceiling,
            "fit_passes": 1,
        },
        "structural_encoder": {
            "contract": encoder_contract,
            "contract_digest": encoder_digest,
        },
        "resource_match": {"pair_audit": pair_audit},
        "model_state_digest_at_initialization": model_state_digest_at_initialization,
        "model_state_digest_after_fit": model_state_digest_after_fit,
        "model_state_digest_before_evaluation": model_state_digest_before_evaluation,
        "model_state_digest_after_evaluation": model_state_digest_after_evaluation,
        "model_state_unchanged_during_evaluation": model_state_unchanged,
        "partition_audit": {
            "fit_eval_disjoint": fit_eval_disjoint,
            "identity_semantics": "phase/domain/replicate/environment/candidate-id/content-digest",
        },
        "fit": {
            "candidate_count": len(fit_instance_ids),
            "trainable_candidate_count": fit_trainable_count,
            "candidate_instance_ids": fit_instance_ids,
            "partition_digest": fit_partition_digest,
            "inconclusive_count": fit_inconclusive_count,
            "fit_steps": fit_steps,
            "primary_supervision_matched": True,
            "batch_order_matched": True,
            "optimizer_geometry_matched": True,
            "auxiliary_loss_weights": {
                "binary_supervision_control": control.auxiliary_loss_weight,
                "court_teacher_then_native": teacher.auxiliary_loss_weight,
            },
            "final_losses": {
                "binary_supervision_control": final_control_loss,
                "court_teacher_then_native": final_teacher_loss,
            },
            "semantic_probe_evaluations": fit_semantic_probe_evaluations,
            "semantic_domain_operations": fit_semantic_domain_operations,
        },
        "evaluation": {
            "candidate_count": evaluation_count,
            "candidate_instance_ids": eval_instance_ids,
            "partition_digest": eval_partition_digest,
            "inconclusive_count": eval_inconclusive_count,
            "raw_candidates": raw,
            "raw_digest": raw_digest,
            "domains": domains,
            "worst_domain_native_gain": worst_gain,
            "semantic_probe_evaluations": eval_semantic_probe_evaluations,
            "semantic_domain_operations": eval_semantic_domain_operations,
        },
        "cost_ledger": {
            "forward_neural_flops_per_candidate": forward_flops,
            "training_neural_flops_proxy_per_arm": int(
                3 * forward_flops * fit_trainable_count
            ),
            "evaluation_neural_flops_proxy_per_arm": int(
                forward_flops * evaluation_count
            ),
            "fit_semantic_probe_evaluations": fit_semantic_probe_evaluations,
            "fit_semantic_domain_operations": fit_semantic_domain_operations,
            "heldout_evaluator_semantic_probe_evaluations": eval_semantic_probe_evaluations,
            "heldout_evaluator_semantic_domain_operations": eval_semantic_domain_operations,
            "hardware_profiler_flops_claimed": False,
        },
        "heldout_teacher_scaffold_consumed": False,
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
        "durable_lifelong_internalization_claimed": False,
        "exp300_execution_authorized": False,
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def validate_exp299_root(
    payload: dict[str, Any],
    *,
    reconstruct: bool = False,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("EXP-299 root schema mismatch")
    if payload.get("experiment_id") != "EXP-299" or payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-299 root epistemic identity mismatch")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-299 root artifact digest mismatch")
    for flag in FALSE_FLAGS:
        if payload.get(flag) is not False:
            errors.append(f"EXP-299 forbidden flag enabled: {flag}")
    if payload.get("successor_design_authorized") is not False:
        errors.append("EXP-299 root successor authority must remain closed")
    if payload.get("authorization_scope") != "NONE":
        errors.append("EXP-299 root authorization scope must remain NONE")
    if payload.get("heldout_teacher_scaffold_consumed") is not False:
        errors.append("EXP-299 held-out teacher scaffold leakage detected")

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
    if config.get("domains") != list(DOMAIN_ORDER):
        errors.append("EXP-299 frozen domain order mismatch")
    if config.get("candidates_per_replicate") != 16:
        errors.append("EXP-299 candidate cardinality mismatch")
    if config.get("fit_passes") != 1:
        errors.append("EXP-299 fit pass count mismatch")

    encoder = payload.get("structural_encoder") or {}
    contract = encoder.get("contract") or {}
    if encoder.get("contract_digest") != canonical_sha256(contract):
        errors.append("EXP-299 structural encoder contract digest mismatch")
    if contract.get("evaluator_information_consumed") is not False:
        errors.append("EXP-299 structural encoder evaluator leakage")

    fit = payload.get("fit") or {}
    evaluation = payload.get("evaluation") or {}
    fit_ids = list(fit.get("candidate_instance_ids") or [])
    eval_ids = list(evaluation.get("candidate_instance_ids") or [])
    disjoint = set(fit_ids).isdisjoint(set(eval_ids))
    if not disjoint or payload.get("partition_audit", {}).get("fit_eval_disjoint") is not True:
        errors.append("EXP-299 fit/eval partition overlap")
    if fit.get("partition_digest") != canonical_sha256(fit_ids):
        errors.append("EXP-299 fit partition digest mismatch")
    if evaluation.get("partition_digest") != canonical_sha256(eval_ids):
        errors.append("EXP-299 evaluation partition digest mismatch")

    raw = list(evaluation.get("raw_candidates") or [])
    if evaluation.get("raw_digest") != canonical_sha256(raw):
        errors.append("EXP-299 evaluation raw digest mismatch")
    if evaluation.get("candidate_count") != len(raw) or len(eval_ids) != len(raw):
        errors.append("EXP-299 evaluation candidate count mismatch")
    if fit.get("candidate_count") != len(fit_ids):
        errors.append("EXP-299 fit candidate count mismatch")

    for row in raw:
        if row.get("prediction_committed_before_court") is not True:
            errors.append("EXP-299 held-out prediction was not committed before court")
            break
        causal_input = row.get("model_causal_input") or {}
        if set(causal_input) != {"structural_pair_digest", "executable"}:
            errors.append("EXP-299 held-out causal input boundary mismatch")
            break
        arms = row.get("arms") or {}
        for arm_id in ("binary_supervision_control", "court_teacher_then_native"):
            arm = arms.get(arm_id) or {}
            if arm.get("native_inference") is not True or arm.get("teacher_scaffold_consumed") is not False:
                errors.append("EXP-299 held-out arm consumed scaffold or was non-native")
                break
        else:
            commitment = {
                "binary_supervision_control": arms.get("binary_supervision_control"),
                "court_teacher_then_native": arms.get("court_teacher_then_native"),
            }
            if row.get("prediction_commitment_digest") != canonical_sha256(commitment):
                errors.append("EXP-299 prediction commitment digest mismatch")
                break
            continue
        break

    try:
        expected_domains = _summarize_domains(
            raw,
            native_ba_floor=float(config["native_ba_floor"]),
            gain_mesi=float(config["gain_mesi"]),
            wrong_authority_ceiling=float(config["wrong_authority_ceiling"]),
            faithful_rejection_ceiling=float(config["faithful_rejection_ceiling"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        expected_domains = None
        errors.append("EXP-299 aggregate reconstruction config invalid")
    if expected_domains is not None and evaluation.get("domains") != expected_domains:
        errors.append("EXP-299 aggregate reconstruction mismatch")

    state_unchanged = payload.get("model_state_digest_before_evaluation") == payload.get(
        "model_state_digest_after_evaluation"
    )
    if payload.get("model_state_unchanged_during_evaluation") is not state_unchanged:
        errors.append("EXP-299 model-state immutability declaration mismatch")
    if not state_unchanged:
        errors.append("EXP-299 model mutated during held-out evaluation")

    pair_audit = ((payload.get("resource_match") or {}).get("pair_audit") or {})
    pair_ok = all(
        bool(pair_audit.get(key))
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "optimizer_visible_parameter_match",
            "initialization_match",
            "neural_accounted_flops_match",
            "forward_path_match",
        )
    )
    if not pair_ok:
        errors.append("EXP-299 matched-arm audit failed")

    expected_established = bool(
        expected_domains is not None
        and all(bool(expected_domains[domain_id]["pass"]) for domain_id in DOMAIN_ORDER)
        and pair_ok
        and disjoint
        and fit.get("inconclusive_count") == 0
        and evaluation.get("inconclusive_count") == 0
        and state_unchanged
        and len(raw) == len(DOMAIN_ORDER) * int(config.get("eval_replicates", -1)) * 16
        and contract.get("evaluator_information_consumed") is False
    )
    expected_decision = (
        "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL"
        if expected_established
        else "NATIVE_FIDELITY_NOT_ESTABLISHED"
    )
    if payload.get("decision") != expected_decision:
        errors.append("EXP-299 root decision mismatch")

    if reconstruct and not errors:
        try:
            expected = run_exp299_root(
                root_seed=str(payload["root_seed"]),
                canonical_index=int(payload["canonical_index"]),
                fit_replicates=int(config["fit_replicates"]),
                fit_start_replicate=int(config["fit_start_replicate"]),
                eval_replicates=int(config["eval_replicates"]),
                eval_start_replicate=int(config["eval_start_replicate"]),
                d_model=int(config["d_model"]),
                hidden_size=int(config["hidden_size"]),
                target_parameters=int(config["target_parameters"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                authority_threshold=float(config["authority_threshold"]),
                max_exact_probes=int(config["max_exact_probes"]),
                protocol_digest=str(payload["protocol_digest"]),
                geometry_digest=str(payload["geometry_digest"]),
                code_digest=str(payload["code_digest"]),
                repository_head=str(payload["repository_head"]),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(
                f"EXP-299 root reconstruction failed: {type(exc).__name__}"
            )
        else:
            if expected.get("artifact_digest") != payload.get("artifact_digest"):
                errors.append("EXP-299 root reconstruction mismatch")
    return errors
