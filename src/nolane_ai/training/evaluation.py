from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.protocol.evidence import canonical_sha256
from .curriculum import StageACurriculum
from .pilot import StageAPilotTrainer, functional_model_state_digest


@dataclass(frozen=True, slots=True)
class StageABatchEvaluation:
    replicate: int
    batch_digest: str
    examples: int
    belief_accuracy: float
    belief_brier: float
    conflict_balanced_accuracy: float
    conflict_brier: float
    fidelity_balanced_accuracy: float
    fidelity_brier: float


@dataclass(frozen=True, slots=True)
class StageAEvaluationResult:
    rng_stream: str
    start_replicate: int
    batches: int
    examples: int
    batch_digests: tuple[str, ...]
    per_batch: tuple[StageABatchEvaluation, ...]
    belief_accuracy: float
    belief_brier: float
    conflict_balanced_accuracy: float
    conflict_brier: float
    fidelity_balanced_accuracy: float
    fidelity_brier: float


@dataclass(frozen=True, slots=True)
class HeldoutMetricDelta:
    belief_accuracy: float
    belief_brier_reduction: float
    conflict_balanced_accuracy: float
    conflict_brier_reduction: float
    fidelity_balanced_accuracy: float
    fidelity_brier_reduction: float


@dataclass(frozen=True, slots=True)
class PairedHeldoutBatchDelta:
    replicate: int
    batch_digest: str
    belief_accuracy: float
    belief_brier_reduction: float
    conflict_balanced_accuracy: float
    conflict_brier_reduction: float
    fidelity_balanced_accuracy: float
    fidelity_brier_reduction: float


@dataclass(frozen=True, slots=True)
class StageAHeldoutDevelopmentReport:
    schema: str
    evidence_level: str
    decision: str
    scope: str
    training_stream: str
    evaluation_stream: str
    protocol_digest: str
    code_digest: str
    config_digest: str
    model_init_seed: int
    initial_model_digest: str
    model_before_digest: str
    model_after_digest: str
    training_steps: int
    train_start_replicate: int
    pre: StageAEvaluationResult
    post: StageAEvaluationResult
    delta: HeldoutMetricDelta
    paired_batch_deltas: tuple[PairedHeldoutBatchDelta, ...]
    training_curriculum_digests: tuple[str, ...]
    report_digest: str


def _balanced_accuracy(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    target = targets.to(torch.long).reshape(-1)
    pred = predictions.to(torch.long).reshape(-1)
    recalls: list[torch.Tensor] = []
    for label in (0, 1):
        mask = target == label
        if mask.any():
            recalls.append((pred[mask] == label).to(torch.float32).mean())
    if not recalls:
        return 0.0
    return float(torch.stack(recalls).mean().item())


def evaluate_stage_a(
    model: NolaneLivingModel,
    *,
    curriculum: StageACurriculum,
    start_replicate: int,
    batches: int,
    batch_size: int,
    variables: int,
    constraints: int,
) -> StageAEvaluationResult:
    if start_replicate < 0:
        raise ValueError("start_replicate must be non-negative")
    if min(batches, batch_size, variables, constraints) <= 0:
        raise ValueError("batches, batch_size, variables and constraints must be positive")

    device = next(model.parameters()).device
    belief_probs: list[torch.Tensor] = []
    belief_targets: list[torch.Tensor] = []
    conflict_probs: list[torch.Tensor] = []
    conflict_targets: list[torch.Tensor] = []
    fidelity_probs: list[torch.Tensor] = []
    fidelity_targets: list[torch.Tensor] = []
    digests: list[str] = []
    per_batch: list[StageABatchEvaluation] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for offset in range(batches):
                batch = curriculum.make_batch(
                    replicate=start_replicate + offset,
                    batch_size=batch_size,
                    variables=variables,
                    constraints=constraints,
                    d_model=model.config.d_model,
                    device=device,
                    rng_stream="evaluation",
                )
                structured = model.structured_reason(
                    batch.batch.variable_states,
                    batch.batch.incidence,
                )
                belief_probs.append(
                    torch.softmax(structured.belief_logits, dim=-1)[..., 1].detach().cpu()
                )
                belief_targets.append(batch.batch.belief_targets.detach().cpu().to(torch.float32))
                conflict_probs.append(torch.sigmoid(structured.conflict_scores).detach().cpu())
                conflict_targets.append(batch.batch.conflict_targets.detach().cpu().to(torch.float32))
                fidelity_probs.append(
                    model.semantic_fidelity_score(
                        batch.batch.source_semantics,
                        batch.batch.candidate_semantics,
                    ).detach().cpu()
                )
                fidelity_target = batch.batch.fidelity_targets.detach().cpu().to(torch.float32)
                fidelity_targets.append(fidelity_target)

                batch_belief_p = belief_probs[-1].reshape(-1)
                batch_belief_t = belief_targets[-1].reshape(-1)
                batch_conflict_p = conflict_probs[-1].reshape(-1)
                batch_conflict_t = conflict_targets[-1].reshape(-1)
                batch_fidelity_p = fidelity_probs[-1].reshape(-1)
                batch_fidelity_t = fidelity_target.reshape(-1)
                per_batch.append(
                    StageABatchEvaluation(
                        replicate=start_replicate + offset,
                        batch_digest=batch.digest,
                        examples=batch_size,
                        belief_accuracy=float(((batch_belief_p >= 0.5).to(torch.long) == batch_belief_t.to(torch.long)).to(torch.float32).mean().item()),
                        belief_brier=float(torch.mean((batch_belief_p - batch_belief_t) ** 2).item()),
                        conflict_balanced_accuracy=_balanced_accuracy(batch_conflict_t, (batch_conflict_p >= 0.5).to(torch.long)),
                        conflict_brier=float(torch.mean((batch_conflict_p - batch_conflict_t) ** 2).item()),
                        fidelity_balanced_accuracy=_balanced_accuracy(batch_fidelity_t, (batch_fidelity_p >= 0.5).to(torch.long)),
                        fidelity_brier=float(torch.mean((batch_fidelity_p - batch_fidelity_t) ** 2).item()),
                    )
                )
                digests.append(batch.digest)
    finally:
        model.train(was_training)

    belief_p = torch.cat([value.reshape(-1) for value in belief_probs])
    belief_t = torch.cat([value.reshape(-1) for value in belief_targets])
    conflict_p = torch.cat([value.reshape(-1) for value in conflict_probs])
    conflict_t = torch.cat([value.reshape(-1) for value in conflict_targets])
    fidelity_p = torch.cat([value.reshape(-1) for value in fidelity_probs])
    fidelity_t = torch.cat([value.reshape(-1) for value in fidelity_targets])

    belief_pred = (belief_p >= 0.5).to(torch.long)
    conflict_pred = (conflict_p >= 0.5).to(torch.long)
    fidelity_pred = (fidelity_p >= 0.5).to(torch.long)

    return StageAEvaluationResult(
        rng_stream="evaluation",
        start_replicate=start_replicate,
        batches=batches,
        examples=batches * batch_size,
        batch_digests=tuple(digests),
        per_batch=tuple(per_batch),
        belief_accuracy=float((belief_pred == belief_t.to(torch.long)).to(torch.float32).mean().item()),
        belief_brier=float(torch.mean((belief_p - belief_t) ** 2).item()),
        conflict_balanced_accuracy=_balanced_accuracy(conflict_t, conflict_pred),
        conflict_brier=float(torch.mean((conflict_p - conflict_t) ** 2).item()),
        fidelity_balanced_accuracy=_balanced_accuracy(fidelity_t, fidelity_pred),
        fidelity_brier=float(torch.mean((fidelity_p - fidelity_t) ** 2).item()),
    )


def _report_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("report_digest", None)
    return canonical_sha256(clean)


def validate_heldout_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema", "evidence_level", "decision", "scope", "training_stream", "evaluation_stream",
        "protocol_digest", "code_digest", "config_digest", "model_init_seed",
        "initial_model_digest", "model_before_digest", "model_after_digest",
        "training_steps", "train_start_replicate", "training_curriculum_digests",
        "pre", "post", "delta", "paired_batch_deltas", "report_digest",
    )
    for field in required:
        if field not in report or report[field] in (None, ""):
            errors.append(f"missing {field}")
    if report.get("schema") != "NLM-STAGE-A-HELDOUT-DEV-EVAL-V1":
        errors.append("invalid held-out development report schema")
    if report.get("evidence_level") != "EV-E2":
        errors.append("held-out development report cannot claim EV-E3+")
    if report.get("decision") != "UNVERIFIED":
        errors.append("held-out development report cannot promote a neural claim")
    if report.get("scope") not in (None, "synthetic-stage-a-heldout-development-evaluation"):
        errors.append("invalid held-out development scope")
    if report.get("training_stream") == report.get("evaluation_stream"):
        errors.append("training and evaluation RNG streams must be distinct")
    if report.get("training_stream") not in (None, "augmentation"):
        errors.append("training stream must be augmentation")
    if report.get("evaluation_stream") not in (None, "evaluation"):
        errors.append("evaluation stream must be evaluation")

    training_steps = int(report.get("training_steps", 0) or 0)
    if training_steps <= 0 and "training_steps" in report:
        errors.append("training_steps must be positive")
    training_digests = report.get("training_curriculum_digests") or ()
    if training_steps > 0 and len(training_digests) != training_steps:
        errors.append("training curriculum digest count must match training_steps")

    pre = report.get("pre") or {}
    post = report.get("post") or {}
    if pre.get("rng_stream") not in (None, "evaluation") or post.get("rng_stream") not in (None, "evaluation"):
        errors.append("pre/post results must use the evaluation RNG stream")
    pre_digests = tuple(pre.get("batch_digests") or ())
    post_digests = tuple(post.get("batch_digests") or ())
    if pre_digests != post_digests:
        errors.append("pre/post evaluation must reuse exact held-out batch digests")
    pre_per_batch = tuple(item.get("batch_digest") for item in (pre.get("per_batch") or ()))
    post_per_batch = tuple(item.get("batch_digest") for item in (post.get("per_batch") or ()))
    if pre_per_batch and pre_per_batch != pre_digests:
        errors.append("pre per-batch lineage does not match aggregate batch digests")
    if post_per_batch and post_per_batch != post_digests:
        errors.append("post per-batch lineage does not match aggregate batch digests")
    paired_digests = tuple(item.get("batch_digest") for item in (report.get("paired_batch_deltas") or ()))
    if paired_digests and paired_digests != pre_digests:
        errors.append("paired delta lineage does not match held-out batch digests")

    if report.get("report_digest") not in (None, ""):
        expected = _report_digest(report)
        if report["report_digest"] != expected:
            errors.append("held-out report digest mismatch")
    return errors


def run_heldout_development_evaluation(
    trainer: StageAPilotTrainer,
    *,
    train_steps: int,
    train_start_replicate: int,
    eval_start_replicate: int,
    eval_batches: int,
    batch_size: int,
    variables: int,
    constraints: int,
) -> StageAHeldoutDevelopmentReport:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    pre = evaluate_stage_a(
        trainer.model,
        curriculum=trainer.curriculum,
        start_replicate=eval_start_replicate,
        batches=eval_batches,
        batch_size=batch_size,
        variables=variables,
        constraints=constraints,
    )
    model_before_digest = functional_model_state_digest(trainer.model)
    telemetry = trainer.train_steps(
        steps=train_steps,
        start_replicate=train_start_replicate,
        batch_size=batch_size,
        variables=variables,
        constraints=constraints,
    )
    model_after_digest = functional_model_state_digest(trainer.model)
    post = evaluate_stage_a(
        trainer.model,
        curriculum=trainer.curriculum,
        start_replicate=eval_start_replicate,
        batches=eval_batches,
        batch_size=batch_size,
        variables=variables,
        constraints=constraints,
    )
    if pre.batch_digests != post.batch_digests:
        raise RuntimeError("held-out pre/post evaluation world lineage drifted")

    delta = HeldoutMetricDelta(
        belief_accuracy=post.belief_accuracy - pre.belief_accuracy,
        belief_brier_reduction=pre.belief_brier - post.belief_brier,
        conflict_balanced_accuracy=post.conflict_balanced_accuracy - pre.conflict_balanced_accuracy,
        conflict_brier_reduction=pre.conflict_brier - post.conflict_brier,
        fidelity_balanced_accuracy=post.fidelity_balanced_accuracy - pre.fidelity_balanced_accuracy,
        fidelity_brier_reduction=pre.fidelity_brier - post.fidelity_brier,
    )
    paired_batch_deltas = tuple(
        PairedHeldoutBatchDelta(
            replicate=before.replicate,
            batch_digest=before.batch_digest,
            belief_accuracy=after.belief_accuracy - before.belief_accuracy,
            belief_brier_reduction=before.belief_brier - after.belief_brier,
            conflict_balanced_accuracy=after.conflict_balanced_accuracy - before.conflict_balanced_accuracy,
            conflict_brier_reduction=before.conflict_brier - after.conflict_brier,
            fidelity_balanced_accuracy=after.fidelity_balanced_accuracy - before.fidelity_balanced_accuracy,
            fidelity_brier_reduction=before.fidelity_brier - after.fidelity_brier,
        )
        for before, after in zip(pre.per_batch, post.per_batch, strict=True)
    )
    report = StageAHeldoutDevelopmentReport(
        schema="NLM-STAGE-A-HELDOUT-DEV-EVAL-V1",
        evidence_level="EV-E2",
        decision="UNVERIFIED",
        scope="synthetic-stage-a-heldout-development-evaluation",
        training_stream="augmentation",
        evaluation_stream="evaluation",
        protocol_digest=trainer.protocol_digest,
        code_digest=trainer.code_digest,
        config_digest=trainer.config_digest,
        model_init_seed=trainer.model_init_seed,
        initial_model_digest=trainer.initial_model_digest,
        model_before_digest=model_before_digest,
        model_after_digest=model_after_digest,
        training_steps=train_steps,
        train_start_replicate=train_start_replicate,
        pre=pre,
        post=post,
        delta=delta,
        paired_batch_deltas=paired_batch_deltas,
        training_curriculum_digests=tuple(item.curriculum_digest for item in telemetry),
        report_digest="",
    )
    payload = asdict(report)
    report = replace(report, report_digest=_report_digest(payload))
    errors = validate_heldout_report(asdict(report))
    if errors:
        raise RuntimeError("invalid held-out development report: " + "; ".join(errors))
    return report
