from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from .evaluation import StageAHeldoutDevelopmentReport, validate_heldout_report
from .pilot import StageAPilotTrainer

ARTIFACT_SCHEMA = "NLM-STAGE-A-HELDOUT-DEV-ARTIFACT-V1"


def _optimizer_contract(trainer: StageAPilotTrainer) -> dict[str, Any]:
    if len(trainer.optimizer.param_groups) != 1:
        raise ValueError("held-out development artifact requires one optimizer parameter group")
    group = trainer.optimizer.param_groups[0]
    betas = group.get("betas", (0.9, 0.999))
    return {
        "type": type(trainer.optimizer).__name__,
        "lr": float(group["lr"]),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "betas": [float(betas[0]), float(betas[1])],
        "eps": float(group.get("eps", 1e-8)),
        "amsgrad": bool(group.get("amsgrad", False)),
        "maximize": bool(group.get("maximize", False)),
    }


def _support(target: torch.Tensor) -> dict[str, int]:
    flat = target.detach().cpu().to(torch.long).reshape(-1)
    return {
        "positive": int((flat == 1).sum().item()),
        "negative": int((flat == 0).sum().item()),
    }


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def validate_heldout_artifact(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != ARTIFACT_SCHEMA:
        errors.append("invalid held-out artifact schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("held-out artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("held-out artifact cannot promote a neural claim")
    report = payload.get("report") or {}
    errors.extend(f"report: {item}" for item in validate_heldout_report(report))
    contract = payload.get("execution_contract") or {}
    training = contract.get("training") or {}
    evaluation = contract.get("evaluation") or {}
    if training.get("rng_stream") != "augmentation":
        errors.append("artifact training contract must use augmentation RNG")
    if evaluation.get("rng_stream") != "evaluation":
        errors.append("artifact evaluation contract must use evaluation RNG")
    if not training.get("optimizer"):
        errors.append("artifact must bind optimizer hyperparameters")
    support = payload.get("class_support") or {}
    aggregate = support.get("aggregate") or {}
    batch_size = int(evaluation.get("batch_size", 0) or 0)
    batches = int(evaluation.get("batches", 0) or 0)
    variables = int(evaluation.get("variables", 0) or 0)
    constraints = int(evaluation.get("constraints", 0) or 0)
    expected = {
        "belief": batches * batch_size * variables,
        "conflict": batches * batch_size * constraints,
        "fidelity": batches * batch_size,
    }
    for head, total in expected.items():
        counts = aggregate.get(head) or {}
        if int(counts.get("positive", 0)) + int(counts.get("negative", 0)) != total:
            errors.append(f"{head} class-support count does not match evaluation geometry")
    if payload.get("artifact_digest") not in (None, ""):
        if payload["artifact_digest"] != _artifact_digest(payload):
            errors.append("held-out artifact digest mismatch")
    return errors


def build_heldout_artifact(
    report: StageAHeldoutDevelopmentReport,
    *,
    trainer: StageAPilotTrainer,
    train_steps: int,
    train_start_replicate: int,
    eval_start_replicate: int,
    eval_batches: int,
    batch_size: int,
    variables: int,
    constraints: int,
) -> dict[str, Any]:
    if min(train_steps, eval_batches, batch_size, variables, constraints) <= 0:
        raise ValueError("artifact geometry must be positive")
    report_payload = asdict(report)
    report_errors = validate_heldout_report(report_payload)
    if report_errors:
        raise ValueError("invalid held-out report: " + "; ".join(report_errors))

    observed: list[dict[str, Any]] = []
    aggregate = {
        "belief": {"positive": 0, "negative": 0},
        "conflict": {"positive": 0, "negative": 0},
        "fidelity": {"positive": 0, "negative": 0},
    }
    digests: list[str] = []
    for offset in range(eval_batches):
        batch = trainer.curriculum.make_batch(
            replicate=eval_start_replicate + offset,
            batch_size=batch_size,
            variables=variables,
            constraints=constraints,
            d_model=trainer.model.config.d_model,
            device="cpu",
            rng_stream="evaluation",
        )
        digests.append(batch.digest)
        row = {
            "replicate": eval_start_replicate + offset,
            "batch_digest": batch.digest,
            "belief": _support(batch.batch.belief_targets),
            "conflict": _support(batch.batch.conflict_targets),
            "fidelity": _support(batch.batch.fidelity_targets),
        }
        observed.append(row)
        for head in ("belief", "conflict", "fidelity"):
            aggregate[head]["positive"] += row[head]["positive"]
            aggregate[head]["negative"] += row[head]["negative"]
    if tuple(digests) != tuple(report.pre.batch_digests) or tuple(digests) != tuple(report.post.batch_digests):
        raise RuntimeError("artifact class-support regeneration does not match held-out report lineage")

    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scope": "synthetic-stage-a-heldout-development-artifact",
        "report": report_payload,
        "execution_contract": {
            "training": {
                "rng_stream": "augmentation",
                "steps": train_steps,
                "start_replicate": train_start_replicate,
                "batch_size": batch_size,
                "variables": variables,
                "constraints": constraints,
                "optimizer": _optimizer_contract(trainer),
            },
            "evaluation": {
                "rng_stream": "evaluation",
                "start_replicate": eval_start_replicate,
                "batches": eval_batches,
                "batch_size": batch_size,
                "variables": variables,
                "constraints": constraints,
            },
        },
        "class_support": {"per_batch": observed, "aggregate": aggregate},
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_heldout_artifact(payload)
    if errors:
        raise RuntimeError("invalid held-out artifact: " + "; ".join(errors))
    return payload
