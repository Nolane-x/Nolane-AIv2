from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes
from .exp282_partial_observability import Exp282PartialObservabilityGenerator
from .matched_belief_arms import (
    ExplicitBeliefArm,
    RecurrentHiddenBeliefArm,
    audit_matched_belief_arm_pair,
    build_matched_belief_arm_pair,
)

SCHEMA = "NLM-EXP-282-PAIRED-DEV-EVAL-V1"


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def validate_exp282_paired_development(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-282 paired development schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-282 paired development artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-282 paired development artifact cannot promote a claim")
    if payload.get("confirmatory_ready") is not False:
        errors.append("EXP-282 paired development artifact cannot be confirmatory-ready")
    if not payload.get("protocol_digest") or not payload.get("code_digest"):
        errors.append("EXP-282 paired development provenance is incomplete")

    initial = payload.get("initial_state") or {}
    if initial.get("functional_digest_match") is not True or initial.get("recurrent_hidden_digest") != initial.get("explicit_belief_digest"):
        errors.append("EXP-282 matched arms must share identical functional initialization")

    resource = payload.get("resource_match") or {}
    if resource.get("parameter_match") is not True:
        errors.append("EXP-282 parameter match must be closed")
    if resource.get("observation_history_match") is not True:
        errors.append("EXP-282 observation-history match must be closed")
    if resource.get("accounted_flop_match") is not True:
        errors.append("EXP-282 accounted FLOP match must be closed")
    if float(resource.get("relative_accounted_flop_difference", 1.0)) > 0.05:
        errors.append("EXP-282 accounted FLOP difference exceeds frozen guard")

    training = payload.get("training") or {}
    evaluation = payload.get("evaluation") or {}
    if training.get("rng_stream") != "augmentation":
        errors.append("EXP-282 training stream must be augmentation")
    if evaluation.get("rng_stream") != "evaluation":
        errors.append("EXP-282 evaluation stream must be evaluation")
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = training.get("batch_digests") or []
    if train_count <= 0 or len(train_digests) != train_count:
        errors.append("EXP-282 training replicate count does not match batch lineage")
    eval_count = int(evaluation.get("replicates", 0) or 0)
    raw = evaluation.get("per_replicate") or []
    if eval_count <= 0 or len(raw) != eval_count:
        errors.append("EXP-282 evaluation replicate count does not match raw lineage")
    if raw:
        start = int(evaluation.get("start_replicate", -1))
        indexes = [row.get("replicate") for row in raw]
        if indexes != list(range(start, start + len(raw))):
            errors.append("EXP-282 evaluation replicate lineage is reordered or incomplete")
        digests = [row.get("batch_digest") for row in raw]
        if any(not digest for digest in digests) or len(set(digests)) != len(digests):
            errors.append("EXP-282 evaluation batch digests must be non-empty and unique")

    primary = payload.get("primary_endpoint") or {}
    protected = payload.get("protected_endpoints") or {}
    if primary.get("metric") != "grounded_decision_accuracy":
        errors.append("EXP-282 primary endpoint drift")
    if protected.get("brier_score_guard") != "explicit_belief <= recurrent_hidden + 0.02":
        errors.append("EXP-282 Brier guard drift")
    if protected.get("accounted_flops_guard") != "relative difference <= 0.05":
        errors.append("EXP-282 FLOP guard drift")

    if payload.get("artifact_digest") not in (None, "") and payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-282 paired artifact digest mismatch")
    return errors


def _functional_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-282-FUNCTIONAL-STATE-V1\0")
    for name, parameter in functional_trainable_named_parameters(model):
        cpu = parameter.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype), "byteorder": tensor_byteorder()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(header)
        hasher.update(b"\0")
        hasher.update(tensor_raw_bytes(cpu))
    return hasher.hexdigest()


def _build_seeded_pair(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
) -> tuple[RecurrentHiddenBeliefArm, ExplicitBeliefArm, int]:
    seed = derive_stream_seed(root_seed, "EXP-282", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        recurrent, explicit = build_matched_belief_arm_pair(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            device="cpu",
        )
    explicit.load_state_dict(recurrent.state_dict())
    return recurrent, explicit, seed


def _evaluate_arm(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=-1)[..., 1]
    predictions = logits.argmax(dim=-1)
    return {
        "grounded_decision_accuracy": float((predictions == targets).to(torch.float32).mean().item()),
        "brier_score": float(((probabilities - targets.to(torch.float32)) ** 2).mean().item()),
    }


def _train_one(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(observations)
    loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def run_exp282_paired_development(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    train_replicates: int,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    visibility_rate: float,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    if not root_seed:
        raise ValueError("root_seed is required")
    if not protocol_digest or not code_digest:
        raise ValueError("protocol_digest and code_digest are required")
    if min(
        d_model,
        hidden_size,
        target_parameters,
        train_replicates,
        eval_replicates,
        batch_size,
        timesteps,
        variables,
    ) <= 0:
        raise ValueError("execution counts and dimensions must be positive")
    if eval_start_replicate < 0:
        raise ValueError("eval_start_replicate must be non-negative")
    if lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer hyperparameters are invalid")

    recurrent, explicit, model_init_seed = _build_seeded_pair(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
    )
    initial_recurrent = _functional_state_digest(recurrent)
    initial_explicit = _functional_state_digest(explicit)
    if initial_recurrent != initial_explicit:
        raise RuntimeError("matched EXP-282 arms do not share identical functional initialization")

    pair_audit = audit_matched_belief_arm_pair(
        recurrent,
        explicit,
        timesteps=timesteps,
        variables=variables,
    )
    if not (
        pair_audit["parameter_match"]
        and pair_audit["observation_history_match"]
        and pair_audit["full_accounted_flop_match"]
        and pair_audit["relative_accounted_flop_difference"] <= 0.05
    ):
        raise RuntimeError("EXP-282 resource match did not close before paired execution")

    generator = Exp282PartialObservabilityGenerator(root_seed=root_seed)
    recurrent_optimizer = build_functional_optimizer(recurrent, lr=lr, weight_decay=weight_decay)
    explicit_optimizer = build_functional_optimizer(explicit, lr=lr, weight_decay=weight_decay)
    training_digests: list[str] = []
    recurrent_losses: list[float] = []
    explicit_losses: list[float] = []
    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            variables=variables,
            d_model=d_model,
            visibility_rate=visibility_rate,
            noise_std=noise_std,
            rng_stream="augmentation",
        )
        training_digests.append(batch.digest)
        recurrent_losses.append(_train_one(recurrent, recurrent_optimizer, batch.observations, batch.targets))
        explicit_losses.append(_train_one(explicit, explicit_optimizer, batch.observations, batch.targets))

    recurrent.eval()
    explicit.eval()
    per_replicate: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(eval_replicates):
            replicate = eval_start_replicate + offset
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                d_model=d_model,
                visibility_rate=visibility_rate,
                noise_std=noise_std,
                rng_stream="evaluation",
            )
            recurrent_metrics = _evaluate_arm(recurrent(batch.observations), batch.targets)
            explicit_metrics = _evaluate_arm(explicit(batch.observations), batch.targets)
            per_replicate.append(
                {
                    "replicate": replicate,
                    "batch_digest": batch.digest,
                    "recurrent_hidden": recurrent_metrics,
                    "explicit_belief": explicit_metrics,
                    "explicit_minus_recurrent_accuracy": explicit_metrics["grounded_decision_accuracy"]
                    - recurrent_metrics["grounded_decision_accuracy"],
                    "explicit_minus_recurrent_brier": explicit_metrics["brier_score"]
                    - recurrent_metrics["brier_score"],
                }
            )

    mean_accuracy_gain = sum(row["explicit_minus_recurrent_accuracy"] for row in per_replicate) / eval_replicates
    mean_brier_difference = sum(row["explicit_minus_recurrent_brier"] for row in per_replicate) / eval_replicates
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scope": "synthetic-exp282-paired-partial-observability-development",
        "confirmatory_ready": False,
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "initial_state": {
            "recurrent_hidden_digest": initial_recurrent,
            "explicit_belief_digest": initial_explicit,
            "functional_digest_match": initial_recurrent == initial_explicit,
        },
        "final_state": {
            "recurrent_hidden_digest": _functional_state_digest(recurrent),
            "explicit_belief_digest": _functional_state_digest(explicit),
        },
        "resource_match": {
            "parameter_match": pair_audit["parameter_match"],
            "observation_history_match": pair_audit["observation_history_match"],
            "accounted_flop_match": pair_audit["full_accounted_flop_match"],
            "relative_accounted_flop_difference": pair_audit["relative_accounted_flop_difference"],
            "compute_ledger": pair_audit["compute_ledger"],
        },
        "primary_endpoint": {
            "metric": "grounded_decision_accuracy",
            "direction": "higher",
            "mesi_absolute_gain": 0.03,
        },
        "protected_endpoints": {
            "brier_score_guard": "explicit_belief <= recurrent_hidden + 0.02",
            "accounted_flops_guard": "relative difference <= 0.05",
        },
        "training": {
            "rng_stream": "augmentation",
            "replicates": train_replicates,
            "batch_digests": training_digests,
            "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": weight_decay},
            "recurrent_hidden_losses": recurrent_losses,
            "explicit_belief_losses": explicit_losses,
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": eval_start_replicate,
            "replicates": eval_replicates,
            "per_replicate": per_replicate,
            "aggregate": {
                "n": eval_replicates,
                "mean_accuracy_gain": mean_accuracy_gain,
                "mean_brier_difference": mean_brier_difference,
            },
        },
        "world_geometry": {
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "d_model": d_model,
            "visibility_rate": visibility_rate,
            "noise_std": noise_std,
        },
        "remaining_blockers": [
            "development worlds remain synthetic and cannot establish EV-E3 by themselves",
            "confirmatory sample size, paired analysis freeze, and post-freeze challenge execution are not closed",
        ],
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp282_paired_development(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 paired development artifact: " + "; ".join(errors))
    return payload
