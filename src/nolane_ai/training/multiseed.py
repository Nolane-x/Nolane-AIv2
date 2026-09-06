from __future__ import annotations

import hashlib
import math
from typing import Any

from nolane_ai.model.config import NLMConfig
from nolane_ai.protocol.evidence import canonical_sha256
from .curriculum import StageACurriculum
from .evaluation import run_heldout_development_evaluation
from .heldout_artifact import build_heldout_artifact
from .pilot import StageAPilotTrainer, build_seeded_pilot_model, pilot_model_init_seed

SCHEMA = "NLM-STAGE-A-MULTISEED-HELDOUT-DEV-V1"
METRICS = (
    "belief_accuracy",
    "belief_brier_reduction",
    "conflict_balanced_accuracy",
    "conflict_brier_reduction",
    "fidelity_balanced_accuracy",
    "fidelity_brier_reduction",
)


def _bootstrap_index(seed_text: str, sample: int, draw: int, n: int) -> int:
    payload = f"NLM-MULTISEED-BOOTSTRAP-V1|{seed_text}|{sample}|{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _metric_stats(values: list[float], *, bootstrap_seed: str, bootstrap_samples: int) -> dict[str, Any]:
    n = len(values)
    if n <= 0:
        raise ValueError("metric values cannot be empty")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / max(1, n - 1)
    bootstrap_means: list[float] = []
    for sample in range(bootstrap_samples):
        draw = [values[_bootstrap_index(bootstrap_seed, sample, offset, n)] for offset in range(n)]
        bootstrap_means.append(sum(draw) / n)
    bootstrap_means.sort()
    return {
        "n": n,
        "mean": mean,
        "sample_std": math.sqrt(variance),
        "ci_low": _percentile(bootstrap_means, 0.025),
        "ci_high": _percentile(bootstrap_means, 0.975),
        "positive_sign_fraction": sum(value > 0.0 for value in values) / n,
        "zero_sign_fraction": sum(value == 0.0 for value in values) / n,
        "negative_sign_fraction": sum(value < 0.0 for value in values) / n,
    }


def _digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("multiseed_digest", None)
    return canonical_sha256(clean)


def validate_multiseed_heldout(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid multiseed held-out schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("multiseed development artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("multiseed development artifact cannot promote a neural claim")
    seed_count = int(payload.get("seed_count", 0) or 0)
    outcomes = payload.get("per_seed") or []
    if seed_count <= 0:
        errors.append("seed_count must be positive")
    if len(outcomes) != seed_count:
        errors.append("seed_count does not match per_seed outcomes")
    indexes = [item.get("seed_index") for item in outcomes]
    roots = [item.get("root_seed") for item in outcomes]
    if indexes != list(range(seed_count)):
        errors.append("per_seed outcomes must retain every seed index in order")
    if outcomes and len(set(roots)) != len(roots):
        errors.append("per_seed root seeds must be unique")

    successful = [item for item in outcomes if item.get("status") == "SUCCESS"]
    failed = [item for item in outcomes if item.get("status") == "FAILED"]
    unknown = [item for item in outcomes if item.get("status") not in {"SUCCESS", "FAILED"}]
    if unknown:
        errors.append("per_seed outcome has invalid status")
    success_digests = [item.get("artifact_digest") for item in successful]
    if successful and (any(not value for value in success_digests) or len(set(success_digests)) != len(success_digests)):
        errors.append("successful per_seed artifact digests must be non-empty and unique")
    for item in successful:
        delta = item.get("delta") or {}
        if any(metric not in delta for metric in METRICS):
            errors.append("successful per_seed outcome missing required metric delta")
            break
    for item in failed:
        if not item.get("error_type"):
            errors.append("failed per_seed outcome missing error_type")
            break
        if item.get("delta") not in (None, {}):
            errors.append("failed per_seed outcome cannot carry metric deltas")
            break

    aggregate_valid = payload.get("aggregate_valid") is True
    metrics = payload.get("metrics") or {}
    if failed:
        if aggregate_valid:
            errors.append("aggregate statistics cannot be valid when any seed failed")
        for metric in METRICS:
            stats = metrics.get(metric) or {}
            if stats.get("analysis_status") != "INVALID_DUE_TO_FAILED_SEED":
                errors.append(f"{metric} must be invalidated when any seed failed")
                break
            if stats.get("mean") is not None or stats.get("ci_low") is not None or stats.get("ci_high") is not None:
                errors.append(f"{metric} cannot report mean/CI over successful seeds only")
                break
    else:
        if outcomes and not aggregate_valid:
            errors.append("aggregate statistics must be valid when every seed succeeded")
        for metric in METRICS:
            stats = metrics.get(metric) or {}
            if int(stats.get("n", 0) or 0) != seed_count:
                errors.append(f"{metric} statistic count does not match seed_count")
    if payload.get("successful_seed_count") != len(successful):
        errors.append("successful_seed_count mismatch")
    if payload.get("failed_seed_count") != len(failed):
        errors.append("failed_seed_count mismatch")
    if payload.get("analysis_rng") != "sha256-counter-bootstrap-v1":
        errors.append("invalid multiseed analysis RNG contract")
    if payload.get("multiseed_digest") not in (None, "") and payload["multiseed_digest"] != _digest(payload):
        errors.append("multiseed artifact digest mismatch")
    return errors


def run_multiseed_heldout_development(
    *,
    config: NLMConfig,
    root_seed: str,
    seeds: int,
    train_steps: int,
    eval_batches: int,
    batch_size: int,
    variables: int,
    constraints: int,
    bootstrap_samples: int,
    protocol_digest: str,
    code_digest: str,
    train_start_replicate: int = 0,
    eval_start_replicate: int = 10000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    if not root_seed or not protocol_digest or not code_digest:
        raise ValueError("root_seed, protocol_digest and code_digest are required")
    if min(seeds, train_steps, eval_batches, batch_size, variables, constraints, bootstrap_samples) <= 0:
        raise ValueError("multiseed execution counts must be positive")
    if min(train_start_replicate, eval_start_replicate) < 0:
        raise ValueError("replicate indices must be non-negative")

    per_seed: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for seed_index in range(seeds):
        seed_root = f"{root_seed}|seed={seed_index}"
        init_seed = pilot_model_init_seed(seed_root)
        try:
            curriculum = StageACurriculum(root_seed=seed_root)
            trainer = StageAPilotTrainer(
                build_seeded_pilot_model(config, root_seed=seed_root, device="cpu"),
                curriculum=curriculum,
                lr=lr,
                weight_decay=weight_decay,
                protocol_digest=protocol_digest,
                code_digest=code_digest,
                model_init_seed=init_seed,
            )
            report = run_heldout_development_evaluation(
                trainer,
                train_steps=train_steps,
                train_start_replicate=train_start_replicate,
                eval_start_replicate=eval_start_replicate,
                eval_batches=eval_batches,
                batch_size=batch_size,
                variables=variables,
                constraints=constraints,
            )
            artifact = build_heldout_artifact(
                report,
                trainer=trainer,
                train_steps=train_steps,
                train_start_replicate=train_start_replicate,
                eval_start_replicate=eval_start_replicate,
                eval_batches=eval_batches,
                batch_size=batch_size,
                variables=variables,
                constraints=constraints,
            )
            report_payload = artifact["report"]
            delta = {metric: float(report_payload["delta"][metric]) for metric in METRICS}
            for metric, value in delta.items():
                metric_values[metric].append(value)
            per_seed.append(
                {
                    "seed_index": seed_index,
                    "status": "SUCCESS",
                    "root_seed": seed_root,
                    "model_init_seed": init_seed,
                    "artifact_digest": artifact["artifact_digest"],
                    "report_digest": report_payload["report_digest"],
                    "delta": delta,
                }
            )
        except Exception as exc:
            per_seed.append(
                {
                    "seed_index": seed_index,
                    "status": "FAILED",
                    "root_seed": seed_root,
                    "model_init_seed": init_seed,
                    "artifact_digest": None,
                    "report_digest": None,
                    "delta": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    failed_seed_count = sum(item["status"] == "FAILED" for item in per_seed)
    successful_seed_count = seeds - failed_seed_count
    aggregate_valid = failed_seed_count == 0
    if aggregate_valid:
        metrics = {
            metric: _metric_stats(
                values,
                bootstrap_seed=f"{root_seed}|{metric}",
                bootstrap_samples=bootstrap_samples,
            )
            for metric, values in metric_values.items()
        }
    else:
        metrics = {
            metric: {
                "analysis_status": "INVALID_DUE_TO_FAILED_SEED",
                "n_requested": seeds,
                "n_successful": successful_seed_count,
                "mean": None,
                "sample_std": None,
                "ci_low": None,
                "ci_high": None,
                "positive_sign_fraction": None,
                "zero_sign_fraction": None,
                "negative_sign_fraction": None,
            }
            for metric in METRICS
        }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scope": "synthetic-stage-a-multiseed-heldout-development",
        "root_seed": root_seed,
        "seed_count": seeds,
        "successful_seed_count": successful_seed_count,
        "failed_seed_count": failed_seed_count,
        "aggregate_valid": aggregate_valid,
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "config_budget_version": config.budget.version,
        "execution_contract": {
            "train_steps": train_steps,
            "train_start_replicate": train_start_replicate,
            "eval_start_replicate": eval_start_replicate,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "variables": variables,
            "constraints": constraints,
            "lr": lr,
            "weight_decay": weight_decay,
            "bootstrap_samples": bootstrap_samples,
        },
        "analysis_rng": "sha256-counter-bootstrap-v1",
        "per_seed": per_seed,
        "metrics": metrics,
        "multiseed_digest": "",
    }
    payload["multiseed_digest"] = _digest(payload)
    errors = validate_multiseed_heldout(payload)
    if errors:
        raise RuntimeError("invalid multiseed held-out artifact: " + "; ".join(errors))
    return payload
