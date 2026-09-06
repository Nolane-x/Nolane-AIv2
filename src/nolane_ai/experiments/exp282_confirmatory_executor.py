from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256, require_frozen_stage_a_v1_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from .exp282_partial_observability import Exp282PartialObservabilityGenerator
from .exp282_paired_runner import _functional_state_digest, validate_exp282_paired_development
from .exp282_reconstruction_court import validate_exp282_confirmatory_reconstruction
from .matched_belief_arms import build_matched_belief_arm_pair

SCHEMA = "NLM-EXP-282-CONFIRMATORY-OPEN-RAW-V1"
CONFIRMATORY_SCOPE = "synthetic-exp282-partial-observability-confirmatory-open"


def _raw_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _exp282(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1":
        raise ValueError("invalid frozen protocol id")
    if protocol.get("status") != "FROZEN_V1":
        raise ValueError("protocol is not FROZEN_V1")
    rng = protocol.get("rng") or {}
    if not rng.get("root_seed"):
        raise ValueError("frozen protocol RNG root is missing")
    if "evaluation" not in (rng.get("streams") or []):
        raise ValueError("frozen protocol evaluation RNG stream is missing")
    experiment = next(
        (item for item in (protocol.get("experiments") or []) if item.get("experiment_id") == "EXP-282"),
        None,
    )
    if experiment is None:
        raise ValueError("frozen protocol EXP-282 is missing")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != "grounded_decision_accuracy" or primary.get("direction") != "higher":
        raise ValueError("frozen EXP-282 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "absolute_gain" or float(mesi.get("value", -1.0)) != 0.03:
        raise ValueError("frozen EXP-282 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if int(sample.get("min_n", -1)) != 32 or int(sample.get("max_n", -1)) != 128 or sample.get("paired") is not True:
        raise ValueError("frozen EXP-282 sample-size contract drift")
    return experiment


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)[..., 1]
    predictions = logits.argmax(dim=-1)
    correct = int((predictions == targets).sum().item())
    total = int(targets.numel())
    brier_terms = (probabilities - targets.to(torch.float32)) ** 2
    brier_sum = float(brier_terms.sum().item())
    return {
        "grounded_decision_accuracy": correct / total,
        "correct": correct,
        "total": total,
        "brier_score": brier_sum / total,
        "brier_sum": brier_sum,
        "brier_count": total,
    }


def _expected_reserve_keys(model: torch.nn.Module) -> set[str]:
    return {name for name, _ in model.state_dict().items() if "capacity_reserve" in name}


def _load_functional_state(model: torch.nn.Module, state: dict[str, torch.Tensor], *, label: str) -> None:
    if any("capacity_reserve" in name for name in state):
        raise ValueError(f"{label} checkpoint contains capacity_reserve tensor")
    result = model.load_state_dict(state, strict=False)
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    expected_missing = _expected_reserve_keys(model)
    if missing != expected_missing or unexpected:
        raise ValueError(
            f"{label} checkpoint state mismatch: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )


def _validate_protocol_lineage(
    *,
    protocol: dict[str, Any],
    protocol_digest: str,
    paired: dict[str, Any],
    reconstruction: dict[str, Any],
) -> str:
    _exp282(protocol)
    if not protocol_digest:
        raise ValueError("protocol_digest is required")
    require_frozen_stage_a_v1_sha256(protocol_digest)
    if paired.get("protocol_digest") != protocol_digest:
        raise ValueError("paired execution protocol digest mismatch")
    lineage = reconstruction.get("lineage") or {}
    if lineage.get("protocol_digest") != protocol_digest:
        raise ValueError("reconstruction protocol digest mismatch")
    return str((protocol.get("rng") or {})["root_seed"])


def validate_exp282_confirmatory_open_raw(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-282 confirmatory raw schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("confirmatory raw artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("confirmatory raw artifact cannot promote a neural claim")
    if payload.get("status") != "CONFIRMATORY_OPEN_EXECUTED_UNANALYZED":
        errors.append("confirmatory raw status drift")
    if payload.get("confirmatory_data_consumed") is not True:
        errors.append("confirmatory raw artifact must record consumed confirmatory data")
    if payload.get("seed_materialization_status") != "EXECUTED":
        errors.append("confirmatory raw seed materialization status drift")
    if payload.get("challenge_materialized") is not False:
        errors.append("confirmatory-open executor cannot materialize challenge randomness")

    root_seed = payload.get("confirmatory_world_root_seed")
    if not root_seed:
        errors.append("confirmatory world root seed is missing")
    if payload.get("world_rng_stream") != "evaluation":
        errors.append("confirmatory-open worlds must use evaluation RNG")

    confirmatory_n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(confirmatory_n, int) or not (32 <= confirmatory_n <= 128):
        errors.append("confirmatory raw n is outside frozen bounds")
    elif len(reserved) != confirmatory_n:
        errors.append("confirmatory raw reserved replicate count mismatch")
    if len(set(reserved)) != len(reserved):
        errors.append("confirmatory raw reserved replicate IDs must be unique")
    if reserved and reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("confirmatory raw reserved replicate IDs must be contiguous and ordered")

    rows = payload.get("per_replicate") or []
    row_ids = [row.get("replicate") for row in rows]
    if row_ids != reserved:
        errors.append("confirmatory raw replicate lineage must exactly match reserved IDs")
    if isinstance(confirmatory_n, int) and len(rows) != confirmatory_n:
        errors.append("confirmatory raw row count does not match confirmatory n")

    for row in rows:
        replicate = row.get("replicate")
        world = row.get("world") or {}
        if world.get("scope") != CONFIRMATORY_SCOPE:
            errors.append("confirmatory raw world scope drift")
            break
        if world.get("root_seed") != root_seed:
            errors.append("confirmatory raw world root drift")
            break
        if world.get("rng_stream") != "evaluation":
            errors.append("confirmatory raw world RNG stream drift")
            break
        if not isinstance(replicate, int) or replicate < 0:
            errors.append("confirmatory raw replicate ID is invalid")
            break
        if root_seed:
            if world.get("latent_seed") != derive_stream_seed(str(root_seed), "EXP-282", replicate, "environment"):
                errors.append("confirmatory raw latent seed lineage mismatch")
                break
            if world.get("observation_seed") != derive_stream_seed(str(root_seed), "EXP-282", replicate, "evaluation"):
                errors.append("confirmatory raw observation seed lineage mismatch")
                break
            if world.get("code_seed") != derive_stream_seed(str(root_seed), "EXP-282", 0, "intervention"):
                errors.append("confirmatory raw intervention seed lineage mismatch")
                break
        recurrent = row.get("recurrent_hidden") or {}
        explicit = row.get("explicit_belief") or {}
        for arm in (recurrent, explicit):
            if not _finite_number(arm.get("grounded_decision_accuracy")) or not 0.0 <= float(arm["grounded_decision_accuracy"]) <= 1.0:
                errors.append("confirmatory raw accuracy is invalid")
                break
            if not _finite_number(arm.get("brier_score")) or float(arm["brier_score"]) < 0.0:
                errors.append("confirmatory raw Brier score is invalid")
                break
            total = int(arm.get("total", 0) or 0)
            brier_count = int(arm.get("brier_count", 0) or 0)
            correct = int(arm.get("correct", -1))
            if total <= 0 or brier_count != total or not (0 <= correct <= total):
                errors.append("confirmatory raw metric counts are invalid")
                break
            if not math.isclose(float(arm["grounded_decision_accuracy"]), correct / total, rel_tol=0.0, abs_tol=1e-12):
                errors.append("confirmatory raw accuracy/count mismatch")
                break
            if not _finite_number(arm.get("brier_sum")) or not math.isclose(
                float(arm["brier_score"]), float(arm["brier_sum"]) / brier_count, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append("confirmatory raw Brier/count mismatch")
                break
        expected_accuracy = float(explicit.get("grounded_decision_accuracy", math.nan)) - float(
            recurrent.get("grounded_decision_accuracy", math.nan)
        )
        expected_brier = float(explicit.get("brier_score", math.nan)) - float(recurrent.get("brier_score", math.nan))
        if not _finite_number(row.get("explicit_minus_recurrent_accuracy")) or not math.isclose(
            float(row["explicit_minus_recurrent_accuracy"]), expected_accuracy, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("confirmatory raw accuracy contrast mismatch")
            break
        if not _finite_number(row.get("explicit_minus_recurrent_brier")) or not math.isclose(
            float(row["explicit_minus_recurrent_brier"]), expected_brier, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("confirmatory raw Brier contrast mismatch")
            break
        if not row.get("batch_digest"):
            errors.append("confirmatory raw batch digest is missing")
            break

    lineage = payload.get("lineage") or {}
    for key in (
        "protocol_digest",
        "paired_execution_artifact_digest",
        "reconstruction_digest",
        "paired_checkpoint_sha256",
        "execution_contract_digest",
        "executor_code_digest",
    ):
        if not lineage.get(key):
            errors.append(f"missing confirmatory raw lineage {key}")

    if payload.get("artifact_digest") not in (None, "") and payload.get("artifact_digest") != _raw_digest(payload):
        errors.append("confirmatory raw artifact digest mismatch")
    return errors


def execute_exp282_confirmatory_open(
    *,
    protocol: dict[str, Any],
    protocol_digest: str,
    paired_execution_artifact: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    checkpoint_path: str | Path,
    executor_code_digest: str,
) -> dict[str, Any]:
    if not executor_code_digest:
        raise ValueError("executor_code_digest is required")

    paired_errors = validate_exp282_paired_development(paired_execution_artifact)
    if paired_errors:
        raise ValueError("invalid paired execution artifact: " + "; ".join(paired_errors))
    reconstruction_errors = validate_exp282_confirmatory_reconstruction(reconstruction_authorization)
    if reconstruction_errors:
        raise ValueError("invalid reconstruction authorization: " + "; ".join(reconstruction_errors))
    protocol_root = _validate_protocol_lineage(
        protocol=protocol,
        protocol_digest=protocol_digest,
        paired=paired_execution_artifact,
        reconstruction=reconstruction_authorization,
    )

    checkpoint_path = Path(checkpoint_path)
    expected_checkpoint_sha = reconstruction_authorization.get("checkpoint_sha256")
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if not expected_checkpoint_sha or actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("checkpoint SHA mismatch")
    paired_checkpoint = paired_execution_artifact.get("checkpoint") or {}
    if paired_checkpoint.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError("paired artifact checkpoint SHA mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "NLM-EXP-282-PAIRED-TENSORS-V1":
        raise ValueError("paired checkpoint tensor schema drift")
    if checkpoint.get("state_policy") != "functional-only":
        raise ValueError("paired checkpoint state policy drift")
    if checkpoint.get("protocol_digest") != protocol_digest:
        raise ValueError("paired checkpoint protocol digest mismatch")

    checkpoint_contract = checkpoint.get("execution_contract") or {}
    checkpoint_contract_digest = checkpoint.get("execution_contract_digest")
    reconstructed_contract = reconstruction_authorization.get("execution_contract") or {}
    reconstructed_digest = reconstruction_authorization.get("execution_contract_digest")
    if not checkpoint_contract_digest or checkpoint_contract_digest != canonical_sha256(checkpoint_contract):
        raise ValueError("checkpoint execution contract digest mismatch")
    if checkpoint_contract != reconstructed_contract or checkpoint_contract_digest != reconstructed_digest:
        raise ValueError("execution contract mismatch between checkpoint and reconstruction authorization")

    recurrent_state = checkpoint.get("recurrent_hidden_state") or {}
    explicit_state = checkpoint.get("explicit_belief_state") or {}
    if not recurrent_state or not explicit_state:
        raise ValueError("paired checkpoint functional states are missing")
    if any("capacity_reserve" in name for name in [*recurrent_state, *explicit_state]):
        raise ValueError("paired checkpoint contains capacity_reserve tensor")

    arm_geometry = checkpoint_contract.get("arm_geometry") or {}
    recurrent, explicit = build_matched_belief_arm_pair(
        d_model=int(arm_geometry["d_model"]),
        hidden_size=int(arm_geometry["hidden_size"]),
        target_parameters=int(arm_geometry["target_parameters"]),
        device="cpu",
    )
    _load_functional_state(recurrent, recurrent_state, label="recurrent_hidden")
    _load_functional_state(explicit, explicit_state, label="explicit_belief")
    final_state = paired_execution_artifact.get("final_state") or {}
    if _functional_state_digest(recurrent) != final_state.get("recurrent_hidden_digest"):
        raise ValueError("recurrent checkpoint final-state digest mismatch")
    if _functional_state_digest(explicit) != final_state.get("explicit_belief_digest"):
        raise ValueError("explicit checkpoint final-state digest mismatch")
    recurrent.eval()
    explicit.eval()

    reserved = list(reconstruction_authorization.get("reserved_replicate_ids") or [])
    confirmatory_n = int(reconstruction_authorization.get("confirmatory_n", 0) or 0)
    if len(reserved) != confirmatory_n or reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        raise ValueError("reserved confirmatory replicate lineage is invalid")

    world_geometry = checkpoint_contract.get("world_geometry") or {}
    generator = Exp282PartialObservabilityGenerator(root_seed=protocol_root)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for replicate in reserved:
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=int(world_geometry["batch_size"]),
                timesteps=int(world_geometry["timesteps"]),
                variables=int(world_geometry["variables"]),
                d_model=int(world_geometry["d_model"]),
                visibility_rate=float(world_geometry["visibility_rate"]),
                noise_std=float(world_geometry["noise_std"]),
                rng_stream="evaluation",
                scope=CONFIRMATORY_SCOPE,
                device="cpu",
            )
            recurrent_metrics = _metrics(recurrent(batch.observations), batch.targets)
            explicit_metrics = _metrics(explicit(batch.observations), batch.targets)
            world = {
                key: batch.metadata[key]
                for key in (
                    "scope",
                    "root_seed",
                    "replicate",
                    "rng_stream",
                    "latent_seed",
                    "observation_seed",
                    "code_seed",
                )
            }
            rows.append(
                {
                    "replicate": replicate,
                    "batch_digest": batch.digest,
                    "world": world,
                    "recurrent_hidden": recurrent_metrics,
                    "explicit_belief": explicit_metrics,
                    "explicit_minus_recurrent_accuracy": explicit_metrics["grounded_decision_accuracy"]
                    - recurrent_metrics["grounded_decision_accuracy"],
                    "explicit_minus_recurrent_brier": explicit_metrics["brier_score"]
                    - recurrent_metrics["brier_score"],
                }
            )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "CONFIRMATORY_OPEN_EXECUTED_UNANALYZED",
        "scope": "exp282-confirmatory-open-raw-execution",
        "confirmatory_data_consumed": True,
        "seed_materialization_status": "EXECUTED",
        "challenge_materialized": False,
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "confirmatory_world_root_seed": protocol_root,
        "world_rng_stream": "evaluation",
        "per_replicate": rows,
        "lineage": {
            "protocol_digest": protocol_digest,
            "paired_execution_artifact_digest": paired_execution_artifact.get("artifact_digest"),
            "reconstruction_digest": reconstruction_authorization.get("reconstruction_digest"),
            "paired_checkpoint_sha256": actual_checkpoint_sha,
            "execution_contract_digest": checkpoint_contract_digest,
            "executor_code_digest": executor_code_digest,
        },
        "remaining_blockers": [
            "frozen confirmatory analysis has not been applied to this raw artifact",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _raw_digest(payload)
    errors = validate_exp282_confirmatory_open_raw(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 confirmatory-open raw artifact: " + "; ".join(errors))
    return payload
