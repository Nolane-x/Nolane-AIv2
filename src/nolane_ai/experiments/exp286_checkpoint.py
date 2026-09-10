from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from .exp286_conflict_worlds import Exp286ConflictGenerator
from .exp286_paired_runner import (
    _build_seeded_pair,
    _functional_state_digest,
    _train_step,
    validate_exp286_paired_development,
)
from .matched_conflict_arms import (
    ChronologicalFailureArm,
    OracleConflictCoreArm,
    audit_matched_exp286_arm_pair,
)

RECEIPT_SCHEMA = "NLM-EXP-286-TRAINED-CHECKPOINT-V1"
TENSOR_SCHEMA = "NLM-EXP-286-TRAINED-TENSORS-V1"
CONTRACT_SCHEMA = "NLM-EXP-286-CHECKPOINT-EXECUTION-CONTRACT-V1"
IDENTITY_SCHEMA = "NLM-EXP-286-TRAINED-CHECKPOINT-SCIENTIFIC-IDENTITY-V1"
STATE_POLICY = "functional-only"
EXPERIMENT_ID = "EXP-286"


def _receipt_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("receipt_digest", None)
    return canonical_sha256(clean)


def _scientific_identity_digest(
    *,
    execution_contract_digest: str,
    chronological_failure_final_digest: str,
    oracle_conflict_core_final_digest: str,
) -> str:
    return canonical_sha256(
        {
            "schema": IDENTITY_SCHEMA,
            "execution_contract_digest": execution_contract_digest,
            "chronological_failure_final_digest": chronological_failure_final_digest,
            "oracle_conflict_core_final_digest": oracle_conflict_core_final_digest,
        }
    )


def _functional_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in functional_trainable_named_parameters(model)
    }


def _load_functional_state(
    model: torch.nn.Module,
    state: dict[str, Any],
    *,
    arm_id: str,
) -> None:
    expected = {
        name: parameter
        for name, parameter in functional_trainable_named_parameters(model)
    }
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"EXP-286 {arm_id} functional checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"EXP-286 {arm_id} checkpoint tensor {name} is not a tensor"
                )
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(
                    f"EXP-286 {arm_id} checkpoint tensor {name} shape/dtype mismatch"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _execution_contract(execution_artifact: dict[str, Any]) -> dict[str, Any]:
    training = execution_artifact.get("training") or {}
    resource = execution_artifact.get("resource_match") or {}
    return {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_digest": execution_artifact.get("protocol_digest"),
        "development_code_digest": execution_artifact.get("code_digest"),
        "development_execution_digest": execution_artifact.get("artifact_digest"),
        "model_init_seed": execution_artifact.get("model_init_seed"),
        "execution_config": deepcopy(execution_artifact.get("execution_config") or {}),
        "training_lineage": {
            "rng_stream": training.get("rng_stream"),
            "start_replicate": training.get("start_replicate"),
            "replicates": training.get("replicates"),
            "paired_batch_digests": deepcopy(training.get("paired_batch_digests") or []),
            "optimizer": deepcopy(training.get("optimizer") or {}),
            "chronological_failure_losses": deepcopy(
                training.get("chronological_failure_losses") or []
            ),
            "oracle_conflict_core_losses": deepcopy(
                training.get("oracle_conflict_core_losses") or []
            ),
        },
        "resource_binding": {
            "declared_max_accounted_flops_per_episode": resource.get(
                "declared_max_accounted_flops_per_episode"
            ),
            "pair_audit_digest": resource.get("pair_audit_digest"),
            "compute_budget_closed": resource.get("compute_budget_closed"),
            "oracle_information_separation": resource.get(
                "oracle_information_separation"
            ),
        },
        "information_receipt": deepcopy(execution_artifact.get("information_receipt") or {}),
        "development_final_state": deepcopy(execution_artifact.get("final_state") or {}),
    }


def _validate_execution_for_replay(execution_artifact: dict[str, Any]) -> None:
    errors = validate_exp286_paired_development(execution_artifact)
    if errors:
        raise ValueError(
            "EXP-286 trained checkpoint requires valid paired DEVELOPMENT execution: "
            + "; ".join(errors)
        )
    if (
        execution_artifact.get("evidence_level") != "EV-E2"
        or execution_artifact.get("decision") != "UNVERIFIED"
    ):
        raise ValueError(
            "EXP-286 trained checkpoint requires EV-E2 / UNVERIFIED DEVELOPMENT evidence"
        )
    if (
        execution_artifact.get("confirmatory_data_consumed") is not False
        or execution_artifact.get("challenge_materialized") is not False
        or execution_artifact.get("challenge_seed_materialized") is not False
    ):
        raise ValueError(
            "EXP-286 checkpoint cannot be reconstructed from confirmatory/challenge data"
        )


def _replay_training(
    execution_artifact: dict[str, Any],
) -> tuple[ChronologicalFailureArm, OracleConflictCoreArm]:
    _validate_execution_for_replay(execution_artifact)
    config = execution_artifact["execution_config"]
    training = execution_artifact["training"]
    resource = execution_artifact["resource_match"]

    chronological, oracle, model_init_seed = _build_seeded_pair(
        root_seed=str(config["root_seed"]),
        d_model=int(config["d_model"]),
        hidden_size=int(config["hidden_size"]),
        target_parameters=int(config["target_parameters"]),
    )
    if int(model_init_seed) != int(execution_artifact["model_init_seed"]):
        raise RuntimeError("EXP-286 model-init seed replay mismatch")

    ceiling = int(resource["declared_max_accounted_flops_per_episode"])
    replay_audit = audit_matched_exp286_arm_pair(
        chronological,
        oracle,
        timesteps=int(config["timesteps"]),
        variables=int(config["variables"]),
        max_search_steps=int(config["max_search_steps"]),
        max_accounted_flops_per_episode=ceiling,
    )
    if canonical_sha256(replay_audit) != resource["pair_audit_digest"]:
        raise RuntimeError("EXP-286 checkpoint replay pair-audit mismatch")
    if (
        replay_audit.get("compute_budget_closed") is not True
        or replay_audit.get("oracle_information_separation") is not True
    ):
        raise RuntimeError("EXP-286 checkpoint replay resource/oracle court failed")

    optimizer_contract = training["optimizer"]
    optimizers = {
        "chronological_failure": build_functional_optimizer(
            chronological,
            lr=float(optimizer_contract["lr"]),
            weight_decay=float(optimizer_contract["weight_decay"]),
        ),
        "oracle_conflict_core": build_functional_optimizer(
            oracle,
            lr=float(optimizer_contract["lr"]),
            weight_decay=float(optimizer_contract["weight_decay"]),
        ),
    }
    arms: dict[str, ChronologicalFailureArm | OracleConflictCoreArm] = {
        "chronological_failure": chronological,
        "oracle_conflict_core": oracle,
    }
    expected_losses = {
        "chronological_failure": list(training["chronological_failure_losses"]),
        "oracle_conflict_core": list(training["oracle_conflict_core_losses"]),
    }
    generator = Exp286ConflictGenerator(root_seed=str(config["root_seed"]))
    batch_digests = list(training["paired_batch_digests"])

    for replicate in range(int(training["replicates"])):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=int(config["batch_size"]),
            timesteps=int(config["timesteps"]),
            variables=int(config["variables"]),
            decoys=int(config["decoys"]),
            d_model=int(config["d_model"]),
            noise_std=float(config["noise_std"]),
            rng_stream="augmentation",
        )
        if batch.digest != batch_digests[replicate]:
            raise RuntimeError(
                f"EXP-286 training batch digest replay mismatch at replicate {replicate}"
            )
        for arm_id in ("chronological_failure", "oracle_conflict_core"):
            loss = _train_step(
                arms[arm_id],
                optimizers[arm_id],
                arm_id=arm_id,
                batch=batch,
            )
            if loss != float(expected_losses[arm_id][replicate]):
                raise RuntimeError(
                    f"EXP-286 {arm_id} training loss replay mismatch at replicate {replicate}"
                )

    final_state = execution_artifact["final_state"]
    chronological_digest = _functional_state_digest(chronological)
    oracle_digest = _functional_state_digest(oracle)
    if chronological_digest != final_state["chronological_failure_digest"]:
        raise RuntimeError(
            "EXP-286 chronological replay final functional digest mismatch"
        )
    if oracle_digest != final_state["oracle_conflict_core_digest"]:
        raise RuntimeError("EXP-286 oracle replay final functional digest mismatch")

    chronological.eval()
    oracle.eval()
    return chronological, oracle


def validate_exp286_checkpoint_receipt(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != RECEIPT_SCHEMA:
        errors.append("invalid EXP-286 checkpoint receipt schema")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-286 checkpoint experiment identity drift")
    if (
        payload.get("evidence_level") != "EV-E2"
        or payload.get("decision") != "UNVERIFIED"
    ):
        errors.append("EXP-286 trained checkpoint must remain EV-E2 / UNVERIFIED")
    if payload.get("state_policy") != STATE_POLICY:
        errors.append("EXP-286 checkpoint state policy drift")
    if payload.get("training_replay_verified") is not True:
        errors.append("EXP-286 checkpoint training replay is not verified")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-286 checkpoint cannot consume confirmatory data")
    if payload.get("challenge_materialized") is not False:
        errors.append("EXP-286 checkpoint cannot materialize challenge data")
    if payload.get("challenge_seed_materialized") is not False:
        errors.append("EXP-286 checkpoint cannot materialize challenge seed")

    contract = payload.get("execution_contract") or {}
    if contract.get("schema") != CONTRACT_SCHEMA:
        errors.append("EXP-286 checkpoint execution contract schema drift")
    if contract.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-286 checkpoint execution contract identity drift")
    expected_contract_digest = canonical_sha256(contract)
    if payload.get("execution_contract_digest") != expected_contract_digest:
        errors.append("EXP-286 checkpoint execution contract digest mismatch")

    lineage = contract.get("training_lineage") or {}
    replicate_count = lineage.get("replicates")
    if (
        not isinstance(replicate_count, int)
        or isinstance(replicate_count, bool)
        or replicate_count <= 0
        or lineage.get("rng_stream") != "augmentation"
        or lineage.get("start_replicate") != 0
        or len(lineage.get("paired_batch_digests") or []) != replicate_count
        or len(lineage.get("chronological_failure_losses") or []) != replicate_count
        or len(lineage.get("oracle_conflict_core_losses") or []) != replicate_count
    ):
        errors.append("EXP-286 checkpoint training lineage drift")
    optimizer = lineage.get("optimizer") or {}
    if (
        optimizer.get("type") != "AdamW"
        or not isinstance(optimizer.get("lr"), (int, float))
        or isinstance(optimizer.get("lr"), bool)
        or float(optimizer.get("lr", 0.0)) <= 0.0
        or not isinstance(optimizer.get("weight_decay"), (int, float))
        or isinstance(optimizer.get("weight_decay"), bool)
        or float(optimizer.get("weight_decay", -1.0)) < 0.0
    ):
        errors.append("EXP-286 checkpoint optimizer contract drift")

    resource = contract.get("resource_binding") or {}
    if (
        resource.get("compute_budget_closed") is not True
        or resource.get("oracle_information_separation") is not True
        or not isinstance(resource.get("declared_max_accounted_flops_per_episode"), int)
        or isinstance(resource.get("declared_max_accounted_flops_per_episode"), bool)
        or int(resource.get("declared_max_accounted_flops_per_episode", 0)) <= 0
        or not isinstance(resource.get("pair_audit_digest"), str)
        or len(resource.get("pair_audit_digest", "")) != 64
    ):
        errors.append("EXP-286 checkpoint resource binding drift")

    final_state = contract.get("development_final_state") or {}
    chronological_digest = payload.get("chronological_failure_final_digest")
    oracle_digest = payload.get("oracle_conflict_core_final_digest")
    if chronological_digest != final_state.get("chronological_failure_digest"):
        errors.append("EXP-286 chronological checkpoint/final-state digest mismatch")
    if oracle_digest != final_state.get("oracle_conflict_core_digest"):
        errors.append("EXP-286 oracle checkpoint/final-state digest mismatch")
    for name, digest in (
        ("chronological", chronological_digest),
        ("oracle", oracle_digest),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"EXP-286 {name} checkpoint final digest missing")
        else:
            try:
                int(digest, 16)
            except ValueError:
                errors.append(f"EXP-286 {name} checkpoint final digest invalid")

    checkpoint_sha = payload.get("checkpoint_file_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        errors.append("EXP-286 checkpoint file SHA256 missing")
    else:
        try:
            int(checkpoint_sha, 16)
        except ValueError:
            errors.append("EXP-286 checkpoint file SHA256 invalid")

    expected_identity = _scientific_identity_digest(
        execution_contract_digest=str(payload.get("execution_contract_digest") or ""),
        chronological_failure_final_digest=str(chronological_digest or ""),
        oracle_conflict_core_final_digest=str(oracle_digest or ""),
    )
    if payload.get("scientific_identity_digest") != expected_identity:
        errors.append("EXP-286 checkpoint scientific identity digest mismatch")
    if payload.get("receipt_digest") != _receipt_digest(payload):
        errors.append("EXP-286 checkpoint receipt digest mismatch")
    return errors


def build_exp286_trained_checkpoint(
    *,
    execution_artifact: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")

    chronological, oracle = _replay_training(execution_artifact)
    execution_contract = _execution_contract(execution_artifact)
    execution_contract_digest = canonical_sha256(execution_contract)
    chronological_digest = _functional_state_digest(chronological)
    oracle_digest = _functional_state_digest(oracle)

    tensor_payload = {
        "schema": TENSOR_SCHEMA,
        "state_policy": STATE_POLICY,
        "execution_contract": execution_contract,
        "execution_contract_digest": execution_contract_digest,
        "chronological_failure_state": _functional_state(chronological),
        "oracle_conflict_core_state": _functional_state(oracle),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_payload, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "state_policy": STATE_POLICY,
        "training_replay_verified": True,
        "confirmatory_data_consumed": False,
        "challenge_seed_materialized": False,
        "challenge_materialized": False,
        "execution_contract": execution_contract,
        "execution_contract_digest": execution_contract_digest,
        "chronological_failure_final_digest": chronological_digest,
        "oracle_conflict_core_final_digest": oracle_digest,
        "checkpoint_file_sha256": checkpoint_sha,
        "scientific_identity_digest": _scientific_identity_digest(
            execution_contract_digest=execution_contract_digest,
            chronological_failure_final_digest=chronological_digest,
            oracle_conflict_core_final_digest=oracle_digest,
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    errors = validate_exp286_checkpoint_receipt(receipt)
    if errors:
        checkpoint_path.unlink(missing_ok=True)
        raise RuntimeError(
            "invalid EXP-286 trained checkpoint receipt before return: "
            + "; ".join(errors)
        )
    return receipt


def load_exp286_trained_checkpoint(
    *,
    checkpoint_path: str | Path,
    receipt: dict[str, Any],
) -> tuple[ChronologicalFailureArm, OracleConflictCoreArm]:
    errors = validate_exp286_checkpoint_receipt(receipt)
    if errors:
        raise ValueError("invalid EXP-286 checkpoint receipt: " + "; ".join(errors))

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file is missing: {checkpoint_path}")
    if file_sha256(checkpoint_path) != receipt["checkpoint_file_sha256"]:
        raise ValueError("EXP-286 checkpoint file SHA256 mismatch")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("EXP-286 checkpoint tensor payload is not a mapping")
    if payload.get("schema") != TENSOR_SCHEMA:
        raise ValueError("EXP-286 checkpoint tensor schema drift")
    if payload.get("state_policy") != STATE_POLICY:
        raise ValueError("EXP-286 checkpoint tensor state policy drift")
    if payload.get("execution_contract_digest") != receipt["execution_contract_digest"]:
        raise ValueError("EXP-286 checkpoint tensor contract digest mismatch")
    if payload.get("execution_contract") != receipt["execution_contract"]:
        raise ValueError("EXP-286 checkpoint tensor execution contract mismatch")

    config = receipt["execution_contract"]["execution_config"]
    chronological, oracle, model_init_seed = _build_seeded_pair(
        root_seed=str(config["root_seed"]),
        d_model=int(config["d_model"]),
        hidden_size=int(config["hidden_size"]),
        target_parameters=int(config["target_parameters"]),
    )
    if int(model_init_seed) != int(receipt["execution_contract"]["model_init_seed"]):
        raise ValueError("EXP-286 checkpoint model-init identity mismatch")

    chronological_state = payload.get("chronological_failure_state")
    oracle_state = payload.get("oracle_conflict_core_state")
    if not isinstance(chronological_state, dict) or not isinstance(oracle_state, dict):
        raise ValueError("EXP-286 checkpoint functional state payload missing")
    _load_functional_state(
        chronological,
        chronological_state,
        arm_id="chronological_failure",
    )
    _load_functional_state(
        oracle,
        oracle_state,
        arm_id="oracle_conflict_core",
    )

    if _functional_state_digest(chronological) != receipt[
        "chronological_failure_final_digest"
    ]:
        raise ValueError("EXP-286 chronological loaded functional digest mismatch")
    if _functional_state_digest(oracle) != receipt["oracle_conflict_core_final_digest"]:
        raise ValueError("EXP-286 oracle loaded functional digest mismatch")

    chronological.eval()
    oracle.eval()
    return chronological, oracle
