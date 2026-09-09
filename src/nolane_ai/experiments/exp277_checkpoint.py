from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from .exp277_paired_runner import (
    _build_seeded_pair,
    _functional_state_digest,
    _train_arcs,
    _train_oracle,
)
from .exp277_structure_dense import Exp277StructureDenseGenerator
from .matched_cbrf_arms import ARCSBranchArm, OracleCBRFArm, audit_matched_exp277_arm_pair

RECEIPT_SCHEMA = "NLM-EXP-277-TRAINED-CHECKPOINT-V1"
TENSOR_SCHEMA = "NLM-EXP-277-TRAINED-TENSORS-V1"
STATE_POLICY = "functional-only"


def _receipt_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("receipt_digest", None)
    return canonical_sha256(clean)


def _scientific_identity_digest(
    *,
    execution_contract_digest: str,
    arcs_branch_final_digest: str,
    oracle_cbrf_final_digest: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "NLM-EXP-277-TRAINED-CHECKPOINT-SCIENTIFIC-IDENTITY-V1",
            "execution_contract_digest": execution_contract_digest,
            "arcs_branch_final_digest": arcs_branch_final_digest,
            "oracle_cbrf_final_digest": oracle_cbrf_final_digest,
        }
    )


def _functional_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in functional_trainable_named_parameters(model)
    }


def _load_functional_state(model: torch.nn.Module, state: dict[str, Any], *, arm_id: str) -> None:
    expected = {name: parameter for name, parameter in functional_trainable_named_parameters(model)}
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"EXP-277 {arm_id} functional checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"EXP-277 {arm_id} checkpoint tensor {name} is not a tensor")
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"EXP-277 {arm_id} checkpoint tensor {name} shape/dtype mismatch")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _execution_contract(execution_artifact: dict[str, Any]) -> dict[str, Any]:
    training = execution_artifact.get("training") or {}
    return {
        "root_seed": execution_artifact.get("root_seed"),
        "model_init_seed": execution_artifact.get("model_init_seed"),
        "protocol_digest": execution_artifact.get("protocol_digest"),
        "development_code_digest": execution_artifact.get("code_digest"),
        "arm_geometry": deepcopy(execution_artifact.get("arm_geometry") or {}),
        "world_geometry": deepcopy(execution_artifact.get("world_geometry") or {}),
        "train_replicates": training.get("replicates"),
        "training_lineage": {
            "rng_stream": training.get("rng_stream"),
            "start_replicate": training.get("start_replicate"),
            "replicates": training.get("replicates"),
            "paired_batch_digests": deepcopy(training.get("paired_batch_digests") or []),
        },
        "optimizer": deepcopy(training.get("optimizer") or {}),
        "development_final_state": deepcopy(execution_artifact.get("final_state") or {}),
        "declared_max_accounted_flops_per_episode": (execution_artifact.get("resource_match") or {}).get(
            "declared_max_accounted_flops_per_episode"
        ),
    }


def _validate_execution_for_replay(execution_artifact: dict[str, Any]) -> None:
    if execution_artifact.get("schema") != "NLM-EXP-277-PAIRED-DEV-EVAL-V1":
        raise ValueError("EXP-277 trained checkpoint requires paired DEVELOPMENT execution")
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-277 trained checkpoint requires EV-E2 / UNVERIFIED DEVELOPMENT evidence")
    if execution_artifact.get("confirmatory_data_consumed") is not False or execution_artifact.get("challenge_materialized") is not False:
        raise ValueError("EXP-277 checkpoint cannot be reconstructed from confirmatory/challenge data")
    if not execution_artifact.get("root_seed") or not execution_artifact.get("protocol_digest") or not execution_artifact.get("code_digest"):
        raise ValueError("EXP-277 DEVELOPMENT checkpoint lineage is incomplete")
    arm_geometry = execution_artifact.get("arm_geometry") or {}
    world_geometry = execution_artifact.get("world_geometry") or {}
    for key in ("d_model", "hidden_size", "target_parameters"):
        if not isinstance(arm_geometry.get(key), int) or isinstance(arm_geometry.get(key), bool) or int(arm_geometry[key]) <= 0:
            raise ValueError(f"EXP-277 checkpoint arm geometry {key} is invalid")
    for key in ("batch_size", "timesteps", "variables", "constraints", "d_model"):
        if not isinstance(world_geometry.get(key), int) or isinstance(world_geometry.get(key), bool) or int(world_geometry[key]) <= 0:
            raise ValueError(f"EXP-277 checkpoint world geometry {key} is invalid")
    if int(world_geometry["d_model"]) != int(arm_geometry["d_model"]):
        raise ValueError("EXP-277 checkpoint d_model geometry mismatch")
    noise_std = world_geometry.get("noise_std")
    if not isinstance(noise_std, (int, float)) or float(noise_std) < 0.0:
        raise ValueError("EXP-277 checkpoint noise geometry is invalid")
    training = execution_artifact.get("training") or {}
    if training.get("rng_stream") != "augmentation" or training.get("start_replicate") != 0:
        raise ValueError("EXP-277 checkpoint training lineage drift")
    train_count = training.get("replicates")
    batch_digests = training.get("paired_batch_digests") or []
    arcs_losses = training.get("arcs_branch_losses") or []
    oracle_losses = training.get("oracle_cbrf_losses") or []
    if not isinstance(train_count, int) or isinstance(train_count, bool) or train_count <= 0:
        raise ValueError("EXP-277 checkpoint training replicate count invalid")
    if len(batch_digests) != train_count or len(arcs_losses) != train_count or len(oracle_losses) != train_count:
        raise ValueError("EXP-277 checkpoint training lineage/loss count mismatch")
    optimizer = training.get("optimizer") or {}
    if optimizer.get("type") != "AdamW":
        raise ValueError("EXP-277 checkpoint optimizer type drift")
    if not isinstance(optimizer.get("lr"), (int, float)) or float(optimizer["lr"]) <= 0.0:
        raise ValueError("EXP-277 checkpoint optimizer lr invalid")
    if not isinstance(optimizer.get("weight_decay"), (int, float)) or float(optimizer["weight_decay"]) < 0.0:
        raise ValueError("EXP-277 checkpoint optimizer weight decay invalid")
    resource = execution_artifact.get("resource_match") or {}
    for key in ("parameter_match", "functional_parameter_match", "same_world_lineage", "compute_budget_closed"):
        if resource.get(key) is not True:
            raise ValueError(f"EXP-277 checkpoint resource court open: {key}")
    receipt = execution_artifact.get("oracle_information_receipt") or {}
    if receipt.get("arcs_received_oracle_incidence") is not False or receipt.get("delivered_to") != ["oracle_cbrf"]:
        raise ValueError("EXP-277 checkpoint oracle-information separation drift")
    final_state = execution_artifact.get("final_state") or {}
    if not final_state.get("arcs_branch_digest") or not final_state.get("oracle_cbrf_digest"):
        raise ValueError("EXP-277 DEVELOPMENT final functional digests missing")


def _replay_training(execution_artifact: dict[str, Any]) -> tuple[ARCSBranchArm, OracleCBRFArm, list[float], list[float]]:
    _validate_execution_for_replay(execution_artifact)
    arm_geometry = execution_artifact["arm_geometry"]
    world_geometry = execution_artifact["world_geometry"]
    training = execution_artifact["training"]
    arcs, oracle, model_init_seed = _build_seeded_pair(
        root_seed=execution_artifact["root_seed"],
        d_model=int(arm_geometry["d_model"]),
        hidden_size=int(arm_geometry["hidden_size"]),
        target_parameters=int(arm_geometry["target_parameters"]),
    )
    if int(model_init_seed) != int(execution_artifact.get("model_init_seed")):
        raise RuntimeError("EXP-277 model-init seed replay mismatch")

    pair_audit = audit_matched_exp277_arm_pair(
        arcs,
        oracle,
        timesteps=int(world_geometry["timesteps"]),
        variables=int(world_geometry["variables"]),
        constraints=int(world_geometry["constraints"]),
        max_accounted_flops_per_episode=int(execution_artifact["resource_match"]["declared_max_accounted_flops_per_episode"]),
    )
    if not pair_audit.get("oracle_information_separation") or not pair_audit.get("compute_budget_closed"):
        raise RuntimeError("EXP-277 checkpoint replay resource/oracle court failed")

    optimizer_contract = training["optimizer"]
    arcs_optimizer = build_functional_optimizer(
        arcs,
        lr=float(optimizer_contract["lr"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    oracle_optimizer = build_functional_optimizer(
        oracle,
        lr=float(optimizer_contract["lr"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    generator = Exp277StructureDenseGenerator(root_seed=execution_artifact["root_seed"])
    arcs_losses: list[float] = []
    oracle_losses: list[float] = []
    expected_batch_digests = training["paired_batch_digests"]
    for replicate in range(int(training["replicates"])):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=int(world_geometry["batch_size"]),
            timesteps=int(world_geometry["timesteps"]),
            variables=int(world_geometry["variables"]),
            constraints=int(world_geometry["constraints"]),
            d_model=int(world_geometry["d_model"]),
            noise_std=float(world_geometry["noise_std"]),
            rng_stream="augmentation",
        )
        if batch.digest != expected_batch_digests[replicate]:
            raise RuntimeError(f"EXP-277 training batch digest replay mismatch at replicate {replicate}")
        arcs_losses.append(
            _train_arcs(
                arcs,
                arcs_optimizer,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                targets=batch.targets,
            )
        )
        oracle_losses.append(
            _train_oracle(
                oracle,
                oracle_optimizer,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.oracle_incidence,
                targets=batch.targets,
            )
        )

    expected_arcs_losses = training["arcs_branch_losses"]
    expected_oracle_losses = training["oracle_cbrf_losses"]
    for idx, (actual, expected) in enumerate(zip(arcs_losses, expected_arcs_losses, strict=True)):
        if actual != float(expected):
            raise RuntimeError(f"EXP-277 ARCS training loss replay mismatch at replicate {idx}")
    for idx, (actual, expected) in enumerate(zip(oracle_losses, expected_oracle_losses, strict=True)):
        if actual != float(expected):
            raise RuntimeError(f"EXP-277 oracle training loss replay mismatch at replicate {idx}")

    final_state = execution_artifact["final_state"]
    arcs_digest = _functional_state_digest(arcs)
    oracle_digest = _functional_state_digest(oracle)
    if arcs_digest != final_state["arcs_branch_digest"] or oracle_digest != final_state["oracle_cbrf_digest"]:
        raise RuntimeError("EXP-277 replay final functional digest does not match DEVELOPMENT artifact")
    return arcs, oracle, arcs_losses, oracle_losses


def validate_exp277_checkpoint_receipt(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != RECEIPT_SCHEMA:
        errors.append("invalid EXP-277 checkpoint receipt schema")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-277 trained checkpoint must remain EV-E2 / UNVERIFIED")
    if payload.get("state_policy") != STATE_POLICY:
        errors.append("EXP-277 checkpoint state policy drift")
    if payload.get("training_replay_verified") is not True:
        errors.append("EXP-277 checkpoint training replay is not verified")
    contract = payload.get("execution_contract") or {}
    if payload.get("execution_contract_digest") != canonical_sha256(contract):
        errors.append("EXP-277 checkpoint execution contract digest mismatch")
    lineage = contract.get("training_lineage") or {}
    train_replicates = contract.get("train_replicates")
    if (
        not isinstance(train_replicates, int)
        or isinstance(train_replicates, bool)
        or train_replicates <= 0
        or lineage.get("rng_stream") != "augmentation"
        or lineage.get("start_replicate") != 0
        or lineage.get("replicates") != train_replicates
        or len(lineage.get("paired_batch_digests") or []) != train_replicates
    ):
        errors.append("EXP-277 checkpoint training lineage drift")
    optimizer = contract.get("optimizer") or {}
    if optimizer.get("type") != "AdamW" or not isinstance(optimizer.get("lr"), (int, float)) or float(optimizer.get("lr", 0.0)) <= 0.0:
        errors.append("EXP-277 checkpoint optimizer contract drift")
    if not isinstance(optimizer.get("weight_decay"), (int, float)) or float(optimizer.get("weight_decay", -1.0)) < 0.0:
        errors.append("EXP-277 checkpoint optimizer weight-decay drift")
    expected_final = contract.get("development_final_state") or {}
    if payload.get("arcs_branch_final_digest") != expected_final.get("arcs_branch_digest"):
        errors.append("EXP-277 ARCS checkpoint/final-state digest mismatch")
    if payload.get("oracle_cbrf_final_digest") != expected_final.get("oracle_cbrf_digest"):
        errors.append("EXP-277 oracle checkpoint/final-state digest mismatch")
    if not isinstance(payload.get("checkpoint_file_sha256"), str) or len(payload.get("checkpoint_file_sha256", "")) != 64:
        errors.append("EXP-277 checkpoint file SHA256 missing")
    expected_identity = _scientific_identity_digest(
        execution_contract_digest=str(payload.get("execution_contract_digest") or ""),
        arcs_branch_final_digest=str(payload.get("arcs_branch_final_digest") or ""),
        oracle_cbrf_final_digest=str(payload.get("oracle_cbrf_final_digest") or ""),
    )
    if payload.get("scientific_identity_digest") != expected_identity:
        errors.append("EXP-277 checkpoint scientific identity digest mismatch")
    if payload.get("receipt_digest") != _receipt_digest(payload):
        errors.append("EXP-277 checkpoint receipt digest mismatch")
    return errors


def build_exp277_trained_checkpoint(
    *,
    execution_artifact: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")
    arcs, oracle, _, _ = _replay_training(execution_artifact)
    execution_contract = _execution_contract(execution_artifact)
    execution_contract_digest = canonical_sha256(execution_contract)
    tensor_payload = {
        "schema": TENSOR_SCHEMA,
        "state_policy": STATE_POLICY,
        "execution_contract": execution_contract,
        "execution_contract_digest": execution_contract_digest,
        "arcs_branch_state": _functional_state(arcs),
        "oracle_cbrf_state": _functional_state(oracle),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_payload, checkpoint_path)
    arcs_digest = _functional_state_digest(arcs)
    oracle_digest = _functional_state_digest(oracle)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "experiment_id": "EXP-277",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "state_policy": STATE_POLICY,
        "training_replay_verified": True,
        "execution_contract": execution_contract,
        "execution_contract_digest": execution_contract_digest,
        "arcs_branch_final_digest": arcs_digest,
        "oracle_cbrf_final_digest": oracle_digest,
        "scientific_identity_digest": _scientific_identity_digest(
            execution_contract_digest=execution_contract_digest,
            arcs_branch_final_digest=arcs_digest,
            oracle_cbrf_final_digest=oracle_digest,
        ),
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    errors = validate_exp277_checkpoint_receipt(receipt)
    if errors:
        checkpoint_path.unlink(missing_ok=True)
        raise RuntimeError("invalid EXP-277 trained checkpoint receipt: " + "; ".join(errors))
    return receipt


def load_exp277_trained_checkpoint(
    *,
    checkpoint_path: str | Path,
    receipt: dict[str, Any],
) -> tuple[ARCSBranchArm, OracleCBRFArm]:
    checkpoint_path = Path(checkpoint_path)
    errors = validate_exp277_checkpoint_receipt(receipt)
    if errors:
        raise ValueError("invalid EXP-277 checkpoint receipt: " + "; ".join(errors))
    if not checkpoint_path.is_file():
        raise ValueError("EXP-277 checkpoint file is missing")
    if file_sha256(checkpoint_path) != receipt["checkpoint_file_sha256"]:
        raise ValueError("EXP-277 checkpoint file SHA256 mismatch")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("EXP-277 checkpoint file cannot be loaded safely") from exc
    if not isinstance(payload, dict) or payload.get("schema") != TENSOR_SCHEMA or payload.get("state_policy") != STATE_POLICY:
        raise ValueError("EXP-277 checkpoint tensor payload identity drift")
    if set(payload) != {
        "schema",
        "state_policy",
        "execution_contract",
        "execution_contract_digest",
        "arcs_branch_state",
        "oracle_cbrf_state",
    }:
        raise ValueError("EXP-277 checkpoint tensor payload surface drift")
    contract = receipt["execution_contract"]
    if payload.get("execution_contract") != contract or payload.get("execution_contract_digest") != receipt["execution_contract_digest"]:
        raise ValueError("EXP-277 checkpoint execution contract mismatch")
    if canonical_sha256(payload["execution_contract"]) != payload["execution_contract_digest"]:
        raise ValueError("EXP-277 checkpoint embedded execution contract digest mismatch")

    arm_geometry = contract["arm_geometry"]
    arcs, oracle, model_init_seed = _build_seeded_pair(
        root_seed=contract["root_seed"],
        d_model=int(arm_geometry["d_model"]),
        hidden_size=int(arm_geometry["hidden_size"]),
        target_parameters=int(arm_geometry["target_parameters"]),
    )
    if int(model_init_seed) != int(contract["model_init_seed"]):
        raise ValueError("EXP-277 checkpoint model-init seed mismatch")
    _load_functional_state(arcs, payload["arcs_branch_state"], arm_id="arcs_branch")
    _load_functional_state(oracle, payload["oracle_cbrf_state"], arm_id="oracle_cbrf")
    if _functional_state_digest(arcs) != receipt["arcs_branch_final_digest"]:
        raise ValueError("EXP-277 ARCS checkpoint functional digest mismatch")
    if _functional_state_digest(oracle) != receipt["oracle_cbrf_final_digest"]:
        raise ValueError("EXP-277 oracle checkpoint functional digest mismatch")

    world_geometry = contract["world_geometry"]
    audit = audit_matched_exp277_arm_pair(
        arcs,
        oracle,
        timesteps=int(world_geometry["timesteps"]),
        variables=int(world_geometry["variables"]),
        constraints=int(world_geometry["constraints"]),
        max_accounted_flops_per_episode=int(contract["declared_max_accounted_flops_per_episode"]),
    )
    if not audit.get("parameter_match") or not audit.get("functional_parameter_match") or not audit.get("oracle_information_separation") or not audit.get("compute_budget_closed"):
        raise ValueError("EXP-277 loaded checkpoint resource/oracle court failed")
    return arcs, oracle
