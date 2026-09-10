from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256
from nolane_ai.training.optimizer import functional_trainable_named_parameters
from .exp289_nogood_worlds import Exp289NogoodGenerator
from .exp289_paired_runner import (
    _artifact_digest,
    _build_seeded_pair,
    _functional_state_digest,
    _train_pair,
    validate_exp289_paired_development,
)
from .matched_nogood_arms import (
    LocalNogoodArm,
    NoNogoodArm,
    audit_matched_exp289_arm_pair,
)


RECEIPT_SCHEMA = "NLM-EXP-289-TRAINED-CHECKPOINT-V1"
TENSOR_SCHEMA = "NLM-EXP-289-TRAINED-TENSORS-V1"
CONTRACT_SCHEMA = "NLM-EXP-289-CHECKPOINT-EXECUTION-CONTRACT-V1"
IDENTITY_SCHEMA = "NLM-EXP-289-TRAINED-CHECKPOINT-SCIENTIFIC-IDENTITY-V1"
STATE_POLICY = "functional-only"
EXPERIMENT_ID = "EXP-289"


def _receipt_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("receipt_digest", None)
    return canonical_sha256(clean)


def _scientific_identity_digest(
    *,
    execution_contract_digest: str,
    no_nogood_final_digest: str,
    local_nogood_final_digest: str,
) -> str:
    return canonical_sha256(
        {
            "schema": IDENTITY_SCHEMA,
            "execution_contract_digest": execution_contract_digest,
            "no_nogood_final_digest": no_nogood_final_digest,
            "local_nogood_final_digest": local_nogood_final_digest,
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
            f"EXP-289 {arm_id} functional checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"EXP-289 {arm_id} checkpoint tensor {name} is not a tensor")
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(
                    f"EXP-289 {arm_id} checkpoint tensor {name} shape/dtype mismatch"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _scientific_execution(
    execution_artifact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Return the deterministic scientific payload behind an optional geometry envelope."""

    full_digest = execution_artifact.get("artifact_digest")
    authority = execution_artifact.get("development_geometry_authority")
    if authority is None:
        return execution_artifact, None, str(full_digest or "")
    if not isinstance(authority, dict):
        raise ValueError("EXP-289 DEVELOPMENT geometry authority is malformed")
    if authority.get("confirmatory_authority") is not False:
        raise ValueError("EXP-289 checkpoint cannot accept confirmatory geometry authority")
    if not isinstance(full_digest, str) or full_digest != _artifact_digest(execution_artifact):
        raise ValueError("EXP-289 authoritative DEVELOPMENT artifact self-hash mismatch")

    scientific = deepcopy(execution_artifact)
    scientific.pop("development_geometry_authority", None)
    scientific["artifact_digest"] = _artifact_digest(scientific)
    if authority.get("scientific_execution_digest") != scientific["artifact_digest"]:
        raise ValueError("EXP-289 DEVELOPMENT geometry scientific execution digest mismatch")
    return scientific, deepcopy(authority), full_digest


def _validate_execution_for_replay(execution_artifact: dict[str, Any]) -> dict[str, Any]:
    for field in (
        "confirmatory_data_consumed",
        "challenge_materialized",
        "challenge_seed_materialized",
        "decision_rule_executed",
    ):
        if execution_artifact.get(field) is not False:
            raise ValueError(
                "EXP-289 checkpoint cannot be reconstructed from confirmatory/challenge data"
            )
    if (
        execution_artifact.get("evidence_level") != "EV-E2"
        or execution_artifact.get("decision") != "UNVERIFIED"
    ):
        raise ValueError(
            "EXP-289 trained checkpoint requires EV-E2 / UNVERIFIED DEVELOPMENT evidence"
        )
    errors = validate_exp289_paired_development(execution_artifact)
    if errors:
        raise ValueError(
            "EXP-289 trained checkpoint requires valid paired DEVELOPMENT execution: "
            + "; ".join(errors)
        )
    return execution_artifact


def _replay_training(
    execution_artifact: dict[str, Any],
) -> tuple[NoNogoodArm, LocalNogoodArm, dict[str, Any]]:
    scientific, geometry_authority, authoritative_execution_digest = _scientific_execution(
        execution_artifact
    )
    _validate_execution_for_replay(scientific)
    config = scientific["configuration"]
    training = scientific["training"]
    resource = scientific["resource_match"]

    no_nogood, local_nogood, model_init_seed = _build_seeded_pair(
        root_seed=str(config["root_seed"]),
        d_model=int(config["d_model"]),
        hidden_size=int(config["hidden_size"]),
        target_parameters=int(config["target_parameters"]),
    )
    if int(model_init_seed) != int(scientific["model_init_seed"]):
        raise RuntimeError("EXP-289 model-init seed replay mismatch")
    initial = scientific.get("initial_state") or {}
    if _functional_state_digest(no_nogood) != initial.get("no_nogood_digest"):
        raise RuntimeError("EXP-289 no-nogood initial functional digest replay mismatch")
    if _functional_state_digest(local_nogood) != initial.get("local_nogood_digest"):
        raise RuntimeError("EXP-289 local-nogood initial functional digest replay mismatch")

    replay_audit = audit_matched_exp289_arm_pair(
        no_nogood,
        local_nogood,
        restarts=int(config["restarts"]),
        variables=int(config["variables"]),
        max_search_steps=int(config["max_search_steps"]),
        max_accounted_cost_per_episode=int(
            resource["declared_max_accounted_cost_per_episode"]
        ),
    )
    if replay_audit != resource.get("pair_audit"):
        raise RuntimeError("EXP-289 checkpoint replay pair-audit mismatch")
    if (
        replay_audit.get("compute_budget_closed") is not True
        or replay_audit.get("memory_scope_closed") is not True
    ):
        raise RuntimeError("EXP-289 checkpoint replay resource/scope court failed")

    optimizer = training.get("optimizer_hyperparameters") or {}
    replay_training = _train_pair(
        no_nogood=no_nogood,
        local_nogood=local_nogood,
        generator=Exp289NogoodGenerator(root_seed=str(config["root_seed"])),
        train_replicates=int(config["train_replicates"]),
        batch_size=int(config["batch_size"]),
        timesteps=int(config["timesteps"]),
        restarts=int(config["restarts"]),
        variables=int(config["variables"]),
        decoys=int(config["decoys"]),
        d_model=int(config["d_model"]),
        noise_std=float(config["noise_std"]),
        lr=float(optimizer["lr"]),
        weight_decay=float(optimizer["weight_decay"]),
    )
    if replay_training != training:
        raise RuntimeError("EXP-289 exact training-lineage replay mismatch")

    no_digest = _functional_state_digest(no_nogood)
    local_digest = _functional_state_digest(local_nogood)
    if no_digest != training.get("post_training_no_nogood_digest"):
        raise RuntimeError("EXP-289 no-nogood replay final functional digest mismatch")
    if local_digest != training.get("post_training_local_nogood_digest"):
        raise RuntimeError("EXP-289 local-nogood replay final functional digest mismatch")

    no_nogood.eval()
    local_nogood.eval()
    replay_binding = {
        "scientific_execution": scientific,
        "development_geometry_authority": geometry_authority,
        "authoritative_execution_digest": authoritative_execution_digest,
    }
    return no_nogood, local_nogood, replay_binding


def _execution_contract(
    scientific: dict[str, Any],
    *,
    geometry_authority: dict[str, Any] | None,
    authoritative_execution_digest: str,
) -> dict[str, Any]:
    training = scientific.get("training") or {}
    resource = scientific.get("resource_match") or {}
    return {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_digest": scientific.get("protocol_digest"),
        "development_code_digest": scientific.get("code_digest"),
        "scientific_execution_digest": scientific.get("artifact_digest"),
        "authoritative_execution_digest": authoritative_execution_digest,
        "development_geometry_authority": deepcopy(geometry_authority),
        "model_init_seed": scientific.get("model_init_seed"),
        "configuration": deepcopy(scientific.get("configuration") or {}),
        "training_lineage": {
            "rng_stream": training.get("rng_stream"),
            "start_replicate": training.get("start_replicate"),
            "replicates": training.get("replicates"),
            "paired_batch_digests": deepcopy(training.get("paired_batch_digests") or []),
            "per_replicate": deepcopy(training.get("per_replicate") or []),
            "optimizer_family": training.get("optimizer_family"),
            "optimizer_hyperparameters": deepcopy(
                training.get("optimizer_hyperparameters") or {}
            ),
            "post_training_no_nogood_digest": training.get(
                "post_training_no_nogood_digest"
            ),
            "post_training_local_nogood_digest": training.get(
                "post_training_local_nogood_digest"
            ),
            "post_training_functional_digest_match": training.get(
                "post_training_functional_digest_match"
            ),
        },
        "resource_binding": {
            "pair_audit_digest": resource.get("pair_audit_digest"),
            "parameter_match": resource.get("parameter_match"),
            "functional_parameter_match": resource.get("functional_parameter_match"),
            "active_functional_parameter_match": resource.get(
                "active_functional_parameter_match"
            ),
            "optimizer_visible_parameter_match": resource.get(
                "optimizer_visible_parameter_match"
            ),
            "memory_scope_closed": resource.get("memory_scope_closed"),
            "compute_budget_closed": resource.get("compute_budget_closed"),
            "declared_max_accounted_cost_per_episode": resource.get(
                "declared_max_accounted_cost_per_episode"
            ),
        },
        "scope_policy": deepcopy(scientific.get("scope_policy") or {}),
    }


def _is_hex_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_exp289_checkpoint_receipt(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != RECEIPT_SCHEMA:
        errors.append("invalid EXP-289 checkpoint receipt schema")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-289 checkpoint experiment identity drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-289 trained checkpoint must remain EV-E2 / UNVERIFIED")
    if payload.get("state_policy") != STATE_POLICY:
        errors.append("EXP-289 checkpoint state policy drift")
    if payload.get("training_replay_verified") is not True:
        errors.append("EXP-289 checkpoint training replay is not verified")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-289 checkpoint cannot consume confirmatory data")
    if payload.get("challenge_materialized") is not False:
        errors.append("EXP-289 checkpoint cannot materialize challenge data")
    if payload.get("challenge_seed_materialized") is not False:
        errors.append("EXP-289 checkpoint cannot materialize challenge seed")

    contract = payload.get("execution_contract") or {}
    if contract.get("schema") != CONTRACT_SCHEMA:
        errors.append("EXP-289 checkpoint execution contract schema drift")
    if contract.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-289 checkpoint execution contract identity drift")
    expected_contract_digest = canonical_sha256(contract)
    if payload.get("execution_contract_digest") != expected_contract_digest:
        errors.append("EXP-289 checkpoint execution contract digest mismatch")

    lineage = contract.get("training_lineage") or {}
    replicate_count = lineage.get("replicates")
    paired_digests = list(lineage.get("paired_batch_digests") or [])
    rows = list(lineage.get("per_replicate") or [])
    if (
        not isinstance(replicate_count, int)
        or isinstance(replicate_count, bool)
        or replicate_count <= 0
        or lineage.get("rng_stream") != "augmentation"
        or lineage.get("start_replicate") != 0
        or len(paired_digests) != replicate_count
        or len(rows) != replicate_count
        or lineage.get("optimizer_family") != "AdamW"
    ):
        errors.append("EXP-289 checkpoint training lineage drift")
    else:
        for index, row in enumerate(rows):
            if (
                row.get("replicate") != index
                or row.get("paired_batch_digest") != paired_digests[index]
                or row.get("memory_hit_training_signal") is not False
                or row.get("evaluator_metadata_delivered_to_arm") is not False
            ):
                errors.append("EXP-289 checkpoint per-replicate training lineage drift")
                break
    optimizer = lineage.get("optimizer_hyperparameters") or {}
    if (
        not isinstance(optimizer.get("lr"), (int, float))
        or isinstance(optimizer.get("lr"), bool)
        or float(optimizer.get("lr", 0.0)) <= 0.0
        or not isinstance(optimizer.get("weight_decay"), (int, float))
        or isinstance(optimizer.get("weight_decay"), bool)
        or float(optimizer.get("weight_decay", -1.0)) < 0.0
    ):
        errors.append("EXP-289 checkpoint optimizer contract drift")

    resource = contract.get("resource_binding") or {}
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "memory_scope_closed",
        "compute_budget_closed",
    ):
        if resource.get(key) is not True:
            errors.append(f"EXP-289 checkpoint resource binding drift: {key}")
    if (
        not isinstance(resource.get("declared_max_accounted_cost_per_episode"), int)
        or isinstance(resource.get("declared_max_accounted_cost_per_episode"), bool)
        or int(resource.get("declared_max_accounted_cost_per_episode", 0)) <= 0
        or not _is_hex_digest(resource.get("pair_audit_digest"))
    ):
        errors.append("EXP-289 checkpoint resource cost/audit binding drift")

    no_digest = payload.get("no_nogood_final_digest")
    local_digest = payload.get("local_nogood_final_digest")
    if no_digest != lineage.get("post_training_no_nogood_digest"):
        errors.append("EXP-289 no-nogood checkpoint/final-state digest mismatch")
    if local_digest != lineage.get("post_training_local_nogood_digest"):
        errors.append("EXP-289 local-nogood checkpoint/final-state digest mismatch")
    if not _is_hex_digest(no_digest):
        errors.append("EXP-289 no-nogood checkpoint final digest invalid")
    if not _is_hex_digest(local_digest):
        errors.append("EXP-289 local-nogood checkpoint final digest invalid")
    if not _is_hex_digest(payload.get("checkpoint_file_sha256")):
        errors.append("EXP-289 checkpoint file SHA256 invalid")

    expected_identity = _scientific_identity_digest(
        execution_contract_digest=str(payload.get("execution_contract_digest") or ""),
        no_nogood_final_digest=str(no_digest or ""),
        local_nogood_final_digest=str(local_digest or ""),
    )
    if payload.get("scientific_identity_digest") != expected_identity:
        errors.append("EXP-289 checkpoint scientific identity digest mismatch")
    if payload.get("receipt_digest") != _receipt_digest(payload):
        errors.append("EXP-289 checkpoint receipt digest mismatch")
    return errors


def _publish_checkpoint_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"checkpoint already exists: {path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def build_exp289_trained_checkpoint(
    *,
    execution_artifact: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")

    no_nogood, local_nogood, replay = _replay_training(execution_artifact)
    scientific = replay["scientific_execution"]
    execution_contract = _execution_contract(
        scientific,
        geometry_authority=replay["development_geometry_authority"],
        authoritative_execution_digest=str(replay["authoritative_execution_digest"]),
    )
    execution_contract_digest = canonical_sha256(execution_contract)
    no_digest = _functional_state_digest(no_nogood)
    local_digest = _functional_state_digest(local_nogood)

    tensor_payload = {
        "schema": TENSOR_SCHEMA,
        "state_policy": STATE_POLICY,
        "execution_contract": execution_contract,
        "execution_contract_digest": execution_contract_digest,
        "no_nogood_state": _functional_state(no_nogood),
        "local_nogood_state": _functional_state(local_nogood),
    }
    _publish_checkpoint_exclusive(checkpoint_path, tensor_payload)
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
        "no_nogood_final_digest": no_digest,
        "local_nogood_final_digest": local_digest,
        "checkpoint_file_sha256": checkpoint_sha,
        "scientific_identity_digest": _scientific_identity_digest(
            execution_contract_digest=execution_contract_digest,
            no_nogood_final_digest=no_digest,
            local_nogood_final_digest=local_digest,
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    errors = validate_exp289_checkpoint_receipt(receipt)
    if errors:
        checkpoint_path.unlink(missing_ok=True)
        raise RuntimeError(
            "invalid EXP-289 trained checkpoint receipt before return: "
            + "; ".join(errors)
        )
    return receipt


def load_exp289_trained_checkpoint(
    *,
    checkpoint_path: str | Path,
    receipt: dict[str, Any],
) -> tuple[NoNogoodArm, LocalNogoodArm]:
    errors = validate_exp289_checkpoint_receipt(receipt)
    if errors:
        raise ValueError("invalid EXP-289 checkpoint receipt: " + "; ".join(errors))

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file is missing: {checkpoint_path}")
    if file_sha256(checkpoint_path) != receipt["checkpoint_file_sha256"]:
        raise ValueError("EXP-289 checkpoint file SHA256 mismatch")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("EXP-289 checkpoint tensor payload is not a mapping")
    if payload.get("schema") != TENSOR_SCHEMA:
        raise ValueError("EXP-289 checkpoint tensor schema drift")
    if payload.get("state_policy") != STATE_POLICY:
        raise ValueError("EXP-289 checkpoint tensor state policy drift")
    if payload.get("execution_contract_digest") != receipt["execution_contract_digest"]:
        raise ValueError("EXP-289 checkpoint tensor contract digest mismatch")
    if payload.get("execution_contract") != receipt["execution_contract"]:
        raise ValueError("EXP-289 checkpoint tensor execution contract mismatch")

    config = receipt["execution_contract"]["configuration"]
    no_nogood, local_nogood, model_init_seed = _build_seeded_pair(
        root_seed=str(config["root_seed"]),
        d_model=int(config["d_model"]),
        hidden_size=int(config["hidden_size"]),
        target_parameters=int(config["target_parameters"]),
    )
    if int(model_init_seed) != int(receipt["execution_contract"]["model_init_seed"]):
        raise ValueError("EXP-289 checkpoint model-init identity mismatch")

    no_state = payload.get("no_nogood_state")
    local_state = payload.get("local_nogood_state")
    if not isinstance(no_state, dict) or not isinstance(local_state, dict):
        raise ValueError("EXP-289 checkpoint functional state payload missing")
    _load_functional_state(no_nogood, no_state, arm_id="no_nogood")
    _load_functional_state(local_nogood, local_state, arm_id="local_nogood")

    if _functional_state_digest(no_nogood) != receipt["no_nogood_final_digest"]:
        raise ValueError("EXP-289 no-nogood loaded functional digest mismatch")
    if _functional_state_digest(local_nogood) != receipt["local_nogood_final_digest"]:
        raise ValueError("EXP-289 local-nogood loaded functional digest mismatch")

    no_nogood.eval()
    local_nogood.eval()
    return no_nogood, local_nogood
