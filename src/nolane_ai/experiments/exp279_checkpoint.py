from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256
from nolane_ai.training.optimizer import (
    build_functional_optimizer,
    functional_trainable_named_parameters,
)
from .exp279_paired_runner import (
    ARM_ORDER,
    _build_seeded_triplet,
    _functional_state_digest,
    _replay_contract,
    _train_step,
    validate_exp279_paired_development,
)
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import (
    BranchOnlyArm,
    HybridRoutingArm,
    PropagationOnlyArm,
    audit_matched_exp279_arm_triplet,
)

RECEIPT_SCHEMA = "NLM-EXP-279-TRAINED-CHECKPOINT-V1"
TENSOR_SCHEMA = "NLM-EXP-279-TRAINED-TENSORS-V1"
STATE_POLICY = "functional-only"
EXPERIMENT_ID = "EXP-279"


def _receipt_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("receipt_digest", None)
    return canonical_sha256(clean)


def _scientific_identity_digest(
    *,
    replay_contract_digest: str,
    final_state_digest: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "NLM-EXP-279-TRAINED-CHECKPOINT-SCIENTIFIC-IDENTITY-V1",
            "replay_contract_digest": replay_contract_digest,
            "final_state_digest": final_state_digest,
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
            f"EXP-279 {arm_id} functional checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"EXP-279 {arm_id} checkpoint tensor {name} is not a tensor"
                )
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(
                    f"EXP-279 {arm_id} checkpoint tensor {name} shape/dtype mismatch"
                )
            parameter.copy_(
                tensor.to(device=parameter.device, dtype=parameter.dtype)
            )


def _validate_replay_contract_semantics(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != "NLM-EXP-279-DEVELOPMENT-REPLAY-CONTRACT-V1":
        errors.append("EXP-279 checkpoint replay contract schema drift")

    arm_geometry = contract.get("arm_geometry") or {}
    world_geometry = contract.get("world_geometry") or {}
    route_config = contract.get("route_config") or {}
    for key in ("d_model", "hidden_size", "target_parameters"):
        value = arm_geometry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"EXP-279 checkpoint replay arm geometry {key} invalid")
    for key in ("batch_size", "timesteps", "variables", "constraints", "d_model"):
        value = world_geometry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"EXP-279 checkpoint replay world geometry {key} invalid")
    if arm_geometry.get("d_model") != world_geometry.get("d_model"):
        errors.append("EXP-279 checkpoint replay d_model geometry mismatch")

    try:
        arm_threshold = float(arm_geometry.get("route_threshold"))
        route_threshold = float(route_config.get("threshold"))
    except (TypeError, ValueError):
        arm_threshold = -1.0
        route_threshold = -2.0
    if not 0.0 <= arm_threshold <= 1.0 or not 0.0 <= route_threshold <= 1.0:
        errors.append("EXP-279 checkpoint replay route threshold invalid")
    elif abs(arm_threshold - route_threshold) > 1e-12:
        errors.append("EXP-279 checkpoint replay route threshold binding mismatch")
    if route_config.get("strategy") != "propagation_then_branch_on_residual_uncertainty":
        errors.append("EXP-279 checkpoint replay routing strategy drift")
    if route_config.get("frozen_before_evaluation") is not True:
        errors.append("EXP-279 checkpoint replay route threshold not frozen")

    training = contract.get("training") or {}
    if training.get("rng_stream") != "augmentation" or training.get("start_replicate") != 0:
        errors.append("EXP-279 checkpoint replay training lineage drift")
    if training.get("strata") != list(STRATA):
        errors.append("EXP-279 checkpoint replay training strata drift")
    train_count = training.get("replicates")
    if not isinstance(train_count, int) or isinstance(train_count, bool) or train_count <= 0:
        errors.append("EXP-279 checkpoint replay training replicate count invalid")
        train_count = 0
    batch_digests = list(training.get("paired_batch_digests") or [])
    if len(batch_digests) != train_count or len(set(batch_digests)) != len(batch_digests):
        errors.append("EXP-279 checkpoint replay training batch lineage invalid")

    optimizer = training.get("optimizer") or {}
    if optimizer.get("type") != "AdamW":
        errors.append("EXP-279 checkpoint replay optimizer type drift")
    try:
        lr = float(optimizer.get("lr"))
        weight_decay = float(optimizer.get("weight_decay"))
    except (TypeError, ValueError):
        lr = -1.0
        weight_decay = -1.0
    if lr <= 0.0 or weight_decay < 0.0:
        errors.append("EXP-279 checkpoint replay optimizer contract invalid")

    losses = training.get("losses") or {}
    if set(losses) != set(ARM_ORDER):
        errors.append("EXP-279 checkpoint replay training loss arm set mismatch")
    else:
        for arm_id in ARM_ORDER:
            values = list(losses.get(arm_id) or [])
            if len(values) != train_count:
                errors.append(
                    f"EXP-279 checkpoint replay {arm_id} training loss count mismatch"
                )

    final_state = contract.get("final_state") or {}
    expected_final_keys = {
        "propagation_only_digest",
        "branch_only_digest",
        "hybrid_digest",
    }
    if set(final_state) != expected_final_keys:
        errors.append("EXP-279 checkpoint replay final state surface drift")
    if not contract.get("resource_pair_audit_digest"):
        errors.append("EXP-279 checkpoint replay resource pair-audit digest missing")
    ceiling = contract.get("declared_max_accounted_flops_per_episode")
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
        errors.append("EXP-279 checkpoint replay compute ceiling invalid")
    if not isinstance(contract.get("root_seed"), str) or not contract.get("root_seed"):
        errors.append("EXP-279 checkpoint replay root seed missing")
    if not isinstance(contract.get("model_init_seed"), int) or isinstance(
        contract.get("model_init_seed"), bool
    ):
        errors.append("EXP-279 checkpoint replay model-init seed invalid")
    if not isinstance(contract.get("protocol_digest"), str) or not contract.get(
        "protocol_digest"
    ):
        errors.append("EXP-279 checkpoint replay protocol digest missing")
    if not isinstance(contract.get("development_code_digest"), str) or not contract.get(
        "development_code_digest"
    ):
        errors.append("EXP-279 checkpoint replay development code digest missing")
    return errors


def _validate_execution_for_replay(execution_artifact: dict[str, Any]) -> None:
    errors = validate_exp279_paired_development(execution_artifact)
    if errors:
        raise ValueError(
            "invalid EXP-279 DEVELOPMENT artifact for checkpoint replay: "
            + "; ".join(errors)
        )
    if execution_artifact.get("schema") != "NLM-EXP-279-PAIRED-DEV-EVAL-V1":
        raise ValueError("EXP-279 trained checkpoint requires paired DEVELOPMENT execution")
    if (
        execution_artifact.get("evidence_level") != "EV-E2"
        or execution_artifact.get("decision") != "UNVERIFIED"
    ):
        raise ValueError(
            "EXP-279 trained checkpoint requires EV-E2 / UNVERIFIED DEVELOPMENT evidence"
        )
    if (
        execution_artifact.get("confirmatory_data_consumed") is not False
        or execution_artifact.get("challenge_materialized") is not False
        or execution_artifact.get("decision_rule_executed") is not False
    ):
        raise ValueError(
            "EXP-279 checkpoint cannot be reconstructed from confirmatory/challenge data"
        )
    contract = _replay_contract(execution_artifact)
    contract_errors = _validate_replay_contract_semantics(contract)
    if contract_errors:
        raise ValueError(
            "invalid EXP-279 replay contract: " + "; ".join(contract_errors)
        )
    if execution_artifact.get("replay_contract_digest") != canonical_sha256(contract):
        raise ValueError("EXP-279 DEVELOPMENT replay contract digest mismatch")


def _replay_training(
    execution_artifact: dict[str, Any],
) -> tuple[PropagationOnlyArm, BranchOnlyArm, HybridRoutingArm]:
    _validate_execution_for_replay(execution_artifact)
    contract = _replay_contract(execution_artifact)
    arm_geometry = contract["arm_geometry"]
    world_geometry = contract["world_geometry"]
    training = contract["training"]

    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=str(contract["root_seed"]),
        d_model=int(arm_geometry["d_model"]),
        hidden_size=int(arm_geometry["hidden_size"]),
        target_parameters=int(arm_geometry["target_parameters"]),
        route_threshold=float(arm_geometry["route_threshold"]),
    )
    if int(model_init_seed) != int(contract["model_init_seed"]):
        raise RuntimeError("EXP-279 model-init seed replay mismatch")

    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=int(world_geometry["timesteps"]),
        variables=int(world_geometry["variables"]),
        constraints=int(world_geometry["constraints"]),
        max_accounted_flops_per_episode=int(
            contract["declared_max_accounted_flops_per_episode"]
        ),
    )
    pair_audit_digest = canonical_sha256(pair_audit)
    if pair_audit_digest != contract["resource_pair_audit_digest"]:
        raise RuntimeError("EXP-279 checkpoint replay resource pair-audit digest mismatch")
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "compute_budget_closed",
    ):
        if pair_audit.get(key) is not True:
            raise RuntimeError(f"EXP-279 checkpoint replay resource court open: {key}")
    if pair_audit.get("structure_fit_strata") != list(STRATA):
        raise RuntimeError("EXP-279 checkpoint replay structure-fit strata drift")

    optimizer_contract = training["optimizer"]
    arms: dict[str, PropagationOnlyArm | BranchOnlyArm | HybridRoutingArm] = {
        "propagation_only": propagation,
        "branch_only": branch,
        "hybrid": hybrid,
    }
    optimizers = {
        arm_id: build_functional_optimizer(
            arm,
            lr=float(optimizer_contract["lr"]),
            weight_decay=float(optimizer_contract["weight_decay"]),
        )
        for arm_id, arm in arms.items()
    }
    generator = Exp279RoutingGenerator(root_seed=str(contract["root_seed"]))
    expected_batch_digests = list(training["paired_batch_digests"])
    expected_losses = training["losses"]

    for replicate in range(int(training["replicates"])):
        stratum = STRATA[replicate % len(STRATA)]
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=int(world_geometry["batch_size"]),
            timesteps=int(world_geometry["timesteps"]),
            variables=int(world_geometry["variables"]),
            constraints=int(world_geometry["constraints"]),
            d_model=int(world_geometry["d_model"]),
            noise_std=float(world_geometry["noise_std"]),
            rng_stream="augmentation",
            stratum=stratum,
        )
        if batch.digest != expected_batch_digests[replicate]:
            raise RuntimeError(
                f"EXP-279 training batch digest replay mismatch at replicate {replicate}"
            )
        for arm_id in ARM_ORDER:
            actual_loss = _train_step(
                arms[arm_id],
                optimizers[arm_id],
                arm_id=arm_id,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
                targets=batch.targets,
            )
            expected_loss = float(expected_losses[arm_id][replicate])
            if actual_loss != expected_loss:
                raise RuntimeError(
                    f"EXP-279 {arm_id} training loss replay mismatch at replicate {replicate}"
                )

    final_state = {
        "propagation_only_digest": _functional_state_digest(propagation),
        "branch_only_digest": _functional_state_digest(branch),
        "hybrid_digest": _functional_state_digest(hybrid),
    }
    if final_state != contract["final_state"]:
        raise RuntimeError(
            "EXP-279 replay final functional digest does not match DEVELOPMENT final state"
        )
    if canonical_sha256(final_state) != execution_artifact.get("final_state_digest"):
        raise RuntimeError("EXP-279 replay final state digest mismatch")
    return propagation, branch, hybrid


def validate_exp279_checkpoint_receipt(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != RECEIPT_SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 checkpoint receipt identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 trained checkpoint must remain EV-E2 / UNVERIFIED")
    if payload.get("state_policy") != STATE_POLICY:
        errors.append("EXP-279 checkpoint state policy drift")
    if payload.get("training_replay_verified") is not True:
        errors.append("EXP-279 checkpoint training replay is not verified")
    for key in (
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(key) is not False:
            errors.append(f"EXP-279 checkpoint forbidden flag enabled: {key}")

    contract = payload.get("replay_contract") or {}
    errors.extend(_validate_replay_contract_semantics(contract))
    expected_contract_digest = canonical_sha256(contract)
    if payload.get("replay_contract_digest") != expected_contract_digest:
        errors.append("EXP-279 checkpoint replay contract digest mismatch")

    final_state = payload.get("final_state") or {}
    expected_final = contract.get("final_state") or {}
    if final_state != expected_final:
        errors.append("EXP-279 checkpoint final state/replay contract mismatch")
    expected_final_digest = canonical_sha256(final_state)
    if payload.get("final_state_digest") != expected_final_digest:
        errors.append("EXP-279 checkpoint final state digest mismatch")

    if not isinstance(payload.get("checkpoint_file_sha256"), str) or len(
        payload.get("checkpoint_file_sha256", "")
    ) != 64:
        errors.append("EXP-279 checkpoint file SHA256 missing")

    expected_identity = _scientific_identity_digest(
        replay_contract_digest=str(payload.get("replay_contract_digest") or ""),
        final_state_digest=str(payload.get("final_state_digest") or ""),
    )
    if payload.get("scientific_identity_digest") != expected_identity:
        errors.append("EXP-279 checkpoint scientific identity digest mismatch")
    if payload.get("receipt_digest") != _receipt_digest(payload):
        errors.append("EXP-279 checkpoint receipt digest mismatch")
    return errors


def build_exp279_trained_checkpoint(
    *,
    execution_artifact: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")

    propagation, branch, hybrid = _replay_training(execution_artifact)
    replay_contract = _replay_contract(execution_artifact)
    replay_contract_digest = canonical_sha256(replay_contract)
    if replay_contract_digest != execution_artifact.get("replay_contract_digest"):
        raise RuntimeError("EXP-279 checkpoint replay contract binding mismatch")

    tensor_payload = {
        "schema": TENSOR_SCHEMA,
        "state_policy": STATE_POLICY,
        "replay_contract": replay_contract,
        "replay_contract_digest": replay_contract_digest,
        "propagation_only_state": _functional_state(propagation),
        "branch_only_state": _functional_state(branch),
        "hybrid_state": _functional_state(hybrid),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_payload, checkpoint_path)

    final_state = {
        "propagation_only_digest": _functional_state_digest(propagation),
        "branch_only_digest": _functional_state_digest(branch),
        "hybrid_digest": _functional_state_digest(hybrid),
    }
    final_state_digest = canonical_sha256(final_state)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "state_policy": STATE_POLICY,
        "training_replay_verified": True,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "replay_contract": replay_contract,
        "replay_contract_digest": replay_contract_digest,
        "final_state": final_state,
        "final_state_digest": final_state_digest,
        "scientific_identity_digest": _scientific_identity_digest(
            replay_contract_digest=replay_contract_digest,
            final_state_digest=final_state_digest,
        ),
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    errors = validate_exp279_checkpoint_receipt(receipt)
    if errors:
        checkpoint_path.unlink(missing_ok=True)
        raise RuntimeError(
            "invalid EXP-279 trained checkpoint receipt: " + "; ".join(errors)
        )
    return receipt


def load_exp279_trained_checkpoint(
    *,
    checkpoint_path: str | Path,
    receipt: dict[str, Any],
) -> tuple[PropagationOnlyArm, BranchOnlyArm, HybridRoutingArm]:
    checkpoint_path = Path(checkpoint_path)
    errors = validate_exp279_checkpoint_receipt(receipt)
    if errors:
        raise ValueError("invalid EXP-279 checkpoint receipt: " + "; ".join(errors))
    if not checkpoint_path.is_file():
        raise ValueError("EXP-279 checkpoint file is missing")
    if file_sha256(checkpoint_path) != receipt["checkpoint_file_sha256"]:
        raise ValueError("EXP-279 checkpoint file SHA256 mismatch")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("EXP-279 checkpoint file cannot be loaded safely") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != TENSOR_SCHEMA
        or payload.get("state_policy") != STATE_POLICY
    ):
        raise ValueError("EXP-279 checkpoint tensor payload identity drift")
    if set(payload) != {
        "schema",
        "state_policy",
        "replay_contract",
        "replay_contract_digest",
        "propagation_only_state",
        "branch_only_state",
        "hybrid_state",
    }:
        raise ValueError("EXP-279 checkpoint tensor payload surface drift")

    contract = receipt["replay_contract"]
    if payload.get("replay_contract") != contract:
        raise ValueError("EXP-279 checkpoint embedded replay contract mismatch")
    if payload.get("replay_contract_digest") != receipt["replay_contract_digest"]:
        raise ValueError("EXP-279 checkpoint embedded replay contract digest mismatch")
    if canonical_sha256(payload["replay_contract"]) != payload["replay_contract_digest"]:
        raise ValueError("EXP-279 checkpoint embedded replay contract digest invalid")

    arm_geometry = contract["arm_geometry"]
    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=str(contract["root_seed"]),
        d_model=int(arm_geometry["d_model"]),
        hidden_size=int(arm_geometry["hidden_size"]),
        target_parameters=int(arm_geometry["target_parameters"]),
        route_threshold=float(arm_geometry["route_threshold"]),
    )
    if int(model_init_seed) != int(contract["model_init_seed"]):
        raise ValueError("EXP-279 checkpoint model-init seed mismatch")

    _load_functional_state(
        propagation,
        payload["propagation_only_state"],
        arm_id="propagation_only",
    )
    _load_functional_state(branch, payload["branch_only_state"], arm_id="branch_only")
    _load_functional_state(hybrid, payload["hybrid_state"], arm_id="hybrid")

    loaded_digests = {
        "propagation_only_digest": _functional_state_digest(propagation),
        "branch_only_digest": _functional_state_digest(branch),
        "hybrid_digest": _functional_state_digest(hybrid),
    }
    if loaded_digests != receipt["final_state"]:
        raise ValueError("EXP-279 checkpoint functional digest mismatch")
    if canonical_sha256(loaded_digests) != receipt["final_state_digest"]:
        raise ValueError("EXP-279 checkpoint loaded final state digest mismatch")

    world_geometry = contract["world_geometry"]
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=int(world_geometry["timesteps"]),
        variables=int(world_geometry["variables"]),
        constraints=int(world_geometry["constraints"]),
        max_accounted_flops_per_episode=int(
            contract["declared_max_accounted_flops_per_episode"]
        ),
    )
    if canonical_sha256(pair_audit) != contract["resource_pair_audit_digest"]:
        raise ValueError("EXP-279 loaded checkpoint resource pair-audit digest mismatch")
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "compute_budget_closed",
    ):
        if pair_audit.get(key) is not True:
            raise ValueError(f"EXP-279 loaded checkpoint resource court failed: {key}")
    if pair_audit.get("structure_fit_strata") != list(STRATA):
        raise ValueError("EXP-279 loaded checkpoint structure-fit strata drift")
    return propagation, branch, hybrid
