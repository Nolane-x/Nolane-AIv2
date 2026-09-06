from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes
from .exp277_structure_dense import Exp277StructureDenseGenerator
from .matched_cbrf_arms import (
    ARCSBranchArm,
    OracleCBRFArm,
    audit_matched_exp277_arm_pair,
    build_matched_exp277_arm_pair,
)

SCHEMA = "NLM-EXP-277-PAIRED-DEV-EVAL-V1"


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _functional_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-277-FUNCTIONAL-STATE-V1\0")
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
) -> tuple[ARCSBranchArm, OracleCBRFArm, int]:
    seed = derive_stream_seed(root_seed, "EXP-277", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        arcs, oracle = build_matched_exp277_arm_pair(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            device="cpu",
        )
    return arcs, oracle, seed


def _train_arcs(
    model: ARCSBranchArm,
    optimizer: torch.optim.Optimizer,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(surface_events, variable_states)
    loss = F.cross_entropy(output.decision_logits.reshape(-1, 2), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _train_oracle(
    model: OracleCBRFArm,
    optimizer: torch.optim.Optimizer,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(surface_events, variable_states, incidence)
    loss = F.cross_entropy(output.decision_logits.reshape(-1, 2), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _external_metrics(output: Any, targets: torch.Tensor, *, accounted_flops: int) -> dict[str, float | int]:
    predictions = output.decision_logits.argmax(dim=-1)
    exact_per_episode = (predictions == targets).all(dim=-1)
    verified_solution_rate = float(exact_per_episode.to(torch.float32).mean().item())
    verified_decision_accuracy = float((predictions == targets).to(torch.float32).mean().item())
    utility = verified_solution_rate / max(int(accounted_flops), 1)
    return {
        "verified_solution_rate": verified_solution_rate,
        "verified_decision_accuracy": verified_decision_accuracy,
        "mean_verifier_confidence": float(output.verifier_confidence.mean().item()),
        "accounted_flops_per_episode": int(accounted_flops),
        "verified_utility_per_accounted_flop": utility,
    }


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return sum(float(row[arm][metric]) for row in rows) / len(rows)


def validate_exp277_paired_development(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-277 paired development schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-277 paired development artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-277 paired development artifact cannot promote a claim")
    if payload.get("confirmatory_ready") is not False:
        errors.append("EXP-277 paired development artifact cannot be confirmatory-ready")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-277 development cannot consume confirmatory data")
    if payload.get("challenge_materialized") is not False:
        errors.append("EXP-277 development cannot materialize challenge randomness")
    if payload.get("decision_rule_executed") is not False:
        errors.append("EXP-277 development cannot execute the scientific promotion rule")
    if not payload.get("protocol_digest") or not payload.get("code_digest"):
        errors.append("EXP-277 paired development provenance is incomplete")

    initial = payload.get("initial_state") or {}
    if initial.get("functional_digest_match") is not True or initial.get("arcs_branch_digest") != initial.get("oracle_cbrf_digest"):
        errors.append("EXP-277 matched arms must share identical functional initialization")

    resource = payload.get("resource_match") or {}
    if resource.get("parameter_match") is not True:
        errors.append("EXP-277 parameter match must be closed")
    if resource.get("functional_parameter_match") is not True:
        errors.append("EXP-277 functional parameter match must be closed")
    if resource.get("same_world_lineage") is not True:
        errors.append("EXP-277 paired arms must share world lineage")
    if resource.get("compute_budget_closed") is not True:
        errors.append("EXP-277 compute budget must be closed")
    pair_audit = resource.get("pair_audit") or {}
    if pair_audit.get("schema") != "NLM-EXP-277-MATCHED-ARMS-DEV-V1":
        errors.append("EXP-277 matched pair audit schema drift")
    if resource.get("pair_audit_digest") != canonical_sha256(pair_audit):
        errors.append("EXP-277 pair audit digest mismatch")
    declared_budget = int(resource.get("declared_max_accounted_flops_per_episode", 0) or 0)
    if declared_budget <= 0:
        errors.append("EXP-277 declared compute ceiling is missing")

    expected_receipt = {
        "artifact": "oracle_incidence",
        "ground_truth": True,
        "delivered_to": ["oracle_cbrf"],
        "withheld_from": ["arcs_branch"],
        "arcs_received_oracle_incidence": False,
    }
    if payload.get("oracle_information_receipt") != expected_receipt:
        errors.append("EXP-277 oracle-information receipt drift")

    primary = payload.get("primary_endpoint") or {}
    if primary != {
        "metric": "verified_utility_per_accounted_flop",
        "direction": "higher",
        "mesi_relative_gain": 0.10,
    }:
        errors.append("EXP-277 primary endpoint or MESI drift")
    protected = payload.get("protected_endpoints") or {}
    if protected.get("verified_solution_rate_floor") != "oracle_cbrf >= arcs_branch - 0.005":
        errors.append("EXP-277 protected solution-rate floor drift")

    training = payload.get("training") or {}
    evaluation = payload.get("evaluation") or {}
    if training.get("rng_stream") != "augmentation":
        errors.append("EXP-277 training stream must be augmentation")
    if evaluation.get("rng_stream") != "evaluation":
        errors.append("EXP-277 evaluation stream must be evaluation")
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = list(training.get("paired_batch_digests") or [])
    if train_start != 0:
        errors.append("EXP-277 training replicate lineage must start at 0")
    if train_count <= 0 or len(train_digests) != train_count or len(set(train_digests)) != len(train_digests):
        errors.append("EXP-277 training batch lineage is incomplete or non-unique")

    eval_start = int(evaluation.get("start_replicate", -1))
    eval_count = int(evaluation.get("replicates", 0) or 0)
    rows = list(evaluation.get("per_replicate") or [])
    if eval_count <= 0 or len(rows) != eval_count:
        errors.append("EXP-277 evaluation replicate count does not match raw lineage")
    if train_count > 0 and train_start >= 0 and eval_start < train_start + train_count and eval_start + eval_count > train_start:
        errors.append("EXP-277 training and evaluation replicate lineages overlap")
    if rows:
        indexes = [row.get("replicate") for row in rows]
        if indexes != list(range(eval_start, eval_start + len(rows))):
            errors.append("EXP-277 evaluation replicate lineage is reordered or incomplete")
        digests = [row.get("paired_batch_digest") for row in rows]
        if any(not digest for digest in digests) or len(set(digests)) != len(digests):
            errors.append("EXP-277 evaluation paired batch digests must be non-empty and unique")
        for row in rows:
            if row.get("world_pairing_closed") is not True:
                errors.append("EXP-277 per-replicate world pairing is not closed")
            for arm in ("arcs_branch", "oracle_cbrf"):
                metrics = row.get(arm) or {}
                flops = int(metrics.get("accounted_flops_per_episode", 0) or 0)
                if flops <= 0 or (declared_budget > 0 and flops > declared_budget):
                    errors.append(f"EXP-277 {arm} accounted FLOPs violate declared ceiling")
                if "verified_utility_per_accounted_flop" not in metrics:
                    errors.append(f"EXP-277 {arm} primary utility is missing")
        aggregate = evaluation.get("aggregate") or {}
        if int(aggregate.get("n", 0) or 0) != len(rows):
            errors.append("EXP-277 aggregate n mismatch")
        if rows:
            expected_arcs_utility = _mean(rows, "arcs_branch", "verified_utility_per_accounted_flop")
            expected_oracle_utility = _mean(rows, "oracle_cbrf", "verified_utility_per_accounted_flop")
            expected_arcs_solution = _mean(rows, "arcs_branch", "verified_solution_rate")
            expected_oracle_solution = _mean(rows, "oracle_cbrf", "verified_solution_rate")
            if abs(float(aggregate.get("mean_arcs_utility", 0.0)) - expected_arcs_utility) > 1e-12:
                errors.append("EXP-277 aggregate ARCS utility mismatch")
            if abs(float(aggregate.get("mean_oracle_utility", 0.0)) - expected_oracle_utility) > 1e-12:
                errors.append("EXP-277 aggregate oracle utility mismatch")
            denominator = max(abs(expected_arcs_utility), 1e-12)
            expected_gain = (expected_oracle_utility - expected_arcs_utility) / denominator
            if abs(float(aggregate.get("oracle_relative_utility_gain", 0.0)) - expected_gain) > 1e-12:
                errors.append("EXP-277 aggregate relative utility gain mismatch")
            expected_solution_difference = expected_oracle_solution - expected_arcs_solution
            if abs(float(aggregate.get("oracle_minus_arcs_verified_solution_rate", 0.0)) - expected_solution_difference) > 1e-12:
                errors.append("EXP-277 protected endpoint aggregate mismatch")

    if payload.get("artifact_digest") not in (None, "") and payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-277 paired artifact digest mismatch")
    return errors


def run_exp277_paired_development(
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
    constraints: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    if not root_seed or not protocol_digest or not code_digest:
        raise ValueError("root_seed, protocol_digest and code_digest are required")
    if min(d_model, hidden_size, target_parameters, train_replicates, eval_replicates, batch_size, timesteps, variables, constraints) <= 0:
        raise ValueError("EXP-277 execution counts and dimensions must be positive")
    if constraints > variables or timesteps < constraints:
        raise ValueError("EXP-277 world geometry is invalid")
    if eval_start_replicate < train_replicates:
        raise ValueError("training and evaluation replicate lineages must be disjoint")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("EXP-277 noise/optimizer parameters are invalid")

    arcs, oracle, model_init_seed = _build_seeded_pair(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
    )
    initial_arcs = _functional_state_digest(arcs)
    initial_oracle = _functional_state_digest(oracle)
    if initial_arcs != initial_oracle:
        raise RuntimeError("EXP-277 matched arms do not share identical functional initialization")

    pair_audit = audit_matched_exp277_arm_pair(
        arcs,
        oracle,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
        max_accounted_flops_per_episode=max_accounted_flops_per_episode,
    )
    if not (
        pair_audit["parameter_match"]
        and pair_audit["functional_parameter_match"]
        and pair_audit["oracle_information_separation"]
        and pair_audit["compute_budget_closed"]
    ):
        raise RuntimeError("EXP-277 resource match did not close before paired execution")

    arcs_flops = int(pair_audit["compute_ledger"]["arcs_branch"]["accounted_flops_per_episode"])
    oracle_flops = int(pair_audit["compute_ledger"]["oracle_cbrf"]["accounted_flops_per_episode"])
    generator = Exp277StructureDenseGenerator(root_seed=root_seed)
    arcs_optimizer = build_functional_optimizer(arcs, lr=lr, weight_decay=weight_decay)
    oracle_optimizer = build_functional_optimizer(oracle, lr=lr, weight_decay=weight_decay)

    training_digests: list[str] = []
    arcs_losses: list[float] = []
    oracle_losses: list[float] = []
    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            variables=variables,
            constraints=constraints,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="augmentation",
        )
        training_digests.append(batch.digest)
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

    arcs.eval()
    oracle.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(eval_replicates):
            replicate = eval_start_replicate + offset
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                constraints=constraints,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="evaluation",
            )
            arcs_output = arcs(batch.surface_events, batch.variable_states)
            oracle_output = oracle(batch.surface_events, batch.variable_states, batch.oracle_incidence)
            arcs_metrics = _external_metrics(arcs_output, batch.targets, accounted_flops=arcs_flops)
            oracle_metrics = _external_metrics(oracle_output, batch.targets, accounted_flops=oracle_flops)
            arcs_utility = float(arcs_metrics["verified_utility_per_accounted_flop"])
            oracle_utility = float(oracle_metrics["verified_utility_per_accounted_flop"])
            rows.append(
                {
                    "replicate": replicate,
                    "paired_batch_digest": batch.digest,
                    "world_pairing_closed": True,
                    "arcs_branch": arcs_metrics,
                    "oracle_cbrf": oracle_metrics,
                    "oracle_relative_verified_utility_gain": (oracle_utility - arcs_utility) / max(abs(arcs_utility), 1e-12),
                }
            )

    mean_arcs_utility = _mean(rows, "arcs_branch", "verified_utility_per_accounted_flop")
    mean_oracle_utility = _mean(rows, "oracle_cbrf", "verified_utility_per_accounted_flop")
    mean_arcs_solution = _mean(rows, "arcs_branch", "verified_solution_rate")
    mean_oracle_solution = _mean(rows, "oracle_cbrf", "verified_solution_rate")
    aggregate_gain = (mean_oracle_utility - mean_arcs_utility) / max(abs(mean_arcs_utility), 1e-12)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scope": "synthetic-exp277-paired-oracle-structure-development",
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "initial_state": {
            "arcs_branch_digest": initial_arcs,
            "oracle_cbrf_digest": initial_oracle,
            "functional_digest_match": initial_arcs == initial_oracle,
        },
        "final_state": {
            "arcs_branch_digest": _functional_state_digest(arcs),
            "oracle_cbrf_digest": _functional_state_digest(oracle),
        },
        "arm_geometry": {
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
        },
        "world_geometry": {
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "noise_std": float(noise_std),
        },
        "oracle_information_receipt": {
            "artifact": "oracle_incidence",
            "ground_truth": True,
            "delivered_to": ["oracle_cbrf"],
            "withheld_from": ["arcs_branch"],
            "arcs_received_oracle_incidence": False,
        },
        "resource_match": {
            "parameter_match": pair_audit["parameter_match"],
            "functional_parameter_match": pair_audit["functional_parameter_match"],
            "same_world_lineage": True,
            "compute_budget_closed": pair_audit["compute_budget_closed"],
            "declared_max_accounted_flops_per_episode": pair_audit["declared_max_accounted_flops_per_episode"],
            "pair_audit": deepcopy(pair_audit),
            "pair_audit_digest": canonical_sha256(pair_audit),
        },
        "primary_endpoint": {
            "metric": "verified_utility_per_accounted_flop",
            "direction": "higher",
            "mesi_relative_gain": 0.10,
        },
        "protected_endpoints": {
            "verified_solution_rate_floor": "oracle_cbrf >= arcs_branch - 0.005",
        },
        "training": {
            "rng_stream": "augmentation",
            "start_replicate": 0,
            "replicates": train_replicates,
            "paired_batch_digests": training_digests,
            "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": weight_decay},
            "arcs_branch_losses": arcs_losses,
            "oracle_cbrf_losses": oracle_losses,
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": eval_start_replicate,
            "replicates": eval_replicates,
            "per_replicate": rows,
            "aggregate": {
                "n": eval_replicates,
                "mean_arcs_utility": mean_arcs_utility,
                "mean_oracle_utility": mean_oracle_utility,
                "oracle_relative_utility_gain": aggregate_gain,
                "oracle_minus_arcs_verified_solution_rate": mean_oracle_solution - mean_arcs_solution,
            },
        },
        "remaining_blockers": [
            "development evidence cannot promote H-CBRF-01",
            "confirmatory sample-size/analysis freeze and confirmatory-open execution remain open",
            "post-freeze challenge beacon remains unmaterialized",
        ],
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp277_paired_development(payload)
    if errors:
        raise RuntimeError("invalid EXP-277 paired development artifact: " + "; ".join(errors))
    return payload
