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
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import (
    BranchOnlyArm,
    HybridRoutingArm,
    PropagationOnlyArm,
    audit_matched_exp279_arm_triplet,
    build_matched_exp279_arm_triplet,
)

SCHEMA = "NLM-EXP-279-PAIRED-DEV-EVAL-V1"
ARM_ORDER = ("propagation_only", "branch_only", "hybrid")


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _functional_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-279-FUNCTIONAL-STATE-V1\0")
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


def _build_seeded_triplet(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
) -> tuple[PropagationOnlyArm, BranchOnlyArm, HybridRoutingArm, int]:
    seed = derive_stream_seed(root_seed, "EXP-279", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        propagation, branch, hybrid = build_matched_exp279_arm_triplet(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            route_threshold=route_threshold,
            device="cpu",
        )
    return propagation, branch, hybrid, seed


def _train_arm(
    arm: PropagationOnlyArm | BranchOnlyArm | HybridRoutingArm,
    optimizer: torch.optim.Optimizer,
    *,
    arm_id: str,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    arm.train()
    optimizer.zero_grad(set_to_none=True)
    if arm_id == "branch_only":
        output = arm(surface_events, variable_states)
    else:
        output = arm(surface_events, variable_states, incidence)
    loss = F.cross_entropy(output.decision_logits.reshape(-1, 2), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _external_metrics(
    output: Any,
    targets: torch.Tensor,
    *,
    accounted_flops: int,
) -> dict[str, float | int]:
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
        "verified_utility_per_accounted_flop_on_structure_dense_stratum": utility,
    }


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return sum(float(row[arm][metric]) for row in rows) / len(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("EXP-279 aggregate requires raw paired rows")
    metric = "verified_utility_per_accounted_flop_on_structure_dense_stratum"
    mean_prop = _mean(rows, "propagation_only", metric)
    mean_branch = _mean(rows, "branch_only", metric)
    mean_hybrid = _mean(rows, "hybrid", metric)
    mean_prop_solution = _mean(rows, "propagation_only", "verified_solution_rate")
    mean_branch_solution = _mean(rows, "branch_only", "verified_solution_rate")
    mean_hybrid_solution = _mean(rows, "hybrid", "verified_solution_rate")
    if mean_prop >= mean_branch:
        best_simple = "propagation_only"
        best_simple_utility = mean_prop
        best_simple_solution = mean_prop_solution
    else:
        best_simple = "branch_only"
        best_simple_utility = mean_branch
        best_simple_solution = mean_branch_solution
    denominator = max(abs(best_simple_utility), 1e-12)
    relative_gain = (mean_hybrid - best_simple_utility) / denominator

    by_stratum: dict[str, Any] = {}
    for stratum in STRATA:
        subset = [row for row in rows if row.get("stratum") == stratum]
        if not subset:
            raise ValueError(f"EXP-279 aggregate missing predeclared stratum {stratum}")
        by_stratum[stratum] = {
            "n": len(subset),
            "mean_propagation_utility": _mean(subset, "propagation_only", metric),
            "mean_branch_utility": _mean(subset, "branch_only", metric),
            "mean_hybrid_utility": _mean(subset, "hybrid", metric),
            "mean_propagation_verified_solution_rate": _mean(subset, "propagation_only", "verified_solution_rate"),
            "mean_branch_verified_solution_rate": _mean(subset, "branch_only", "verified_solution_rate"),
            "mean_hybrid_verified_solution_rate": _mean(subset, "hybrid", "verified_solution_rate"),
            "mean_hybrid_route_fraction": sum(float(row["hybrid_route_receipt"]["route_fraction"]) for row in subset) / len(subset),
        }

    return {
        "n": len(rows),
        "structure_fit_strata": list(STRATA),
        "mean_propagation_utility": mean_prop,
        "mean_branch_utility": mean_branch,
        "mean_hybrid_utility": mean_hybrid,
        "mean_propagation_verified_solution_rate": mean_prop_solution,
        "mean_branch_verified_solution_rate": mean_branch_solution,
        "mean_hybrid_verified_solution_rate": mean_hybrid_solution,
        "best_simple_arm": best_simple,
        "best_simple_mean_utility": best_simple_utility,
        "best_simple_mean_verified_solution_rate": best_simple_solution,
        "hybrid_relative_utility_gain": relative_gain,
        "hybrid_minus_best_simple_verified_solution_rate": mean_hybrid_solution - best_simple_solution,
        "by_stratum": by_stratum,
        "analysis_boundary": "descriptive DEVELOPMENT aggregates only; no bootstrap/Holm confirmatory inference executed",
    }


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def validate_exp279_paired_development(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-279 paired development schema")
    if payload.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1":
        errors.append("EXP-279 protocol authority drift")
    if payload.get("arm_order") != list(ARM_ORDER):
        errors.append("EXP-279 arm ordering drift")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-279 paired development artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 paired development artifact cannot promote a claim")
    if payload.get("confirmatory_ready") is not False:
        errors.append("EXP-279 paired development artifact cannot be confirmatory-ready")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-279 development cannot consume confirmatory data")
    if payload.get("challenge_materialized") is not False:
        errors.append("EXP-279 development cannot materialize challenge randomness")
    if payload.get("decision_rule_executed") is not False:
        errors.append("EXP-279 development cannot execute the scientific promotion rule")
    if not payload.get("protocol_digest") or not payload.get("code_digest"):
        errors.append("EXP-279 paired development provenance is incomplete")

    initial = payload.get("initial_state") or {}
    digests = [
        initial.get("propagation_only_digest"),
        initial.get("branch_only_digest"),
        initial.get("hybrid_digest"),
    ]
    if initial.get("functional_digest_match") is not True or any(not digest for digest in digests) or len(set(digests)) != 1:
        errors.append("EXP-279 triplet must share identical functional initialization")

    resource = payload.get("resource_match") or {}
    for field in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "same_world_lineage",
        "compute_budget_closed",
    ):
        if resource.get(field) is not True:
            errors.append(f"EXP-279 resource match field is not closed: {field}")
    if resource.get("structure_fit_strata") != list(STRATA):
        errors.append("EXP-279 predeclared structure-fit strata drift")
    pair_audit = resource.get("triplet_audit") or {}
    if pair_audit.get("schema") != "NLM-EXP-279-MATCHED-ROUTING-ARMS-DEV-V1":
        errors.append("EXP-279 matched triplet audit schema drift")
    if resource.get("triplet_audit_digest") != canonical_sha256(pair_audit):
        errors.append("EXP-279 matched triplet audit digest mismatch")
    declared_budget = int(resource.get("declared_max_accounted_flops_per_episode", 0) or 0)
    if declared_budget <= 0:
        errors.append("EXP-279 declared compute ceiling is missing")
    if pair_audit:
        for field in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "reclaimed_parameter_assignment_closed",
            "compute_budget_closed",
        ):
            if pair_audit.get(field) is not True:
                errors.append(f"EXP-279 triplet audit resource field drift: {field}")
        if pair_audit.get("structure_fit_strata") != list(STRATA):
            errors.append("EXP-279 triplet audit strata drift")

    expected_receipt = {
        "artifact": "constraint_variable_incidence",
        "ground_truth": True,
        "delivered_to": ["propagation_only", "hybrid"],
        "withheld_from": ["branch_only"],
        "branch_only_received_incidence": False,
    }
    if payload.get("information_receipt") != expected_receipt:
        errors.append("EXP-279 incidence information receipt drift")

    route_config = payload.get("route_config") or {}
    if route_config.get("strategy") != "propagation_then_branch_on_residual_uncertainty":
        errors.append("EXP-279 hybrid routing strategy drift")
    try:
        route_threshold = float(route_config.get("threshold"))
    except (TypeError, ValueError):
        route_threshold = -1.0
    if not 0.0 <= route_threshold <= 1.0:
        errors.append("EXP-279 route threshold is invalid")
    if route_config.get("frozen_before_evaluation") is not True:
        errors.append("EXP-279 route threshold was not frozen before evaluation")

    if payload.get("primary_endpoint") != {
        "metric": "verified_utility_per_accounted_flop_on_structure_dense_stratum",
        "direction": "higher",
        "mesi_relative_gain": 0.08,
    }:
        errors.append("EXP-279 primary endpoint or MESI drift")
    protected = payload.get("protected_endpoints") or {}
    if protected.get("verified_solution_rate_floor") != "hybrid >= best_simple - 0.01":
        errors.append("EXP-279 protected solution-rate floor drift")

    training = payload.get("training") or {}
    evaluation = payload.get("evaluation") or {}
    if training.get("rng_stream") != "augmentation":
        errors.append("EXP-279 training stream must be augmentation")
    if evaluation.get("rng_stream") != "evaluation":
        errors.append("EXP-279 evaluation stream must be evaluation")
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = list(training.get("paired_batch_digests") or [])
    train_strata = list(training.get("stratum_sequence") or [])
    if train_start != 0:
        errors.append("EXP-279 training replicate lineage must start at 0")
    if training.get("strata") != list(STRATA):
        errors.append("EXP-279 training strata authority drift")
    if train_count < len(STRATA) or len(train_digests) != train_count or len(set(train_digests)) != len(train_digests):
        errors.append("EXP-279 training batch lineage is incomplete or non-unique")
    expected_train_strata = [STRATA[index % len(STRATA)] for index in range(max(train_count, 0))]
    if train_strata != expected_train_strata:
        errors.append("EXP-279 training stratum ordering drift")

    eval_start = int(evaluation.get("start_replicate", -1))
    eval_count = int(evaluation.get("replicates", 0) or 0)
    rows = list(evaluation.get("per_replicate") or [])
    if eval_count < len(STRATA):
        errors.append("EXP-279 evaluation must cover all predeclared strata")
    if len(rows) != eval_count:
        errors.append("EXP-279 evaluation replicate count does not match raw rows")
    if train_count > 0 and train_start >= 0 and eval_start < train_start + train_count and eval_start + eval_count > train_start:
        errors.append("EXP-279 training and evaluation replicate lineages overlap")

    ledger = pair_audit.get("compute_ledger") or {}
    hybrid_ledger = ledger.get("hybrid") or {}
    hybrid_base = int(hybrid_ledger.get("base_without_branch_flops", 0) or 0)
    hybrid_increment = int(hybrid_ledger.get("branch_increment_flops", 0) or 0)
    if rows:
        expected_replicates = list(range(eval_start, eval_start + len(rows)))
        if [row.get("replicate") for row in rows] != expected_replicates:
            errors.append("EXP-279 evaluation replicate lineage is reordered or incomplete")
        expected_strata = [STRATA[index % len(STRATA)] for index in range(len(rows))]
        if [row.get("stratum") for row in rows] != expected_strata:
            errors.append("EXP-279 blocked structure-fit stratum ordering drift")
        batch_digests = [row.get("paired_batch_digest") for row in rows]
        if any(not digest for digest in batch_digests) or len(set(batch_digests)) != len(batch_digests):
            errors.append("EXP-279 evaluation paired batch digests must be non-empty and unique")

        primary_metric = "verified_utility_per_accounted_flop_on_structure_dense_stratum"
        for row in rows:
            if row.get("world_pairing_closed") is not True:
                errors.append("EXP-279 per-replicate world pairing is not closed")
            receipt = row.get("hybrid_route_receipt") or {}
            if not _numbers_close(receipt.get("threshold"), route_threshold):
                errors.append("EXP-279 route receipt threshold drift")
            route_mask = list(receipt.get("branch_route_mask") or [])
            batch_size = int(receipt.get("batch_size", 0) or 0)
            routed = int(receipt.get("routed_episodes", -1))
            if batch_size <= 0 or len(route_mask) != batch_size or any(type(item) is not bool for item in route_mask):
                errors.append("EXP-279 route receipt mask is malformed")
            expected_routed = sum(bool(item) for item in route_mask)
            if routed != expected_routed:
                errors.append("EXP-279 route receipt routed count mismatch")
            expected_fraction = expected_routed / batch_size if batch_size > 0 else -1.0
            if not _numbers_close(receipt.get("route_fraction"), expected_fraction):
                errors.append("EXP-279 route receipt fraction mismatch")

            for arm in ARM_ORDER:
                metrics = row.get(arm) or {}
                flops = int(metrics.get("accounted_flops_per_episode", 0) or 0)
                if flops <= 0 or (declared_budget > 0 and flops > declared_budget):
                    errors.append(f"EXP-279 {arm} accounted FLOPs violate declared ceiling")
                if primary_metric not in metrics:
                    errors.append(f"EXP-279 {arm} primary utility is missing")
                else:
                    solution_rate = float(metrics.get("verified_solution_rate", 0.0) or 0.0)
                    expected_utility = solution_rate / max(flops, 1)
                    if not _numbers_close(metrics.get(primary_metric), expected_utility):
                        errors.append(f"EXP-279 {arm} normalized utility mismatch")

            if hybrid_base > 0 and batch_size > 0:
                expected_hybrid_flops = int(round(hybrid_base + hybrid_increment * expected_fraction))
                observed_hybrid_flops = int((row.get("hybrid") or {}).get("accounted_flops_per_episode", 0) or 0)
                if observed_hybrid_flops != expected_hybrid_flops:
                    errors.append("EXP-279 hybrid routed compute receipt mismatch")

        try:
            expected_aggregate = _aggregate_rows(rows)
            observed_aggregate = evaluation.get("aggregate") or {}
            if observed_aggregate.get("n") != expected_aggregate["n"]:
                errors.append("EXP-279 aggregate n mismatch")
            if observed_aggregate.get("structure_fit_strata") != expected_aggregate["structure_fit_strata"]:
                errors.append("EXP-279 aggregate strata drift")
            if observed_aggregate.get("best_simple_arm") != expected_aggregate["best_simple_arm"]:
                errors.append("EXP-279 best-simple aggregate mismatch")
            numeric_fields = (
                "mean_propagation_utility",
                "mean_branch_utility",
                "mean_hybrid_utility",
                "mean_propagation_verified_solution_rate",
                "mean_branch_verified_solution_rate",
                "mean_hybrid_verified_solution_rate",
                "best_simple_mean_utility",
                "best_simple_mean_verified_solution_rate",
                "hybrid_relative_utility_gain",
                "hybrid_minus_best_simple_verified_solution_rate",
            )
            for field in numeric_fields:
                if not _numbers_close(observed_aggregate.get(field), expected_aggregate[field]):
                    errors.append(f"EXP-279 aggregate mismatch: {field}")
            observed_by_stratum = observed_aggregate.get("by_stratum") or {}
            if set(observed_by_stratum) != set(STRATA):
                errors.append("EXP-279 by-stratum aggregate coverage drift")
            else:
                for stratum in STRATA:
                    expected_block = expected_aggregate["by_stratum"][stratum]
                    observed_block = observed_by_stratum[stratum]
                    if observed_block.get("n") != expected_block["n"]:
                        errors.append(f"EXP-279 {stratum} aggregate n mismatch")
                    for field, expected_value in expected_block.items():
                        if field == "n":
                            continue
                        if not _numbers_close(observed_block.get(field), expected_value):
                            errors.append(f"EXP-279 {stratum} aggregate mismatch: {field}")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"EXP-279 aggregate recomputation failed: {exc}")

    if payload.get("artifact_digest") not in (None, "") and payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-279 paired artifact digest mismatch")
    return errors


def run_exp279_paired_development(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
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
        raise ValueError("EXP-279 execution counts and dimensions must be positive")
    if train_replicates < len(STRATA) or eval_replicates < len(STRATA):
        raise ValueError("EXP-279 train/eval replicates must be at least 3 to cover all predeclared strata")
    if constraints > variables or timesteps < constraints:
        raise ValueError("EXP-279 world geometry is invalid")
    if eval_start_replicate < train_replicates:
        raise ValueError("EXP-279 training and evaluation replicate lineages must be disjoint")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("EXP-279 noise/optimizer parameters are invalid")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be in [0, 1]")

    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=route_threshold,
    )
    initial_digests = {
        "propagation_only_digest": _functional_state_digest(propagation),
        "branch_only_digest": _functional_state_digest(branch),
        "hybrid_digest": _functional_state_digest(hybrid),
    }
    if len(set(initial_digests.values())) != 1:
        raise RuntimeError("EXP-279 matched triplet does not share identical functional initialization")

    triplet_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
        max_accounted_flops_per_episode=max_accounted_flops_per_episode,
    )
    required_resource_flags = (
        triplet_audit["parameter_match"],
        triplet_audit["functional_parameter_match"],
        triplet_audit["active_functional_parameter_match"],
        triplet_audit["reclaimed_parameter_assignment_closed"],
        triplet_audit["compute_budget_closed"],
    )
    if not all(required_resource_flags):
        raise RuntimeError("EXP-279 matched resource contract is not closed before paired execution")

    generator = Exp279RoutingGenerator(root_seed=root_seed)
    arms: dict[str, PropagationOnlyArm | BranchOnlyArm | HybridRoutingArm] = {
        "propagation_only": propagation,
        "branch_only": branch,
        "hybrid": hybrid,
    }
    optimizers = {
        arm_id: build_functional_optimizer(arm, lr=lr, weight_decay=weight_decay)
        for arm_id, arm in arms.items()
    }

    training_digests: list[str] = []
    training_strata: list[str] = []
    losses: dict[str, list[float]] = {arm_id: [] for arm_id in ARM_ORDER}
    for replicate in range(train_replicates):
        stratum = STRATA[replicate % len(STRATA)]
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            variables=variables,
            constraints=constraints,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="augmentation",
            stratum=stratum,
        )
        training_digests.append(batch.digest)
        training_strata.append(stratum)
        for arm_id in ARM_ORDER:
            losses[arm_id].append(
                _train_arm(
                    arms[arm_id],
                    optimizers[arm_id],
                    arm_id=arm_id,
                    surface_events=batch.surface_events,
                    variable_states=batch.variable_states,
                    incidence=batch.incidence,
                    targets=batch.targets,
                )
            )

    for arm in arms.values():
        arm.eval()

    compute = triplet_audit["compute_ledger"]
    prop_flops = int(compute["propagation_only"]["max_accounted_flops_per_episode"])
    branch_flops = int(compute["branch_only"]["max_accounted_flops_per_episode"])
    hybrid_base = int(compute["hybrid"]["base_without_branch_flops"])
    hybrid_increment = int(compute["hybrid"]["branch_increment_flops"])

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(eval_replicates):
            replicate = eval_start_replicate + offset
            stratum = STRATA[offset % len(STRATA)]
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                constraints=constraints,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="evaluation",
                stratum=stratum,
            )
            prop_output = propagation(batch.surface_events, batch.variable_states, batch.incidence)
            branch_output = branch(batch.surface_events, batch.variable_states)
            hybrid_output = hybrid(batch.surface_events, batch.variable_states, batch.incidence)
            route_mask = [bool(item) for item in hybrid_output.branch_route_mask.detach().cpu().tolist()]
            routed = sum(route_mask)
            route_fraction = routed / batch_size
            hybrid_flops = int(round(hybrid_base + hybrid_increment * route_fraction))
            rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "paired_batch_digest": batch.digest,
                    "world_pairing_closed": True,
                    "propagation_only": _external_metrics(prop_output, batch.targets, accounted_flops=prop_flops),
                    "branch_only": _external_metrics(branch_output, batch.targets, accounted_flops=branch_flops),
                    "hybrid": _external_metrics(hybrid_output, batch.targets, accounted_flops=hybrid_flops),
                    "hybrid_route_receipt": {
                        "threshold": float(route_threshold),
                        "batch_size": batch_size,
                        "routed_episodes": routed,
                        "branch_route_mask": route_mask,
                        "route_fraction": route_fraction,
                        "base_without_branch_flops": hybrid_base,
                        "branch_increment_flops": hybrid_increment,
                        "accounted_flops_per_episode": hybrid_flops,
                    },
                }
            )

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": "EXP-279",
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "arm_order": list(ARM_ORDER),
        "initial_state": {
            "model_init_seed": model_init_seed,
            **initial_digests,
            "functional_digest_match": True,
        },
        "information_receipt": {
            "artifact": "constraint_variable_incidence",
            "ground_truth": True,
            "delivered_to": ["propagation_only", "hybrid"],
            "withheld_from": ["branch_only"],
            "branch_only_received_incidence": False,
        },
        "route_config": {
            "strategy": "propagation_then_branch_on_residual_uncertainty",
            "threshold": float(route_threshold),
            "frozen_before_evaluation": True,
        },
        "primary_endpoint": {
            "metric": "verified_utility_per_accounted_flop_on_structure_dense_stratum",
            "direction": "higher",
            "mesi_relative_gain": 0.08,
        },
        "protected_endpoints": {
            "verified_solution_rate_floor": "hybrid >= best_simple - 0.01",
        },
        "resource_match": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "active_functional_parameter_match": True,
            "reclaimed_parameter_assignment_closed": True,
            "same_world_lineage": True,
            "compute_budget_closed": True,
            "structure_fit_strata": list(STRATA),
            "declared_max_accounted_flops_per_episode": int(triplet_audit["declared_max_accounted_flops_per_episode"]),
            "triplet_audit": triplet_audit,
            "triplet_audit_digest": canonical_sha256(triplet_audit),
        },
        "training": {
            "rng_stream": "augmentation",
            "start_replicate": 0,
            "replicates": train_replicates,
            "strata": list(STRATA),
            "stratum_sequence": training_strata,
            "paired_batch_digests": training_digests,
            "mean_losses": {
                arm_id: sum(losses[arm_id]) / len(losses[arm_id])
                for arm_id in ARM_ORDER
            },
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": eval_start_replicate,
            "replicates": eval_replicates,
            "per_replicate": rows,
            "aggregate": _aggregate_rows(rows),
        },
        "scientific_boundary": {
            "confirmatory_analysis_executed": False,
            "holm_multiplicity_executed": False,
            "promotion_or_kill_decision_emitted": False,
            "note": "development execution closes implementation/provenance only; frozen confirmatory and post-freeze challenge gates remain open",
        },
    }
    artifact["artifact_digest"] = _artifact_digest(artifact)
    errors = validate_exp279_paired_development(artifact)
    if errors:
        raise RuntimeError("EXP-279 paired development artifact failed self-validation: " + "; ".join(errors))
    return artifact
