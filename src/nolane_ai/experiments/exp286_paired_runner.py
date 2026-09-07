from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from statistics import median
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes
from .exp286_conflict_worlds import Exp286ConflictBatch, Exp286ConflictGenerator
from .matched_conflict_arms import (
    ChronologicalFailureArm,
    OracleConflictCoreArm,
    audit_matched_exp286_arm_pair,
    build_matched_exp286_arm_pair,
)

SCHEMA = "NLM-EXP-286-PAIRED-DEV-EVAL-V1"
ARM_ORDER = ("chronological_failure", "oracle_conflict_core")
PRIMARY_METRIC = "accounted_reasoning_flops_to_verified_solution"
PROTECTED_FLOOR = "oracle_conflict_core >= chronological_failure - 0.005"
MULTIPLICITY_FAMILY = "CONFLICT_VALUE"
ANALYSIS_BOUNDARY = (
    "descriptive DEVELOPMENT statistics only; frozen confirmatory bootstrap inference not executed"
)
_INFORMATION_RECEIPT = {
    "artifact": "ground_truth_conflict_core",
    "ground_truth": True,
    "delivery_event": "after_current_contradiction_only",
    "delivered_to": ["oracle_conflict_core"],
    "withheld_from": ["chronological_failure"],
    "chronological_failure_received_conflict_core": False,
    "future_conflict_core_leakage": False,
    "solution_leakage": False,
}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _functional_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-286-FUNCTIONAL-STATE-V1\0")
    for name, parameter in functional_trainable_named_parameters(model):
        cpu = parameter.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "byteorder": tensor_byteorder(),
            },
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
) -> tuple[ChronologicalFailureArm, OracleConflictCoreArm, int]:
    seed = derive_stream_seed(root_seed, "EXP-286", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        chronological, oracle = build_matched_exp286_arm_pair(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            device="cpu",
        )
    return chronological, oracle, seed


def _conflict_targets(batch: Exp286ConflictBatch) -> torch.Tensor:
    return torch.tensor(
        [int(item["conflict_variable"]) for item in batch.metadata["episodes"]],
        dtype=torch.long,
        device=batch.variable_states.device,
    )


def _train_step(
    arm: ChronologicalFailureArm | OracleConflictCoreArm,
    optimizer: torch.optim.Optimizer,
    *,
    arm_id: str,
    batch: Exp286ConflictBatch,
) -> float:
    arm.train()
    optimizer.zero_grad(set_to_none=True)
    if arm_id == "chronological_failure":
        output = arm(
            batch.surface_events,
            batch.variable_states,
            contradiction_observed=True,
        )
    else:
        output = arm(
            batch.surface_events,
            batch.variable_states,
            conflict_core_mask=batch.core_masks,
            contradiction_observed=True,
        )
    rollback_loss = F.cross_entropy(output.rollback_logits, _conflict_targets(batch))
    verifier_loss = F.binary_cross_entropy(
        output.verifier_confidence,
        batch.solution_targets.to(dtype=output.verifier_confidence.dtype),
    )
    loss = rollback_loss + verifier_loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _oracle_null_forward(
    arm: OracleConflictCoreArm,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the matched neural null-core path without delivering privileged information."""
    events, variables = arm._validate_common(surface_events, variable_states)
    null_core = torch.zeros(
        variables.shape[0],
        variables.shape[1],
        1,
        device=variables.device,
        dtype=variables.dtype,
    )
    return arm._shared_state(events, variables, null_core)


def _target_digest(replicate: int, episode_index: int, solution: list[int]) -> str:
    return canonical_sha256(
        {
            "experiment_id": "EXP-286",
            "replicate": int(replicate),
            "episode_index": int(episode_index),
            "verified_solution": [int(value) for value in solution],
        }
    )


def _search_episode(
    *,
    arm_id: str,
    arm: ChronologicalFailureArm | OracleConflictCoreArm,
    batch: Exp286ConflictBatch,
    episode_index: int,
    per_step_flops: int,
    ceiling: int,
    max_search_steps: int,
) -> dict[str, Any]:
    surface = batch.surface_events[episode_index : episode_index + 1]
    variable_states = batch.variable_states[episode_index : episode_index + 1]
    core_mask = batch.core_masks[episode_index : episode_index + 1]
    target = [int(value) for value in batch.solution_targets[episode_index].detach().cpu().tolist()]
    episode = batch.metadata["episodes"][episode_index]
    conflict_variable = int(episode["conflict_variable"])
    core_variables = [int(value) for value in episode["core_variables"]]
    decoy_variables = [int(value) for value in episode["decoy_variables"]]
    bad_value = int(episode["bad_branch_value"])
    original_order = [int(value) for value in batch.decoy_order[episode_index].detach().cpu().tolist()]

    queue = list(original_order)
    assignment: dict[int, int] = {}
    visited: list[int] = []
    receipts: list[dict[str, Any]] = []
    contradiction_count = 0
    core_delivery_count = 0
    bad_attempted = False
    solved = False

    for step_index in range(1, max_search_steps + 1):
        if not queue:
            break
        variable = int(queue.pop(0))
        visited.append(variable)
        contradiction = variable == conflict_variable and not bad_attempted
        core_delivered = False

        if contradiction:
            bad_attempted = True
            contradiction_count += 1
            assigned_value = bad_value
            if arm_id == "chronological_failure":
                output = arm(
                    surface,
                    variable_states,
                    contradiction_observed=True,
                )
                rollback_logits = output.rollback_logits[0].detach().cpu()
                replay_decoys = [value for value in reversed(decoy_variables) if value in assignment]
                for value in replay_decoys:
                    assignment.pop(value, None)
                remaining = [
                    value
                    for value in original_order
                    if value not in replay_decoys
                    and value != conflict_variable
                    and value not in assignment
                ]
                queue = replay_decoys + [conflict_variable] + remaining
            else:
                output = arm(
                    surface,
                    variable_states,
                    conflict_core_mask=core_mask,
                    contradiction_observed=True,
                )
                core_delivered = True
                core_delivery_count += 1
                rollback_logits = output.rollback_logits[0].detach().cpu()
                ordered_core = sorted(
                    core_variables,
                    key=lambda value: (-float(rollback_logits[value].item()), value),
                )
                remaining = [
                    value
                    for value in original_order
                    if value not in ordered_core and value not in assignment
                ]
                queue = ordered_core + remaining
        else:
            assigned_value = target[variable]
            assignment[variable] = assigned_value
            if arm_id == "chronological_failure":
                output = arm(
                    surface,
                    variable_states,
                    contradiction_observed=False,
                )
                rollback_logits = output.rollback_logits[0].detach().cpu()
            else:
                rollback_logits, _ = _oracle_null_forward(arm, surface, variable_states)
                rollback_logits = rollback_logits[0].detach().cpu()

        receipts.append(
            {
                "step_index": step_index,
                "variable": variable,
                "assigned_value": int(assigned_value),
                "external_target_value": int(target[variable]),
                "contradiction_observed": bool(contradiction),
                "conflict_core_delivered": bool(core_delivered),
                "rollback_score_at_visited_variable": float(rollback_logits[variable].item()),
                "charged_accounted_flops": int(per_step_flops),
            }
        )

        solved = len(assignment) == len(target) and all(
            assignment.get(index) == expected for index, expected in enumerate(target)
        )
        if solved:
            break

    candidate_solution = [assignment.get(index, -1) for index in range(len(target))]
    actual_path_cost = len(receipts) * int(per_step_flops)
    if actual_path_cost > ceiling:
        raise RuntimeError("EXP-286 actual search path exceeded the declared compute ceiling")
    censored = not solved
    charged_cost = int(ceiling if censored else actual_path_cost)
    solution_digest = _target_digest(batch.replicate, episode_index, target) if solved else ""

    return {
        "episode_index": int(episode_index),
        "verified_solution": bool(solved),
        "candidate_solution": [int(value) for value in candidate_solution],
        "external_solution_verified": bool(solved),
        "verified_solution_digest": solution_digest,
        "accounted_reasoning_flops_to_verified_solution": charged_cost,
        "actual_executed_flops_before_censoring": int(actual_path_cost),
        "censored_at_max_flops": bool(censored),
        "search_steps": len(receipts),
        "visited_variables": visited,
        "contradiction_count": int(contradiction_count),
        "conflict_core_delivery_count": int(core_delivery_count),
        "conflict_core_precontradiction_delivery": False,
        "step_receipts": receipts,
    }


def _search_batch(
    *,
    arm_id: str,
    arm: ChronologicalFailureArm | OracleConflictCoreArm,
    batch: Exp286ConflictBatch,
    per_step_flops: int,
    ceiling: int,
    max_search_steps: int,
) -> dict[str, Any]:
    episodes = [
        _search_episode(
            arm_id=arm_id,
            arm=arm,
            batch=batch,
            episode_index=index,
            per_step_flops=per_step_flops,
            ceiling=ceiling,
            max_search_steps=max_search_steps,
        )
        for index in range(batch.solution_targets.shape[0])
    ]
    solution_rate = sum(float(item["verified_solution"]) for item in episodes) / len(episodes)
    mean_cost = sum(float(item[PRIMARY_METRIC]) for item in episodes) / len(episodes)
    all_solved = solution_rate == 1.0
    return {
        PRIMARY_METRIC: mean_cost,
        "verified_solution_rate": solution_rate,
        "search_steps": max(int(item["search_steps"]) for item in episodes),
        "visited_variables": [list(item["visited_variables"]) for item in episodes],
        "contradiction_count": sum(int(item["contradiction_count"]) for item in episodes),
        "conflict_core_received": arm_id == "oracle_conflict_core" and any(
            int(item["conflict_core_delivery_count"]) > 0 for item in episodes
        ),
        "conflict_core_delivery_count": sum(
            int(item["conflict_core_delivery_count"]) for item in episodes
        ),
        "conflict_core_precontradiction_delivery": False,
        "censored_at_max_flops": not all_solved,
        "censored_episode_count": sum(bool(item["censored_at_max_flops"]) for item in episodes),
        "external_solution_verified": all_solved,
        "verified_solution_digest": (
            canonical_sha256([item["verified_solution_digest"] for item in episodes])
            if all_solved
            else ""
        ),
        "episodes": episodes,
    }


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return sum(float(row[arm][metric]) for row in rows) / len(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chronological_costs = [float(row["chronological_failure"][PRIMARY_METRIC]) for row in rows]
    oracle_costs = [float(row["oracle_conflict_core"][PRIMARY_METRIC]) for row in rows]
    chronological_solution = _mean(rows, "chronological_failure", "verified_solution_rate")
    oracle_solution = _mean(rows, "oracle_conflict_core", "verified_solution_rate")
    log_ratios = [
        math.log(max(oracle, 1.0) / max(chronological, 1.0))
        for chronological, oracle in zip(chronological_costs, oracle_costs)
    ]
    mean_chronological = sum(chronological_costs) / len(chronological_costs)
    mean_oracle = sum(oracle_costs) / len(oracle_costs)
    return {
        "n": len(rows),
        "mean_chronological_cost": mean_chronological,
        "mean_oracle_cost": mean_oracle,
        "median_chronological_cost": float(median(chronological_costs)),
        "median_oracle_cost": float(median(oracle_costs)),
        "mean_chronological_solution_rate": chronological_solution,
        "mean_oracle_solution_rate": oracle_solution,
        "oracle_minus_chronological_verified_solution_rate": oracle_solution - chronological_solution,
        "paired_mean_log_cost_ratio": sum(log_ratios) / len(log_ratios),
        "descriptive_relative_flop_reduction": (
            (mean_chronological - mean_oracle) / max(abs(mean_chronological), 1.0)
        ),
        "oracle_lower_cost_sign_consistency": sum(
            oracle < chronological
            for chronological, oracle in zip(chronological_costs, oracle_costs)
        )
        / len(rows),
        "chronological_censored_episode_count": sum(
            int(row["chronological_failure"]["censored_episode_count"]) for row in rows
        ),
        "oracle_censored_episode_count": sum(
            int(row["oracle_conflict_core"]["censored_episode_count"]) for row in rows
        ),
        "analysis_boundary": ANALYSIS_BOUNDARY,
    }


def _validate_pair_audit(pair_audit: dict[str, Any], errors: list[str]) -> int:
    if pair_audit.get("schema") != "NLM-EXP-286-MATCHED-CONFLICT-ARMS-DEV-V1":
        errors.append("EXP-286 matched pair audit schema drift")
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "oracle_information_separation",
        "compute_budget_closed",
    ):
        if pair_audit.get(key) is not True:
            errors.append(f"EXP-286 matched pair audit {key} is not closed")
    if pair_audit.get("oracle_information_receipt") != _INFORMATION_RECEIPT:
        errors.append("EXP-286 pair-audit oracle information receipt drift")
    ceiling = int(pair_audit.get("declared_max_accounted_flops_per_episode", 0) or 0)
    if ceiling <= 0:
        errors.append("EXP-286 pair-audit compute ceiling is missing")
    ledger = pair_audit.get("compute_ledger") or {}
    for arm_id in ARM_ORDER:
        arm_ledger = ledger.get(arm_id) or {}
        per_step = int(arm_ledger.get("accounted_flops_per_search_step", 0) or 0)
        maximum = int(arm_ledger.get("max_accounted_flops_per_episode", 0) or 0)
        if per_step <= 0 or maximum <= 0 or maximum > ceiling:
            errors.append(f"EXP-286 {arm_id} analytical compute ledger is invalid")
        if arm_ledger.get("hardware_profiler_flops_claimed") is not False:
            errors.append(f"EXP-286 {arm_id} cannot claim hardware-profiler FLOPs")
    return ceiling


def validate_exp286_paired_development(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-286 paired development schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-286 paired development artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-286 paired development artifact cannot promote a scientific claim")
    for key in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(key) is not False:
            errors.append(f"EXP-286 development boundary requires {key}=false")
    if payload.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1":
        errors.append("EXP-286 protocol authority drift")
    if payload.get("experiment_id") != "EXP-286":
        errors.append("EXP-286 experiment id drift")
    if not payload.get("protocol_digest") or not payload.get("code_digest"):
        errors.append("EXP-286 paired development provenance is incomplete")
    if payload.get("arm_order") != list(ARM_ORDER):
        errors.append("EXP-286 arm ordering drift")
    if payload.get("primary_endpoint") != {
        "metric": PRIMARY_METRIC,
        "direction": "lower",
        "mesi_relative_reduction": 0.15,
    }:
        errors.append("EXP-286 primary endpoint or MESI drift")
    if (payload.get("protected_endpoints") or {}).get("verified_solution_rate_floor") != PROTECTED_FLOOR:
        errors.append("EXP-286 protected solution-rate floor drift")
    if payload.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        errors.append("EXP-286 multiplicity family drift")
    if payload.get("information_receipt") != _INFORMATION_RECEIPT:
        errors.append("EXP-286 contradiction-time oracle information receipt drift")

    initial = payload.get("initial_state") or {}
    digests = [
        initial.get("chronological_failure_digest"),
        initial.get("oracle_conflict_core_digest"),
    ]
    if initial.get("functional_digest_match") is not True or None in digests or len(set(digests)) != 1:
        errors.append("EXP-286 matched arms must share identical functional initialization")

    resource = payload.get("resource_match") or {}
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "same_initial_world_lineage",
        "compute_budget_closed",
        "oracle_information_separation",
    ):
        if resource.get(key) is not True:
            errors.append(f"EXP-286 resource match {key} is not closed")
    pair_audit = resource.get("pair_audit") or {}
    pair_errors: list[str] = []
    pair_ceiling = _validate_pair_audit(pair_audit, pair_errors)
    errors.extend(pair_errors)
    if resource.get("pair_audit_digest") != canonical_sha256(pair_audit):
        errors.append("EXP-286 matched pair audit digest mismatch")
    declared_ceiling = int(resource.get("declared_max_accounted_flops_per_episode", 0) or 0)
    if declared_ceiling != pair_ceiling or declared_ceiling <= 0:
        errors.append("EXP-286 common compute ceiling drift")

    config = payload.get("execution_config") or {}
    try:
        root_seed = str(config["root_seed"])
        batch_size = int(config["batch_size"])
        timesteps = int(config["timesteps"])
        variables = int(config["variables"])
        decoys = int(config["decoys"])
        d_model = int(config["d_model"])
        noise_std = float(config["noise_std"])
        max_search_steps = int(config["max_search_steps"])
    except (KeyError, TypeError, ValueError):
        root_seed = ""
        batch_size = timesteps = variables = decoys = d_model = max_search_steps = 0
        noise_std = -1.0
        errors.append("EXP-286 execution geometry is incomplete")
    generator = Exp286ConflictGenerator(root_seed=root_seed) if root_seed else None

    training = payload.get("training") or {}
    if training.get("rng_stream") != "augmentation":
        errors.append("EXP-286 training stream must be augmentation")
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = list(training.get("paired_batch_digests") or [])
    if train_start != 0:
        errors.append("EXP-286 training replicate lineage must start at 0")
    if train_count <= 0 or len(train_digests) != train_count or len(set(train_digests)) != len(train_digests):
        errors.append("EXP-286 training batch lineage is incomplete or non-unique")
    if generator is not None and train_count > 0:
        for replicate in range(min(train_count, len(train_digests))):
            regenerated = generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                decoys=decoys,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="augmentation",
            )
            if regenerated.digest != train_digests[replicate]:
                errors.append("EXP-286 training paired batch digest does not regenerate from sealed lineage")
                break

    evaluation = payload.get("evaluation") or {}
    if evaluation.get("rng_stream") != "evaluation":
        errors.append("EXP-286 evaluation stream must be evaluation")
    eval_start = int(evaluation.get("start_replicate", -1))
    eval_count = int(evaluation.get("replicates", 0) or 0)
    rows = list(evaluation.get("per_replicate") or [])
    if eval_count <= 0 or len(rows) != eval_count:
        errors.append("EXP-286 evaluation replicate count does not match raw lineage")
    if eval_start < train_start + train_count:
        errors.append("EXP-286 training and evaluation replicate lineages are not disjoint")
    if rows:
        expected_replicates = list(range(eval_start, eval_start + len(rows)))
        if [row.get("replicate") for row in rows] != expected_replicates:
            errors.append("EXP-286 evaluation replicate lineage is reordered or incomplete")
        row_digests = [row.get("paired_batch_digest") for row in rows]
        if any(not digest for digest in row_digests) or len(set(row_digests)) != len(row_digests):
            errors.append("EXP-286 evaluation paired batch digests must be non-empty and unique")

    ledgers = pair_audit.get("compute_ledger") or {}
    for row_offset, row in enumerate(rows):
        if row.get("initial_world_pairing_closed") is not True:
            errors.append("EXP-286 per-replicate initial-world pairing is not closed")
        if row.get("future_conflict_core_leakage") is not False:
            errors.append("EXP-286 future conflict-core leakage is forbidden")
        if row.get("solution_leakage") is not False:
            errors.append("EXP-286 solution leakage is forbidden")
        regenerated = None
        if generator is not None:
            regenerated = generator.make_batch(
                replicate=eval_start + row_offset,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                decoys=decoys,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="evaluation",
            )
            if row.get("paired_batch_digest") != regenerated.digest:
                errors.append("EXP-286 evaluation world digest does not regenerate from sealed lineage")

        for arm_id in ARM_ORDER:
            metrics = row.get(arm_id) or {}
            arm_ledger = ledgers.get(arm_id) or {}
            per_step = int(arm_ledger.get("accounted_flops_per_search_step", 0) or 0)
            episodes = list(metrics.get("episodes") or [])
            if len(episodes) != batch_size:
                errors.append(f"EXP-286 {arm_id} episode receipt count mismatch")
                continue
            episode_costs: list[float] = []
            solved_flags: list[bool] = []
            total_contradictions = 0
            total_deliveries = 0
            for episode_index, episode_result in enumerate(episodes):
                receipts = list(episode_result.get("step_receipts") or [])
                if not receipts or len(receipts) > max_search_steps:
                    errors.append(f"EXP-286 {arm_id} search-step receipt count is invalid")
                if [item.get("step_index") for item in receipts] != list(range(1, len(receipts) + 1)):
                    errors.append(f"EXP-286 {arm_id} search-step receipts are reordered")
                if any(int(item.get("charged_accounted_flops", -1)) != per_step for item in receipts):
                    errors.append(f"EXP-286 {arm_id} per-step accounted FLOPs drift from matched audit")
                contradictions = sum(bool(item.get("contradiction_observed")) for item in receipts)
                deliveries = sum(bool(item.get("conflict_core_delivered")) for item in receipts)
                total_contradictions += contradictions
                total_deliveries += deliveries
                if arm_id == "chronological_failure" and deliveries:
                    errors.append("EXP-286 chronological baseline received forbidden conflict-core information")
                if any(
                    bool(item.get("conflict_core_delivered")) and not bool(item.get("contradiction_observed"))
                    for item in receipts
                ):
                    errors.append("EXP-286 oracle conflict core was delivered before a current contradiction")
                if arm_id == "oracle_conflict_core" and deliveries > contradictions:
                    errors.append("EXP-286 oracle conflict-core delivery count exceeds current contradictions")

                actual_cost = len(receipts) * per_step
                solved = bool(episode_result.get("verified_solution"))
                candidate = list(episode_result.get("candidate_solution") or [])
                if regenerated is not None and len(candidate) == variables:
                    expected_target = [
                        int(value)
                        for value in regenerated.solution_targets[episode_index].detach().cpu().tolist()
                    ]
                    independently_verified = candidate == expected_target
                    if solved != independently_verified:
                        errors.append(f"EXP-286 {arm_id} external solution-verification flag is inconsistent")
                    expected_digest = (
                        _target_digest(eval_start + row_offset, episode_index, expected_target)
                        if independently_verified
                        else ""
                    )
                    if episode_result.get("verified_solution_digest", "") != expected_digest:
                        errors.append(f"EXP-286 {arm_id} verified solution digest mismatch")
                if bool(episode_result.get("external_solution_verified")) != solved:
                    errors.append(f"EXP-286 {arm_id} external verifier receipt mismatch")

                expected_charged = actual_cost if solved else declared_ceiling
                charged = float(episode_result.get(PRIMARY_METRIC, 0.0) or 0.0)
                if abs(charged - expected_charged) > 1e-9:
                    errors.append(f"EXP-286 {arm_id} censored/actual-path cost mismatch")
                if solved and episode_result.get("censored_at_max_flops") is not False:
                    errors.append(f"EXP-286 {arm_id} solved episode cannot be censored")
                if not solved and episode_result.get("censored_at_max_flops") is not True:
                    errors.append(f"EXP-286 {arm_id} unresolved episode must be retained as censored")
                if actual_cost > declared_ceiling:
                    errors.append(f"EXP-286 {arm_id} actual path exceeds common compute ceiling")
                episode_costs.append(float(expected_charged))
                solved_flags.append(solved)

            expected_rate = sum(float(flag) for flag in solved_flags) / len(solved_flags)
            expected_cost = sum(episode_costs) / len(episode_costs)
            if abs(float(metrics.get("verified_solution_rate", -1.0)) - expected_rate) > 1e-12:
                errors.append(f"EXP-286 {arm_id} verified solution rate drift")
            if abs(float(metrics.get(PRIMARY_METRIC, 0.0)) - expected_cost) > 1e-9:
                errors.append(f"EXP-286 {arm_id} primary cost aggregate drift")
            if int(metrics.get("contradiction_count", -1)) != total_contradictions:
                errors.append(f"EXP-286 {arm_id} contradiction-count receipt mismatch")
            if int(metrics.get("conflict_core_delivery_count", -1)) != total_deliveries:
                errors.append(f"EXP-286 {arm_id} conflict-core delivery-count receipt mismatch")
            if metrics.get("conflict_core_precontradiction_delivery") is not False:
                errors.append(f"EXP-286 {arm_id} claims forbidden pre-contradiction conflict-core delivery")
            if arm_id == "chronological_failure" and metrics.get("conflict_core_received") is not False:
                errors.append("EXP-286 chronological baseline conflict-core receipt drift")
            expected_all_solved = expected_rate == 1.0
            if metrics.get("censored_at_max_flops") is not (not expected_all_solved):
                errors.append(f"EXP-286 {arm_id} row censoring flag mismatch")
            if bool(metrics.get("external_solution_verified")) != expected_all_solved:
                errors.append(f"EXP-286 {arm_id} row external verification flag mismatch")

    aggregate = evaluation.get("aggregate") or {}
    if rows:
        expected_aggregate = _aggregate_rows(rows)
        if aggregate.keys() != expected_aggregate.keys():
            errors.append("EXP-286 descriptive aggregate fields drift")
        else:
            for key, expected_value in expected_aggregate.items():
                observed = aggregate.get(key)
                if isinstance(expected_value, float):
                    if abs(float(observed) - expected_value) > 1e-12:
                        errors.append(f"EXP-286 aggregate {key} mismatch")
                elif observed != expected_value:
                    errors.append(f"EXP-286 aggregate {key} mismatch")
    if aggregate.get("analysis_boundary") != ANALYSIS_BOUNDARY:
        errors.append("EXP-286 DEVELOPMENT analysis boundary drift")

    if payload.get("artifact_digest") not in (None, "") and payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-286 paired artifact digest mismatch")
    return errors


def run_exp286_paired_development(
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
    decoys: int,
    max_search_steps: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    if not root_seed or not protocol_digest or not code_digest:
        raise ValueError("root_seed, protocol_digest and code_digest are required")
    if min(
        d_model,
        hidden_size,
        target_parameters,
        train_replicates,
        eval_replicates,
        batch_size,
        timesteps,
        variables,
        max_search_steps,
    ) <= 0:
        raise ValueError("EXP-286 execution counts and dimensions must be positive")
    if variables < 2 or decoys < 0 or decoys > variables - 2:
        raise ValueError("EXP-286 conflict-world geometry is invalid")
    if eval_start_replicate < train_replicates:
        raise ValueError("training and evaluation replicate lineages must be disjoint")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("EXP-286 noise/optimizer parameters are invalid")

    chronological, oracle, model_init_seed = _build_seeded_pair(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
    )
    initial = {
        "chronological_failure_digest": _functional_state_digest(chronological),
        "oracle_conflict_core_digest": _functional_state_digest(oracle),
    }
    initial["functional_digest_match"] = len(set(initial.values())) == 1
    if initial["functional_digest_match"] is not True:
        raise RuntimeError("EXP-286 matched pair does not share identical functional initialization")

    pair_audit = audit_matched_exp286_arm_pair(
        chronological,
        oracle,
        timesteps=timesteps,
        variables=variables,
        max_search_steps=max_search_steps,
        max_accounted_flops_per_episode=max_accounted_flops_per_episode,
    )
    if not all(
        pair_audit.get(key) is True
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "oracle_information_separation",
            "compute_budget_closed",
        )
    ):
        raise RuntimeError("EXP-286 matched resource contract did not close before paired execution")

    ceiling = int(pair_audit["declared_max_accounted_flops_per_episode"])
    ledgers = pair_audit["compute_ledger"]
    per_step = {
        arm_id: int(ledgers[arm_id]["accounted_flops_per_search_step"])
        for arm_id in ARM_ORDER
    }

    generator = Exp286ConflictGenerator(root_seed=root_seed)
    optimizers = {
        "chronological_failure": build_functional_optimizer(
            chronological, lr=lr, weight_decay=weight_decay
        ),
        "oracle_conflict_core": build_functional_optimizer(
            oracle, lr=lr, weight_decay=weight_decay
        ),
    }
    arms: dict[str, ChronologicalFailureArm | OracleConflictCoreArm] = {
        "chronological_failure": chronological,
        "oracle_conflict_core": oracle,
    }

    training_digests: list[str] = []
    training_losses: dict[str, list[float]] = {arm_id: [] for arm_id in ARM_ORDER}
    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            variables=variables,
            decoys=decoys,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="augmentation",
        )
        training_digests.append(batch.digest)
        for arm_id in ARM_ORDER:
            training_losses[arm_id].append(
                _train_step(
                    arms[arm_id],
                    optimizers[arm_id],
                    arm_id=arm_id,
                    batch=batch,
                )
            )

    for arm in arms.values():
        arm.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(eval_replicates):
            replicate = eval_start_replicate + offset
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                decoys=decoys,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="evaluation",
            )
            row = {
                "replicate": replicate,
                "paired_batch_digest": batch.digest,
                "initial_world_pairing_closed": True,
                "future_conflict_core_leakage": False,
                "solution_leakage": False,
                "chronological_failure": _search_batch(
                    arm_id="chronological_failure",
                    arm=chronological,
                    batch=batch,
                    per_step_flops=per_step["chronological_failure"],
                    ceiling=ceiling,
                    max_search_steps=max_search_steps,
                ),
                "oracle_conflict_core": _search_batch(
                    arm_id="oracle_conflict_core",
                    arm=oracle,
                    batch=batch,
                    per_step_flops=per_step["oracle_conflict_core"],
                    ceiling=ceiling,
                    max_search_steps=max_search_steps,
                ),
            }
            rows.append(row)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "experiment_id": "EXP-286",
        "arm_order": list(ARM_ORDER),
        "model_init_seed": int(model_init_seed),
        "initial_state": initial,
        "information_receipt": deepcopy(_INFORMATION_RECEIPT),
        "primary_endpoint": {
            "metric": PRIMARY_METRIC,
            "direction": "lower",
            "mesi_relative_reduction": 0.15,
        },
        "protected_endpoints": {
            "verified_solution_rate_floor": PROTECTED_FLOOR,
        },
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "analysis_method_boundary": ANALYSIS_BOUNDARY,
        "execution_config": {
            "root_seed": root_seed,
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "decoys": decoys,
            "max_search_steps": max_search_steps,
            "noise_std": float(noise_std),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "resource_match": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "active_functional_parameter_match": True,
            "optimizer_visible_parameter_match": True,
            "same_initial_world_lineage": True,
            "compute_budget_closed": True,
            "oracle_information_separation": True,
            "declared_max_accounted_flops_per_episode": ceiling,
            "pair_audit": pair_audit,
            "pair_audit_digest": canonical_sha256(pair_audit),
        },
        "training": {
            "rng_stream": "augmentation",
            "start_replicate": 0,
            "replicates": train_replicates,
            "paired_batch_digests": training_digests,
            "mean_losses": {
                arm_id: sum(losses) / len(losses)
                for arm_id, losses in training_losses.items()
            },
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": eval_start_replicate,
            "replicates": eval_replicates,
            "per_replicate": rows,
            "aggregate": _aggregate_rows(rows),
        },
        "learned_conflict_localizer_validated": False,
        "remaining_blockers": [
            "confirmatory sample-size and paired log-cost/bootstrap analysis freeze remain open",
            "confirmatory-open execution remains unrun",
            "post-freeze challenge randomness remains unmaterialized",
            "learned ConflictCoreRegion localization remains unvalidated",
        ],
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    validation_errors = validate_exp286_paired_development(payload)
    if validation_errors:
        raise RuntimeError(
            "invalid EXP-286 paired development artifact: " + "; ".join(validation_errors)
        )
    return payload
