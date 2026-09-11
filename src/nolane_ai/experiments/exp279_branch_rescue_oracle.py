from __future__ import annotations

from collections import defaultdict
from typing import Any

SCHEMA = "NLM-EXP-279-BRANCH-RESCUE-ORACLE-DEV-V1"
PRIMARY_METRIC = "verified_utility_per_accounted_flop_on_structure_dense_stratum"


def _branch_outcome_counts(
    *,
    stop_logits: Any,
    branch_logits: Any,
    targets: Any,
) -> dict[str, int]:
    import torch

    if stop_logits.shape != branch_logits.shape:
        raise ValueError("stop and forced-branch logits must have identical shape")
    if stop_logits.ndim != 3 or stop_logits.shape[-1] != 2:
        raise ValueError("branch rescue logits must be [batch,variables,2]")
    if tuple(targets.shape) != tuple(stop_logits.shape[:-1]):
        raise ValueError("branch rescue target shape mismatch")

    stop_exact = (stop_logits.argmax(dim=-1) == targets).all(dim=-1)
    branch_exact = (branch_logits.argmax(dim=-1) == targets).all(dim=-1)
    rescue = (~stop_exact) & branch_exact
    harm = stop_exact & (~branch_exact)
    both_success = stop_exact & branch_exact
    both_failure = (~stop_exact) & (~branch_exact)

    def _count(mask: Any) -> int:
        return int(mask.to(dtype=torch.int64).sum().item())

    episodes = int(stop_logits.shape[0])
    counts = {
        "episodes": episodes,
        "stop_exact_successes": _count(stop_exact),
        "forced_branch_exact_successes": _count(branch_exact),
        "branch_rescues": _count(rescue),
        "branch_harms": _count(harm),
        "both_success": _count(both_success),
        "both_failure": _count(both_failure),
    }
    if (
        counts["branch_rescues"]
        + counts["branch_harms"]
        + counts["both_success"]
        + counts["both_failure"]
        != episodes
    ):
        raise RuntimeError("EXP-279 branch outcome partition did not close")
    if counts["stop_exact_successes"] != counts["branch_harms"] + counts["both_success"]:
        raise RuntimeError("EXP-279 stop success partition did not close")
    if counts["forced_branch_exact_successes"] != counts["branch_rescues"] + counts["both_success"]:
        raise RuntimeError("EXP-279 branch success partition did not close")
    return counts


def _rescue_oracle_batch_metrics(
    *,
    stop_logits: Any,
    branch_logits: Any,
    targets: Any,
    stop_accounted_flops_per_episode: float,
    branch_accounted_flops_per_episode: float,
) -> dict[str, float | int]:
    stop_cost = float(stop_accounted_flops_per_episode)
    branch_cost = float(branch_accounted_flops_per_episode)
    if stop_cost <= 0.0 or branch_cost < stop_cost:
        raise ValueError("branch rescue oracle requires positive branch cost >= stop cost")

    counts = _branch_outcome_counts(
        stop_logits=stop_logits,
        branch_logits=branch_logits,
        targets=targets,
    )
    episodes = counts["episodes"]
    stop_rate = counts["stop_exact_successes"] / episodes
    branch_rate = counts["forced_branch_exact_successes"] / episodes
    rescue_route_fraction = counts["branch_rescues"] / episodes
    rescue_oracle_rate = (
        counts["stop_exact_successes"] + counts["branch_rescues"]
    ) / episodes
    rescue_oracle_cost = stop_cost + rescue_route_fraction * (branch_cost - stop_cost)

    stop_utility = stop_rate / stop_cost
    branch_utility = branch_rate / branch_cost
    rescue_oracle_utility = rescue_oracle_rate / rescue_oracle_cost
    denominator = max(abs(stop_utility), 1e-12)
    rescue_oracle_gain = (rescue_oracle_utility - stop_utility) / denominator

    return {
        **counts,
        "stop_accounted_flops_per_episode": stop_cost,
        "branch_accounted_flops_per_episode": branch_cost,
        "stop_solution_rate": stop_rate,
        "forced_branch_solution_rate": branch_rate,
        "stop_utility": stop_utility,
        "always_branch_utility": branch_utility,
        "rescue_oracle_route_fraction": rescue_route_fraction,
        "rescue_oracle_solution_rate": rescue_oracle_rate,
        "rescue_oracle_accounted_flops_per_episode": rescue_oracle_cost,
        "rescue_oracle_utility": rescue_oracle_utility,
        "rescue_oracle_relative_utility_gain": rescue_oracle_gain,
    }


def _classification(*, rescues: int, mean_stop_utility: float, mean_oracle_utility: float) -> str:
    if rescues == 0:
        return "NO_BRANCH_RESCUE_CAPACITY"
    tolerance = 1e-12 * max(1.0, abs(mean_stop_utility), abs(mean_oracle_utility))
    if mean_oracle_utility <= mean_stop_utility + tolerance:
        return "RESCUES_EXIST_BUT_ORACLE_NOT_COST_EFFECTIVE"
    return "BRANCH_RESCUE_CAPACITY_COST_EFFECTIVE"


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("branch rescue summary requires rows")
    n = len(rows)
    total_episodes = sum(int(row["metrics"]["episodes"]) for row in rows)
    total_stop_successes = sum(int(row["metrics"]["stop_exact_successes"]) for row in rows)
    total_branch_successes = sum(
        int(row["metrics"]["forced_branch_exact_successes"]) for row in rows
    )
    total_rescues = sum(int(row["metrics"]["branch_rescues"]) for row in rows)
    total_harms = sum(int(row["metrics"]["branch_harms"]) for row in rows)
    total_both_success = sum(int(row["metrics"]["both_success"]) for row in rows)
    total_both_failure = sum(int(row["metrics"]["both_failure"]) for row in rows)
    if total_rescues + total_harms + total_both_success + total_both_failure != total_episodes:
        raise RuntimeError("EXP-279 aggregate branch outcome partition did not close")

    mean_stop_utility = sum(float(row["metrics"]["stop_utility"]) for row in rows) / n
    mean_branch_utility = sum(float(row["metrics"]["always_branch_utility"]) for row in rows) / n
    mean_oracle_utility = sum(float(row["metrics"]["rescue_oracle_utility"]) for row in rows) / n
    mean_stop_solution = sum(float(row["metrics"]["stop_solution_rate"]) for row in rows) / n
    mean_branch_solution = sum(
        float(row["metrics"]["forced_branch_solution_rate"]) for row in rows
    ) / n
    mean_oracle_solution = sum(
        float(row["metrics"]["rescue_oracle_solution_rate"]) for row in rows
    ) / n
    mean_oracle_cost = sum(
        float(row["metrics"]["rescue_oracle_accounted_flops_per_episode"]) for row in rows
    ) / n
    denominator = max(abs(mean_stop_utility), 1e-12)
    oracle_gain = (mean_oracle_utility - mean_stop_utility) / denominator

    return {
        "evaluation_replicates": n,
        "total_episodes": total_episodes,
        "stop_exact_successes": total_stop_successes,
        "forced_branch_exact_successes": total_branch_successes,
        "branch_rescues": total_rescues,
        "branch_harms": total_harms,
        "both_success": total_both_success,
        "both_failure": total_both_failure,
        "rescue_rate_among_all_episodes": total_rescues / total_episodes,
        "rescue_rate_among_stop_failures": (
            total_rescues / max(total_episodes - total_stop_successes, 1)
        ),
        "mean_stop_solution_rate": mean_stop_solution,
        "mean_forced_branch_solution_rate": mean_branch_solution,
        "mean_rescue_oracle_solution_rate": mean_oracle_solution,
        "mean_stop_utility": mean_stop_utility,
        "mean_always_branch_utility": mean_branch_utility,
        "mean_rescue_oracle_utility": mean_oracle_utility,
        "mean_rescue_oracle_accounted_flops_per_episode": mean_oracle_cost,
        "rescue_oracle_relative_utility_gain": oracle_gain,
        "branch_rescue_classification": _classification(
            rescues=total_rescues,
            mean_stop_utility=mean_stop_utility,
            mean_oracle_utility=mean_oracle_utility,
        ),
    }


def run_exp279_branch_rescue_oracle_development(
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
    import torch

    from nolane_ai.training.optimizer import build_functional_optimizer

    from .exp279_paired_runner import (
        _build_seeded_triplet,
        _functional_state_digest,
        _hybrid_forced_branch_decision_logits,
        _hybrid_stop_decision_logits,
        _train_step,
    )
    from .exp279_routing_worlds import STRATA, Exp279RoutingGenerator
    from .matched_routing_arms import audit_matched_exp279_arm_triplet

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
        constraints,
    ) <= 0:
        raise ValueError("EXP-279 branch rescue counts and dimensions must be positive")
    if eval_replicates < len(STRATA):
        raise ValueError("EXP-279 branch rescue evaluation requires all predeclared strata")
    if constraints > variables or timesteps < constraints:
        raise ValueError("EXP-279 branch rescue world geometry is invalid")
    if eval_start_replicate < train_replicates:
        raise ValueError("training and evaluation replicate lineages must be disjoint")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0, 1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("EXP-279 branch rescue noise/optimizer parameters are invalid")

    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=route_threshold,
    )
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
        max_accounted_flops_per_episode=max_accounted_flops_per_episode,
    )
    if not all(
        pair_audit.get(key) is True
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "reclaimed_parameter_assignment_closed",
            "compute_budget_closed",
        )
    ):
        raise RuntimeError("EXP-279 branch rescue matched resource contract did not close")

    hybrid_ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_cost = float(hybrid_ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(hybrid_ledger["branch_accounted_flops_per_episode"])
    if stop_cost <= 0.0 or branch_cost < stop_cost:
        raise RuntimeError("EXP-279 branch rescue hybrid compute ledger is invalid")

    generator = Exp279RoutingGenerator(root_seed=root_seed)
    optimizer = build_functional_optimizer(hybrid, lr=lr, weight_decay=weight_decay)
    training_batch_digests: list[str] = []
    training_losses: list[float] = []
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
        training_batch_digests.append(batch.digest)
        training_losses.append(
            _train_step(
                hybrid,
                optimizer,
                arm_id="hybrid",
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
                targets=batch.targets,
            )
        )

    final_hybrid_digest = _functional_state_digest(hybrid)
    hybrid.eval()
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
            stop_logits = _hybrid_stop_decision_logits(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            branch_logits = _hybrid_forced_branch_decision_logits(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            metrics = _rescue_oracle_batch_metrics(
                stop_logits=stop_logits,
                branch_logits=branch_logits,
                targets=batch.targets,
                stop_accounted_flops_per_episode=stop_cost,
                branch_accounted_flops_per_episode=branch_cost,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "paired_batch_digest": batch.digest,
                    "metrics": metrics,
                }
            )

    aggregate = _summarize_rows(rows)
    stratum_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum_rows[str(row["stratum"])].append(row)
    by_stratum = {
        stratum: _summarize_rows(stratum_rows[stratum])
        for stratum in STRATA
        if stratum_rows[stratum]
    }

    return {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "analysis_scope": "development_same_weights_forced_branch_rescue_upper_bound",
        "oracle_is_deployable": False,
        "oracle_uses_evaluation_targets": True,
        "evaluation_targets_used_for_training": False,
        "evaluation_targets_used_for_deployable_routing": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "route_threshold": float(route_threshold),
        "training_replicates": train_replicates,
        "training_batch_digests": training_batch_digests,
        "training_losses": training_losses,
        "evaluation_start_replicate": eval_start_replicate,
        "evaluation_replicates": eval_replicates,
        "world_geometry": {
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "noise_std": float(noise_std),
        },
        "arm_geometry": {
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
        },
        "final_hybrid_digest": final_hybrid_digest,
        "resource_pair_audit_digest": pair_audit.get("pair_audit_digest"),
        "hybrid_compute_ledger": {
            "stop_accounted_flops_per_episode": stop_cost,
            "branch_accounted_flops_per_episode": branch_cost,
        },
        "rows": rows,
        "aggregate": aggregate,
        "by_stratum": by_stratum,
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
    }
