from __future__ import annotations

from typing import Any

SCHEMA = "NLM-EXP-279-ROUTING-MARGINAL-DEV-V1"


def _forced_stop_output(
    arm: Any,
    *,
    surface_events: Any,
    variable_states: Any,
    incidence: Any,
) -> Any:
    """Evaluate the trained hybrid's stop path without executing branch recurrence."""
    import torch

    _events, variables = arm._validate_common(surface_events, variable_states)
    checked_incidence = arm._validate_incidence(incidence, variables)
    propagated = arm._propagate(variables, checked_incidence)
    propagation_state = variables + propagated
    residual = arm._residual_uncertainty(propagation_state)
    reclaimed = torch.tanh(arm.reclaimed_projection(propagation_state))
    stop_state = propagation_state + reclaimed
    route_mask = torch.zeros(
        stop_state.shape[0], dtype=torch.bool, device=stop_state.device
    )
    return arm._heads(
        stop_state,
        residual_uncertainty=residual,
        route_mask=route_mask,
        semantics="hybrid_forced_stop_shadow_same_weights_no_branch_execution",
    )


def _routing_outcome_counts(
    *,
    stop_logits: Any,
    actual_logits: Any,
    targets: Any,
    route_mask: Any,
) -> dict[str, int]:
    import torch

    if stop_logits.shape != actual_logits.shape:
        raise ValueError("stop and actual logits must have identical shape")
    if stop_logits.ndim != 3 or stop_logits.shape[-1] != 2:
        raise ValueError("routing marginal logits must be [batch,variables,2]")
    if tuple(targets.shape) != tuple(stop_logits.shape[:-1]):
        raise ValueError("routing marginal target shape mismatch")
    if route_mask.ndim != 1 or route_mask.shape[0] != stop_logits.shape[0]:
        raise ValueError("routing marginal route mask must be [batch]")

    route_mask = route_mask.to(dtype=torch.bool)
    stop_exact = (stop_logits.argmax(dim=-1) == targets).all(dim=-1)
    actual_exact = (actual_logits.argmax(dim=-1) == targets).all(dim=-1)
    routed = route_mask
    routed_rescues = routed & (~stop_exact) & actual_exact
    routed_harms = routed & stop_exact & (~actual_exact)
    routed_success_unchanged = routed & stop_exact & actual_exact
    routed_failure_unchanged = routed & (~stop_exact) & (~actual_exact)

    def _count(mask: Any) -> int:
        return int(mask.to(dtype=torch.int64).sum().item())

    routed_count = _count(routed)
    rescues = _count(routed_rescues)
    harms = _count(routed_harms)
    success_unchanged = _count(routed_success_unchanged)
    failure_unchanged = _count(routed_failure_unchanged)
    if rescues + harms + success_unchanged + failure_unchanged != routed_count:
        raise RuntimeError("EXP-279 routing outcome partition did not close")

    episodes = int(stop_logits.shape[0])
    return {
        "episodes": episodes,
        "routed_episodes": routed_count,
        "unrouted_episodes": episodes - routed_count,
        "routed_rescues": rescues,
        "routed_harms": harms,
        "routed_success_unchanged": success_unchanged,
        "routed_failure_unchanged": failure_unchanged,
        "net_routed_exact_delta": rescues - harms,
    }


def _routing_classification(
    *, total_routed_episodes: int, mean_actual_utility: float, mean_stop_utility: float
) -> str:
    if total_routed_episodes == 0:
        return "ROUTING_INACTIVE"
    tolerance = 1e-12 * max(1.0, abs(mean_actual_utility), abs(mean_stop_utility))
    delta = mean_actual_utility - mean_stop_utility
    if delta > tolerance:
        return "ROUTING_ADDS_POSITIVE_UTILITY"
    if delta < -tolerance:
        return "ROUTING_HARMS_UTILITY"
    return "ROUTING_UTILITY_NEUTRAL"


def run_exp279_routing_marginal_development(
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
        PRIMARY_METRIC,
        _build_seeded_triplet,
        _external_metrics,
        _functional_state_digest,
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
        raise ValueError("EXP-279 routing marginal counts and dimensions must be positive")
    if eval_replicates < len(STRATA):
        raise ValueError("EXP-279 routing marginal evaluation requires all predeclared strata")
    if constraints > variables or timesteps < constraints:
        raise ValueError("EXP-279 routing marginal world geometry is invalid")
    if eval_start_replicate < train_replicates:
        raise ValueError("training and evaluation replicate lineages must be disjoint")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0, 1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("EXP-279 routing marginal noise/optimizer parameters are invalid")

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
        raise RuntimeError("EXP-279 routing marginal matched resource contract did not close")

    hybrid_ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_cost = float(hybrid_ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(hybrid_ledger["branch_accounted_flops_per_episode"])
    if stop_cost <= 0.0 or branch_cost < stop_cost:
        raise RuntimeError("EXP-279 routing marginal hybrid compute ledger is invalid")

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
            actual = hybrid(
                batch.surface_events, batch.variable_states, batch.incidence
            )
            forced_stop = _forced_stop_output(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            route_mask = actual.branch_route_mask.to(dtype=torch.bool)
            unrouted = (~route_mask).nonzero(as_tuple=False).squeeze(-1)
            if unrouted.numel() > 0:
                actual_unrouted = actual.decision_logits.index_select(0, unrouted)
                stop_unrouted = forced_stop.decision_logits.index_select(0, unrouted)
                if not torch.equal(actual_unrouted, stop_unrouted):
                    raise RuntimeError(
                        "EXP-279 routing marginal unrouted output diverged from forced-stop shadow"
                    )

            routed = int(route_mask.to(dtype=torch.int64).sum().item())
            route_fraction = routed / batch_size
            actual_cost = stop_cost + route_fraction * (branch_cost - stop_cost)
            actual_metrics = _external_metrics(
                actual,
                batch.targets,
                accounted_flops_per_episode=actual_cost,
            )
            stop_metrics = _external_metrics(
                forced_stop,
                batch.targets,
                accounted_flops_per_episode=stop_cost,
            )
            outcome_counts = _routing_outcome_counts(
                stop_logits=forced_stop.decision_logits,
                actual_logits=actual.decision_logits,
                targets=batch.targets,
                route_mask=route_mask,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "paired_batch_digest": batch.digest,
                    "route_fraction": route_fraction,
                    "routed_episodes": routed,
                    "stop_accounted_flops_per_episode": stop_cost,
                    "branch_accounted_flops_per_episode": branch_cost,
                    "actual_charged_accounted_flops_per_episode": actual_cost,
                    "mean_residual_uncertainty": float(
                        actual.residual_uncertainty.mean().item()
                    ),
                    "routing_outcomes": outcome_counts,
                    "actual_hybrid": actual_metrics,
                    "forced_stop_shadow": stop_metrics,
                }
            )

    n = len(rows)
    mean_actual_solution = sum(
        float(row["actual_hybrid"]["verified_solution_rate"]) for row in rows
    ) / n
    mean_stop_solution = sum(
        float(row["forced_stop_shadow"]["verified_solution_rate"]) for row in rows
    ) / n
    mean_actual_utility = sum(
        float(row["actual_hybrid"][PRIMARY_METRIC]) for row in rows
    ) / n
    mean_stop_utility = sum(
        float(row["forced_stop_shadow"][PRIMARY_METRIC]) for row in rows
    ) / n
    total_routed = sum(int(row["routing_outcomes"]["routed_episodes"]) for row in rows)
    total_rescues = sum(int(row["routing_outcomes"]["routed_rescues"]) for row in rows)
    total_harms = sum(int(row["routing_outcomes"]["routed_harms"]) for row in rows)
    total_success_unchanged = sum(
        int(row["routing_outcomes"]["routed_success_unchanged"]) for row in rows
    )
    total_failure_unchanged = sum(
        int(row["routing_outcomes"]["routed_failure_unchanged"]) for row in rows
    )
    total_episodes = n * batch_size
    if total_rescues + total_harms + total_success_unchanged + total_failure_unchanged != total_routed:
        raise RuntimeError("EXP-279 routing marginal aggregate routed partition did not close")

    marginal_utility_gain = (
        mean_actual_utility - mean_stop_utility
    ) / max(abs(mean_stop_utility), 1e-12)
    aggregate = {
        "evaluation_replicates": n,
        "total_episodes": total_episodes,
        "total_routed_episodes": total_routed,
        "route_fraction": total_routed / total_episodes,
        "routed_rescues": total_rescues,
        "routed_harms": total_harms,
        "routed_success_unchanged": total_success_unchanged,
        "routed_failure_unchanged": total_failure_unchanged,
        "net_routed_exact_delta": total_rescues - total_harms,
        "mean_actual_solution_rate": mean_actual_solution,
        "mean_forced_stop_solution_rate": mean_stop_solution,
        "routing_marginal_solution_rate_delta": mean_actual_solution - mean_stop_solution,
        "mean_actual_utility": mean_actual_utility,
        "mean_forced_stop_utility": mean_stop_utility,
        "routing_marginal_utility_delta": mean_actual_utility - mean_stop_utility,
        "routing_marginal_utility_gain": marginal_utility_gain,
        "routing_classification": _routing_classification(
            total_routed_episodes=total_routed,
            mean_actual_utility=mean_actual_utility,
            mean_stop_utility=mean_stop_utility,
        ),
    }

    return {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "analysis_scope": "development_same_weights_actual_routing_vs_forced_stop_shadow",
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
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
    }
