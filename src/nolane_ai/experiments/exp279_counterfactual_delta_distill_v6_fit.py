from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.seeds import derive_stream_seed

from .exp279_counterfactual_delta_distill_v6_primitives import (
    CDD,
    LinearSelector,
    cheap_features,
    class_counts,
    pairwise_ranking_loss,
    stop_and_branch_states,
    teacher_delta,
)
from .exp279_counterfactual_delta_distill_v6_utility import fold_direct_utility_metrics
from .matched_routing_arms import HybridRoutingArm


def _seeded_cdd(hidden_size: int, seed: int) -> CDD:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CDD(hidden_size)


def _seeded_selector(input_size: int, seed: int) -> LinearSelector:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return LinearSelector(input_size)


def _fit_selector(
    *,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    input_size: int,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[LinearSelector, dict[str, Any]]:
    counts = class_counts(labels)
    supported = counts["positive"] > 0 and counts["negative"] > 0
    selector = _seeded_selector(input_size, seed)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=0.0)
    initial_loss: float | None = None
    final_loss: float | None = None
    if supported:
        selector.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            scores = selector(inputs)
            loss = pairwise_ranking_loss(scores, labels)
            if step == 0:
                initial_loss = float(loss.detach().item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
    selector.eval()
    return selector, {
        "counts": counts,
        "supported": supported,
        "initial_pairwise_loss": initial_loss,
        "final_pairwise_loss": final_loss,
    }


def _exact_outcomes(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    stop_state, branch_state = stop_and_branch_states(
        arm,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )
    stop_exact = (arm.decision_head(stop_state).argmax(dim=-1) == targets).all(dim=-1)
    branch_exact = (arm.decision_head(branch_state).argmax(dim=-1) == targets).all(dim=-1)
    return stop_exact, branch_exact


def fit_cdd_fold(
    *,
    arm: HybridRoutingArm,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
    folds: int,
    hidden_size: int,
    distill_steps: int,
    distill_lr: float,
    selector_steps: int,
    selector_lr: float,
    probe_root_seed: str,
    stop_cost: float,
    branch_cost: float,
    cdd_cost: float,
    control_cost: float,
) -> dict[str, Any]:
    fit_mask = fold_ids != fold
    heldout_mask = fold_ids == fold

    fit_events = surface_events[fit_mask]
    fit_variables = variable_states[fit_mask]
    fit_incidence = incidence[fit_mask]
    fit_targets = targets[fit_mask]

    # Full branch teacher is created only on the fit partition.
    with torch.no_grad():
        fit_cheap, fit_p_mean = cheap_features(
            arm,
            surface_events=fit_events,
            variable_states=fit_variables,
            incidence=fit_incidence,
        )
        fit_stop_state, fit_branch_state = stop_and_branch_states(
            arm,
            surface_events=fit_events,
            variable_states=fit_variables,
            incidence=fit_incidence,
        )
        fit_teacher = teacher_delta(fit_stop_state, fit_branch_state)
        fit_stop_exact = (arm.decision_head(fit_stop_state).argmax(dim=-1) == fit_targets).all(dim=-1)
        fit_branch_exact = (arm.decision_head(fit_branch_state).argmax(dim=-1) == fit_targets).all(dim=-1)
        fit_labels = ((~fit_stop_exact) & fit_branch_exact).to(torch.long)

    cdd_seed = derive_stream_seed(probe_root_seed, "EXP-279-CDD-V6-DISTILLER", fold, "model_init")
    cdd = _seeded_cdd(hidden_size, cdd_seed)
    cdd_optimizer = torch.optim.AdamW(cdd.parameters(), lr=distill_lr, weight_decay=0.0)
    initial_mse: float | None = None
    final_mse: float | None = None
    cdd.train()
    for step in range(distill_steps):
        cdd_optimizer.zero_grad(set_to_none=True)
        prediction = cdd(fit_cheap)
        loss = F.mse_loss(prediction, fit_teacher)
        if step == 0:
            initial_mse = float(loss.detach().item())
        loss.backward()
        cdd_optimizer.step()
        final_mse = float(loss.detach().item())

    cdd.eval()
    for parameter in cdd.parameters():
        parameter.requires_grad_(False)
    frozen = not any(parameter.requires_grad for parameter in cdd.parameters())
    if not frozen:
        raise RuntimeError("V6 distiller did not freeze before selector")

    with torch.no_grad():
        fit_predicted_delta = cdd(fit_cheap)
    primary_inputs = torch.cat((fit_p_mean, fit_predicted_delta), dim=-1)
    primary_seed = derive_stream_seed(probe_root_seed, "EXP-279-CDD-V6-SELECTOR", fold, "model_init")
    primary_selector, primary_fit = _fit_selector(
        inputs=primary_inputs,
        labels=fit_labels,
        input_size=2 * hidden_size,
        steps=selector_steps,
        lr=selector_lr,
        seed=primary_seed,
    )
    control_seed = derive_stream_seed(probe_root_seed, "EXP-279-CDD-V6-RAW-CONTROL", fold, "model_init")
    control_selector, control_fit = _fit_selector(
        inputs=fit_cheap,
        labels=fit_labels,
        input_size=3 * hidden_size,
        steps=selector_steps,
        lr=selector_lr,
        seed=control_seed,
    )

    heldout_events = surface_events[heldout_mask]
    heldout_variables = variable_states[heldout_mask]
    heldout_incidence = incidence[heldout_mask]
    heldout_targets = targets[heldout_mask]

    # Freeze route scores before post-hoc held-out branch outcomes are materialized.
    with torch.no_grad():
        heldout_cheap, heldout_p_mean = cheap_features(
            arm,
            surface_events=heldout_events,
            variable_states=heldout_variables,
            incidence=heldout_incidence,
        )
        heldout_predicted_delta = cdd(heldout_cheap)
        primary_scores = primary_selector(torch.cat((heldout_p_mean, heldout_predicted_delta), dim=-1))
        control_scores = control_selector(heldout_cheap)
        heldout_stop_exact, heldout_branch_exact = _exact_outcomes(
            arm,
            surface_events=heldout_events,
            variable_states=heldout_variables,
            incidence=heldout_incidence,
            targets=heldout_targets,
        )

    primary_metrics = fold_direct_utility_metrics(
        primary_scores,
        heldout_stop_exact,
        heldout_branch_exact,
        stop_accounted_flops_per_episode=stop_cost,
        branch_accounted_flops_per_episode=branch_cost,
        inference_flops_per_episode=cdd_cost,
    )
    control_metrics = fold_direct_utility_metrics(
        control_scores,
        heldout_stop_exact,
        heldout_branch_exact,
        stop_accounted_flops_per_episode=stop_cost,
        branch_accounted_flops_per_episode=branch_cost,
        inference_flops_per_episode=control_cost,
    )
    primary_heldout_supported = bool(primary_metrics["support_closed"])
    primary_metrics["support_closed"] = bool(primary_fit["supported"] and primary_heldout_supported)
    primary_metrics["direct_utility_improved"] = bool(
        primary_metrics["support_closed"]
        and float(primary_metrics["cdd_routed_verified_utility"]) > float(primary_metrics["stop_verified_utility"])
    )
    control_heldout_supported = bool(control_metrics["support_closed"])
    control_metrics["support_closed"] = bool(control_fit["supported"] and control_heldout_supported)
    control_metrics["direct_utility_improved"] = bool(
        control_metrics["support_closed"]
        and float(control_metrics["cdd_routed_verified_utility"]) > float(control_metrics["stop_verified_utility"])
    )

    return {
        "fold": fold,
        "folds": folds,
        "distiller_seed": cdd_seed,
        "primary_selector_seed": primary_seed,
        "control_selector_seed": control_seed,
        "distiller_initial_mse": initial_mse,
        "distiller_final_mse": final_mse,
        "distiller_frozen_before_selector": frozen,
        "teacher_uses_fit_partition_only": True,
        "heldout_scores_frozen_before_outcomes": True,
        "heldout_branch_hidden_used_for_features": False,
        "primary_fit": primary_fit,
        "control_fit": control_fit,
        "primary_heldout_supported": primary_heldout_supported,
        "control_heldout_supported": control_heldout_supported,
        "primary": primary_metrics,
        "control": control_metrics,
    }
