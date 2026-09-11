from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch
from torch.nn import functional as F

from . import exp279_paired_runner as _base
from .matched_routing_arms import HybridRoutingArm, _compute_ledgers


ROUTING_SUPERVISION = deepcopy(_base.ROUTING_SUPERVISION)
ROUTING_SUPERVISION["hybrid_route_teacher"] = {
    "positive": "stop_exact_failure_and_forced_branch_exact_success",
    "negative": "otherwise",
    "evaluation_targets_used_for_routing": False,
    "decision_threshold_changed": False,
    "prior_corrected_rescue_replay": {
        "enabled": True,
        "source": "augmentation_training_only",
        "positive_replay": "detached_all_prior_rescue_routing_states",
        "natural_prior": "causal_cumulative_raw_augmentation_rescue_frequency",
        "sampling_correction": "exact_case_control_odds_correction",
        "economics": "incremental_branch_flops_divided_by_branch_flops",
        "negative_only_before_first_rescue": "zero_routing_gradient_but_count_in_natural_prior",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
        "tunable_calibration_hyperparameters": False,
    },
}


def _validate_probability_tensor(value: torch.Tensor, *, name: str) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} must lie within [0, 1]")


def _validate_incremental_cost_ratio(value: float) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise ValueError("incremental branch cost ratio must lie strictly within (0, 1)")
    return ratio


def _economic_rescue_score_from_probability(
    natural_rescue_probability: torch.Tensor,
    *,
    incremental_cost_ratio: float,
) -> torch.Tensor:
    """Map a natural rescue posterior to the frozen-threshold economic score.

    For c=(F_branch-F_stop)/F_branch, score > 0.5 exactly when
    q > c/(1+c). No learned/tuned calibration scalar is introduced.
    """

    _validate_probability_tensor(natural_rescue_probability, name="natural rescue probability")
    cost_ratio = _validate_incremental_cost_ratio(incremental_cost_ratio)
    numerator = natural_rescue_probability
    denominator = numerator + cost_ratio * (1.0 - numerator)
    return torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def _natural_rescue_probability_from_score(
    economic_score: torch.Tensor,
    *,
    incremental_cost_ratio: float,
) -> torch.Tensor:
    """Invert `_economic_rescue_score_from_probability` exactly."""

    _validate_probability_tensor(economic_score, name="economic routing score")
    cost_ratio = _validate_incremental_cost_ratio(incremental_cost_ratio)
    numerator = cost_ratio * economic_score
    denominator = 1.0 - economic_score + numerator
    return torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.ones_like(numerator),
    )


def _case_control_sample_probability(
    natural_probability: torch.Tensor,
    *,
    natural_positive_fraction: float,
    sampled_positive_fraction: float,
) -> torch.Tensor:
    """Map a natural posterior into the posterior under case-control sampling.

    The odds multiplier is determined exactly by sampled-vs-natural class odds.
    A model can therefore be optimized on a positive-enriched sample without
    silently changing the population posterior represented by its output.
    """

    _validate_probability_tensor(natural_probability, name="natural probability")
    natural = float(natural_positive_fraction)
    sampled = float(sampled_positive_fraction)
    if not 0.0 < natural < 1.0:
        raise ValueError("natural positive fraction must lie strictly within (0, 1)")
    if not 0.0 < sampled < 1.0:
        raise ValueError("sampled positive fraction must lie strictly within (0, 1)")

    natural_odds = natural / (1.0 - natural)
    sampled_odds = sampled / (1.0 - sampled)
    odds_multiplier = sampled_odds / natural_odds
    numerator = odds_multiplier * natural_probability
    denominator = (1.0 - natural_probability) + numerator
    return numerator / denominator


def _prior_corrected_replay_loss(
    current_economic_scores: torch.Tensor,
    current_rescue_targets: torch.Tensor,
    replay_positive_economic_scores: torch.Tensor,
    *,
    natural_positive_count: int,
    natural_total_count: int,
    stop_accounted_flops_per_episode: float,
    branch_accounted_flops_per_episode: float,
) -> torch.Tensor:
    if current_economic_scores.shape != current_rescue_targets.shape:
        raise ValueError("current routing scores and rescue targets must have identical shape")
    if current_economic_scores.ndim != 1 or replay_positive_economic_scores.ndim != 1:
        raise ValueError("prior-corrected routing scores must be rank-1")
    if natural_total_count <= 0:
        raise ValueError("natural total count must be positive")
    if natural_positive_count < 0 or natural_positive_count > natural_total_count:
        raise ValueError("natural positive count is invalid")
    stop_flops = float(stop_accounted_flops_per_episode)
    branch_flops = float(branch_accounted_flops_per_episode)
    if stop_flops <= 0.0 or branch_flops <= stop_flops:
        raise ValueError("branch FLOPs must be greater than positive stop FLOPs")

    # Before the first observed rescue there is no finite positive case-control
    # stratum. We still count all raw negatives in the causal natural prior, but
    # avoid repeatedly pushing the router into the all-stop optimization basin.
    if natural_positive_count == 0:
        return current_economic_scores.sum() * 0.0 + replay_positive_economic_scores.sum() * 0.0
    if natural_positive_count == natural_total_count:
        return current_economic_scores.sum() * 0.0 + replay_positive_economic_scores.sum() * 0.0

    sampled_scores = torch.cat((current_economic_scores, replay_positive_economic_scores))
    sampled_targets = torch.cat(
        (
            current_rescue_targets.to(dtype=current_economic_scores.dtype),
            torch.ones_like(replay_positive_economic_scores),
        )
    )
    sampled_positive_count = int((sampled_targets > 0.5).sum().item())
    sampled_total_count = int(sampled_targets.numel())
    sampled_negative_count = sampled_total_count - sampled_positive_count
    if sampled_positive_count == 0 or sampled_negative_count == 0:
        return sampled_scores.sum() * 0.0

    natural_positive_fraction = natural_positive_count / natural_total_count
    sampled_positive_fraction = sampled_positive_count / sampled_total_count
    incremental_cost_ratio = (branch_flops - stop_flops) / branch_flops

    natural_probability = _natural_rescue_probability_from_score(
        sampled_scores,
        incremental_cost_ratio=incremental_cost_ratio,
    )
    sampled_probability = _case_control_sample_probability(
        natural_probability,
        natural_positive_fraction=natural_positive_fraction,
        sampled_positive_fraction=sampled_positive_fraction,
    )
    epsilon = torch.finfo(sampled_probability.dtype).eps
    sampled_probability = sampled_probability.clamp(epsilon, 1.0 - epsilon)
    return F.binary_cross_entropy(sampled_probability, sampled_targets)


def _append_rescue_anchor_states(
    anchors: list[torch.Tensor],
    routing_states: torch.Tensor,
    rescue_targets: torch.Tensor,
) -> None:
    if routing_states.ndim != 3:
        raise ValueError("routing states must be [batch,variables,hidden]")
    if rescue_targets.ndim != 1 or rescue_targets.shape[0] != routing_states.shape[0]:
        raise ValueError("rescue targets must align with routing-state batch dimension")
    positive_mask = rescue_targets > 0.5
    if bool(positive_mask.any().item()):
        anchors.append(routing_states.detach()[positive_mask].clone())


def _hybrid_routing_state(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    _, variables = arm._validate_common(surface_events, variable_states)
    checked_incidence = arm._validate_incidence(incidence, variables)
    propagated = arm._propagate(variables, checked_incidence)
    return variables + propagated


def _train_hybrid_with_prior_corrected_replay(
    arm: HybridRoutingArm,
    optimizer: torch.optim.Optimizer,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
    rescue_anchor_states: list[torch.Tensor],
    natural_counts: dict[str, int],
) -> float:
    arm.train()
    optimizer.zero_grad(set_to_none=True)
    output = arm(surface_events, variable_states, incidence)

    final_decision_loss = F.cross_entropy(
        output.decision_logits.reshape(-1, 2),
        targets.reshape(-1),
    )
    stop_logits = _base._hybrid_stop_decision_logits(
        arm,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )
    forced_branch_logits = _base._hybrid_forced_branch_decision_logits(
        arm,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )

    stop_path = ROUTING_SUPERVISION["hybrid_stop_path_supervision"]
    stop_decision_loss = F.cross_entropy(
        stop_logits.reshape(-1, 2),
        targets.reshape(-1),
    )
    forced_branch = ROUTING_SUPERVISION["hybrid_forced_branch_supervision"]
    forced_branch_decision_loss = F.cross_entropy(
        forced_branch_logits.reshape(-1, 2),
        targets.reshape(-1),
    )
    decision_loss = (
        float(stop_path["final_path_weight"]) * final_decision_loss
        + float(stop_path["stop_path_weight"]) * stop_decision_loss
        + float(forced_branch["weight"]) * forced_branch_decision_loss
    )

    rescue_targets = _base._hybrid_branch_rescue_target(
        stop_logits,
        forced_branch_logits,
        targets,
    )
    raw_positive_count = int((rescue_targets > 0.5).sum().item())
    natural_counts["positive"] += raw_positive_count
    natural_counts["total"] += int(rescue_targets.numel())

    routing_state = _hybrid_routing_state(
        arm,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )
    if rescue_anchor_states:
        replay_positive_scores = torch.cat(
            [arm._residual_uncertainty(anchor) for anchor in rescue_anchor_states]
        )
    else:
        replay_positive_scores = output.residual_uncertainty.new_empty((0,))

    ledgers = _compute_ledgers(
        arm,
        timesteps=int(surface_events.shape[1]),
        variables=int(variable_states.shape[1]),
        constraints=int(incidence.shape[1]),
    )
    hybrid_ledger = ledgers["hybrid"]
    routing_loss = _prior_corrected_replay_loss(
        output.residual_uncertainty,
        rescue_targets,
        replay_positive_scores,
        natural_positive_count=natural_counts["positive"],
        natural_total_count=natural_counts["total"],
        stop_accounted_flops_per_episode=float(hybrid_ledger["stop_accounted_flops_per_episode"]),
        branch_accounted_flops_per_episode=float(hybrid_ledger["branch_accounted_flops_per_episode"]),
    )
    routing_weight = float(ROUTING_SUPERVISION["weight"])
    routing_parameters = tuple(arm.routing_head.parameters())
    routing_gradients = torch.autograd.grad(
        routing_weight * routing_loss,
        routing_parameters,
        retain_graph=True,
    )

    decision_loss.backward()
    for parameter, routing_gradient in zip(routing_parameters, routing_gradients):
        if parameter.grad is None:
            parameter.grad = routing_gradient.detach().clone()
        else:
            parameter.grad.add_(routing_gradient)
    optimizer.step()

    _append_rescue_anchor_states(rescue_anchor_states, routing_state, rescue_targets)
    loss_value = decision_loss.detach() + routing_weight * routing_loss.detach()
    return float(loss_value.item())


def _make_train_step(
    rescue_anchor_states: list[torch.Tensor],
    natural_counts: dict[str, int],
    original_train_step: Callable[..., float],
) -> Callable[..., float]:
    def train_step(
        arm: Any,
        optimizer: torch.optim.Optimizer,
        *,
        arm_id: str,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        if arm_id != "hybrid":
            return original_train_step(
                arm,
                optimizer,
                arm_id=arm_id,
                surface_events=surface_events,
                variable_states=variable_states,
                incidence=incidence,
                targets=targets,
            )
        return _train_hybrid_with_prior_corrected_replay(
            arm,
            optimizer,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
            targets=targets,
            rescue_anchor_states=rescue_anchor_states,
            natural_counts=natural_counts,
        )

    return train_step


def _anchor_episode_count(anchors: list[torch.Tensor]) -> int:
    return sum(int(anchor.shape[0]) for anchor in anchors)


def validate_exp279_prior_corrected_rescue_development(payload: dict[str, Any]) -> list[str]:
    original_supervision = _base.ROUTING_SUPERVISION
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    try:
        errors = _base.validate_exp279_paired_development(payload)
    finally:
        _base.ROUTING_SUPERVISION = original_supervision

    replay = (payload.get("training") or {}).get("prior_corrected_rescue_replay") or {}
    if replay.get("source") != "augmentation_training_only":
        errors.append("prior-corrected rescue replay source must remain augmentation training only")
    for key in ("final_anchor_episodes", "anchor_batches", "natural_positive_count", "natural_total_count"):
        value = replay.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"prior-corrected rescue replay {key} is invalid")
    if int(replay.get("natural_positive_count", 0)) > int(replay.get("natural_total_count", 0)):
        errors.append("prior-corrected rescue natural counts are inconsistent")
    if replay.get("evaluation_examples_used") is not False:
        errors.append("evaluation examples cannot enter prior-corrected rescue replay")
    if replay.get("evaluation_targets_used") is not False:
        errors.append("evaluation targets cannot enter prior-corrected rescue replay")
    if replay.get("decision_threshold_changed") is not False:
        errors.append("prior-corrected rescue replay cannot change the frozen route threshold")
    if replay.get("tunable_calibration_hyperparameters") is not False:
        errors.append("prior-corrected rescue replay cannot add tunable calibration hyperparameters")
    stop_flops = replay.get("stop_accounted_flops_per_episode")
    branch_flops = replay.get("branch_accounted_flops_per_episode")
    ratio = replay.get("incremental_cost_ratio")
    if not isinstance(stop_flops, int) or not isinstance(branch_flops, int):
        errors.append("prior-corrected rescue FLOP ledger is incomplete")
    elif not (0 < stop_flops < branch_flops):
        errors.append("prior-corrected rescue FLOP ledger ordering is invalid")
    elif not isinstance(ratio, float) or abs(ratio - (branch_flops - stop_flops) / branch_flops) > 1e-12:
        errors.append("prior-corrected rescue economics ratio does not match sealed FLOP ledger")
    return errors


def run_exp279_prior_corrected_rescue_development(**kwargs: Any) -> dict[str, Any]:
    rescue_anchor_states: list[torch.Tensor] = []
    natural_counts = {"positive": 0, "total": 0}
    original_train_step = _base._train_step
    original_supervision = _base.ROUTING_SUPERVISION
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    _base._train_step = _make_train_step(rescue_anchor_states, natural_counts, original_train_step)
    try:
        payload = _base.run_exp279_paired_development(**kwargs)
        geometry = payload["world_geometry"]
        ledgers = _compute_ledgers(
            _base._build_seeded_triplet(
                root_seed=payload["root_seed"],
                d_model=int(payload["arm_geometry"]["d_model"]),
                hidden_size=int(payload["arm_geometry"]["hidden_size"]),
                target_parameters=int(payload["arm_geometry"]["target_parameters"]),
                route_threshold=float(payload["route_config"]["threshold"]),
            )[2],
            timesteps=int(geometry["timesteps"]),
            variables=int(geometry["variables"]),
            constraints=int(geometry["constraints"]),
        )
        hybrid_ledger = ledgers["hybrid"]
        stop_flops = int(hybrid_ledger["stop_accounted_flops_per_episode"])
        branch_flops = int(hybrid_ledger["branch_accounted_flops_per_episode"])
        payload["training"]["prior_corrected_rescue_replay"] = {
            "source": "augmentation_training_only",
            "final_anchor_episodes": _anchor_episode_count(rescue_anchor_states),
            "anchor_batches": len(rescue_anchor_states),
            "natural_positive_count": natural_counts["positive"],
            "natural_total_count": natural_counts["total"],
            "natural_positive_fraction": (
                natural_counts["positive"] / natural_counts["total"]
                if natural_counts["total"]
                else 0.0
            ),
            "stop_accounted_flops_per_episode": stop_flops,
            "branch_accounted_flops_per_episode": branch_flops,
            "incremental_cost_ratio": (branch_flops - stop_flops) / branch_flops,
            "evaluation_examples_used": False,
            "evaluation_targets_used": False,
            "decision_threshold_changed": False,
            "tunable_calibration_hyperparameters": False,
        }
        payload["artifact_digest"] = _base._artifact_digest(payload)
        errors = validate_exp279_prior_corrected_rescue_development(payload)
        if errors:
            raise RuntimeError(
                "invalid EXP-279 prior-corrected rescue DEVELOPMENT artifact: " + "; ".join(errors)
            )
        return payload
    finally:
        _base._train_step = original_train_step
        _base.ROUTING_SUPERVISION = original_supervision
