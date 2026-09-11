from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch
from torch.nn import functional as F

from . import exp279_paired_runner as _base
from .matched_routing_arms import HybridRoutingArm


ROUTING_SUPERVISION = deepcopy(_base.ROUTING_SUPERVISION)
ROUTING_SUPERVISION["hybrid_route_teacher"] = {
    "positive": "stop_exact_failure_and_forced_branch_exact_success",
    "negative": "otherwise",
    "evaluation_targets_used_for_routing": False,
    "decision_threshold_changed": False,
    "rescue_anchor_replay": {
        "enabled": True,
        "source": "prior_augmentation_training_rescue_routing_states",
        "storage": "detached_all_prior_positive_states",
        "negative_source": "current_augmentation_training_non_rescues",
        "negative_only_before_first_anchor": "zero_routing_gradient",
        "class_balance": "equal_positive_negative_mass_when_both_present",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
    },
}


def _rescue_anchor_replay_loss(
    probabilities: torch.Tensor,
    rescue_targets: torch.Tensor,
    replay_positive_probabilities: torch.Tensor,
) -> torch.Tensor:
    if probabilities.shape != rescue_targets.shape:
        raise ValueError("routing probabilities and rescue targets must have identical shape")
    if probabilities.ndim != 1 or replay_positive_probabilities.ndim != 1:
        raise ValueError("rescue-anchor routing probabilities must be rank-1")

    positive_mask = rescue_targets > 0.5
    negative_mask = ~positive_mask
    current_positive = probabilities[positive_mask]
    current_negative = probabilities[negative_mask]

    positive_parts: list[torch.Tensor] = []
    if current_positive.numel() > 0:
        positive_parts.append(current_positive)
    if replay_positive_probabilities.numel() > 0:
        positive_parts.append(replay_positive_probabilities)

    if not positive_parts:
        # Before the first observed rescue, do not let rescue-free batches
        # repeatedly drive the route head into the all-stop basin.
        return probabilities.sum() * 0.0

    positive = torch.cat(positive_parts)
    positive_loss = F.binary_cross_entropy(positive, torch.ones_like(positive))
    if current_negative.numel() == 0:
        return positive_loss

    negative_loss = F.binary_cross_entropy(
        current_negative,
        torch.zeros_like(current_negative),
    )
    return 0.5 * (positive_loss + negative_loss)


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


def _train_hybrid_with_rescue_anchor_replay(
    arm: HybridRoutingArm,
    optimizer: torch.optim.Optimizer,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
    rescue_anchor_states: list[torch.Tensor],
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
    routing_state = _hybrid_routing_state(
        arm,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )

    if rescue_anchor_states:
        replay_positive_probabilities = torch.cat(
            [arm._residual_uncertainty(anchor) for anchor in rescue_anchor_states]
        )
    else:
        replay_positive_probabilities = output.residual_uncertainty.new_empty((0,))

    routing_loss = _rescue_anchor_replay_loss(
        output.residual_uncertainty,
        rescue_targets,
        replay_positive_probabilities,
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

    _append_rescue_anchor_states(
        rescue_anchor_states,
        routing_state,
        rescue_targets,
    )
    loss_value = decision_loss.detach() + routing_weight * routing_loss.detach()
    return float(loss_value.item())


def _make_train_step(
    rescue_anchor_states: list[torch.Tensor],
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
        return _train_hybrid_with_rescue_anchor_replay(
            arm,
            optimizer,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
            targets=targets,
            rescue_anchor_states=rescue_anchor_states,
        )

    return train_step


def _anchor_episode_count(anchors: list[torch.Tensor]) -> int:
    return sum(int(anchor.shape[0]) for anchor in anchors)


def validate_exp279_rescue_anchor_replay_development(payload: dict[str, Any]) -> list[str]:
    original_supervision = _base.ROUTING_SUPERVISION
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    try:
        errors = _base.validate_exp279_paired_development(payload)
    finally:
        _base.ROUTING_SUPERVISION = original_supervision

    replay = (payload.get("training") or {}).get("rescue_anchor_replay") or {}
    if replay.get("source") != "augmentation_training_only":
        errors.append("rescue-anchor replay source must remain augmentation training only")
    for key in ("final_anchor_episodes", "anchor_batches"):
        value = replay.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"rescue-anchor replay {key} is invalid")
    if replay.get("evaluation_examples_used") is not False:
        errors.append("evaluation examples cannot enter rescue-anchor replay")
    if replay.get("evaluation_targets_used") is not False:
        errors.append("evaluation targets cannot enter rescue-anchor replay")
    return errors


def run_exp279_rescue_anchor_replay_development(**kwargs: Any) -> dict[str, Any]:
    rescue_anchor_states: list[torch.Tensor] = []
    original_train_step = _base._train_step
    original_supervision = _base.ROUTING_SUPERVISION
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    _base._train_step = _make_train_step(rescue_anchor_states, original_train_step)
    try:
        payload = _base.run_exp279_paired_development(**kwargs)
        payload["training"]["rescue_anchor_replay"] = {
            "source": "augmentation_training_only",
            "final_anchor_episodes": _anchor_episode_count(rescue_anchor_states),
            "anchor_batches": len(rescue_anchor_states),
            "evaluation_examples_used": False,
            "evaluation_targets_used": False,
        }
        payload["artifact_digest"] = _base._artifact_digest(payload)
        errors = validate_exp279_rescue_anchor_replay_development(payload)
        if errors:
            raise RuntimeError(
                "invalid EXP-279 rescue-anchor DEVELOPMENT artifact: " + "; ".join(errors)
            )
        return payload
    finally:
        _base._train_step = original_train_step
        _base.ROUTING_SUPERVISION = original_supervision
