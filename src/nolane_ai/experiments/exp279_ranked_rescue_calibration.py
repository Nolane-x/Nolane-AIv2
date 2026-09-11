from __future__ import annotations

from copy import deepcopy
import math
from types import MethodType
from typing import Any, Callable

import torch
from torch.nn import functional as F

from . import exp279_paired_runner as _base
from .matched_routing_arms import HybridRoutingArm, _compute_ledgers


RESIDUAL_STATISTIC = "balanced_rescue_log_likelihood_ratio_with_analytic_prior_cost_calibration"
ROUTING_SUPERVISION = deepcopy(_base.ROUTING_SUPERVISION)
ROUTING_SUPERVISION["hybrid_route_teacher"] = {
    "positive": "stop_exact_failure_and_forced_branch_exact_success",
    "negative": "otherwise",
    "evaluation_targets_used_for_routing": False,
    "decision_threshold_changed": False,
    "ranked_rescue_calibration": {
        "enabled": True,
        "source": "augmentation_training_only",
        "discriminator": "balanced_logistic_proper_loss_on_detached_episode_routing_states",
        "discriminator_logit_semantics": "balanced_class_log_likelihood_ratio",
        "natural_prior": "causal_cumulative_raw_augmentation_rescue_frequency",
        "decision_calibration": "analytic_bayes_prior_plus_sealed_ledger_break_even",
        "decision_threshold": 0.5,
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
        "tunable_calibration_hyperparameters": False,
    },
}


def pairwise_rescue_harm_loss(
    rescue_scores: torch.Tensor,
    harm_scores: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic ranking loss that orders rescue above harm."""
    if rescue_scores.shape != harm_scores.shape:
        raise ValueError("rescue_scores and harm_scores must have identical shape")
    if rescue_scores.numel() == 0:
        raise ValueError("rescue_scores and harm_scores must be non-empty")
    return F.softplus(harm_scores - rescue_scores).mean()


def balanced_log_likelihood_ratio_loss(
    rescue_logits: torch.Tensor,
    nonrescue_logits: torch.Tensor,
) -> torch.Tensor:
    """Balanced logistic proper loss whose optimal logit is a log likelihood ratio.

    Class means are weighted equally, independently of the natural rescue prior.
    This estimates discrimination on a balanced case-control surface; the natural
    prior is reintroduced only by the analytic routing calibration.
    """
    if rescue_logits.ndim != 1 or nonrescue_logits.ndim != 1:
        raise ValueError("rescue_logits and nonrescue_logits must be rank-1")
    if rescue_logits.numel() == 0 or nonrescue_logits.numel() == 0:
        raise ValueError("both rescue and nonrescue logits must be non-empty")
    positive = F.softplus(-rescue_logits).mean()
    negative = F.softplus(nonrescue_logits).mean()
    return 0.5 * (positive + negative)


def branch_rescue_break_even_probability(
    *,
    stop_accounted_flops_per_episode: float,
    branch_accounted_flops_per_episode: float,
) -> float:
    """Return the rescue probability at which expected verified utility breaks even.

    With stop cost S and routed branch cost B, the incremental-cost ratio is
    c=(B-S)/B. A successful rescue contributes the branch-path utility while a
    non-rescue pays the extra branch cost, yielding q*=c/(1+c). This is distinct
    from c itself and is derived only from the sealed analytical ledger.
    """
    stop = float(stop_accounted_flops_per_episode)
    branch = float(branch_accounted_flops_per_episode)
    if not math.isfinite(stop) or not math.isfinite(branch) or stop <= 0.0 or branch <= stop:
        raise ValueError(
            "branch_accounted_flops_per_episode must be finite and greater than positive stop cost"
        )
    incremental_cost_ratio = (branch - stop) / branch
    return incremental_cost_ratio / (1.0 + incremental_cost_ratio)


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def _sigmoid(log_odds: float) -> float:
    if log_odds >= 0.0:
        tail = math.exp(-log_odds)
        return 1.0 / (1.0 + tail)
    head = math.exp(log_odds)
    return head / (1.0 + head)


def calibrated_route_probability(
    *,
    log_likelihood_ratio: float,
    natural_rescue_prior: float,
    break_even_probability: float,
) -> float:
    """Map discrimination evidence to a fixed-0.5 economic routing score."""
    if not 0.0 < natural_rescue_prior < 1.0:
        raise ValueError("natural_rescue_prior must be strictly between 0 and 1")
    if not 0.0 < break_even_probability < 1.0:
        raise ValueError("break_even_probability must be strictly between 0 and 1")
    if not math.isfinite(log_likelihood_ratio):
        raise ValueError("log_likelihood_ratio must be finite")

    posterior_log_odds = _logit(natural_rescue_prior) + float(log_likelihood_ratio)
    economic_log_odds = posterior_log_odds - _logit(break_even_probability)
    return _sigmoid(economic_log_odds)


def calibrated_route_scores(
    log_likelihood_ratio: torch.Tensor,
    *,
    natural_rescue_prior: float,
    break_even_probability: float,
) -> torch.Tensor:
    """Differentiable tensor form of the fixed-0.5 analytic decision calibration."""
    if not torch.is_floating_point(log_likelihood_ratio):
        raise ValueError("log_likelihood_ratio must be floating point")
    if not 0.0 < natural_rescue_prior < 1.0:
        raise ValueError("natural_rescue_prior must be strictly between 0 and 1")
    if not 0.0 < break_even_probability < 1.0:
        raise ValueError("break_even_probability must be strictly between 0 and 1")
    prior_log_odds = _logit(natural_rescue_prior)
    break_even_log_odds = _logit(break_even_probability)
    return torch.sigmoid(log_likelihood_ratio + prior_log_odds - break_even_log_odds)


def _episode_log_likelihood_ratio(arm: HybridRoutingArm, routing_state: torch.Tensor) -> torch.Tensor:
    return arm.routing_head(routing_state).squeeze(-1).mean(dim=-1)


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


def _append_training_states(
    states: list[torch.Tensor],
    targets: list[torch.Tensor],
    routing_state: torch.Tensor,
    rescue_target: torch.Tensor,
) -> None:
    if routing_state.ndim != 3:
        raise ValueError("routing_state must be [batch,variables,hidden]")
    if rescue_target.ndim != 1 or rescue_target.shape[0] != routing_state.shape[0]:
        raise ValueError("rescue_target must align with routing_state batch")
    states.append(routing_state.detach().clone())
    targets.append(rescue_target.detach().to(dtype=torch.bool).clone())


def _balanced_replay_loss(
    arm: HybridRoutingArm,
    states: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    if not states or not targets or len(states) != len(targets):
        return next(arm.routing_head.parameters()).sum() * 0.0
    all_states = torch.cat(states, dim=0)
    all_targets = torch.cat(targets, dim=0)
    logits = _episode_log_likelihood_ratio(arm, all_states)
    positive = logits[all_targets]
    negative = logits[~all_targets]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    return balanced_log_likelihood_ratio_loss(positive, negative)


def _make_residual_method(calibration: dict[str, float | int]) -> Callable[..., torch.Tensor]:
    def residual(self: HybridRoutingArm, hidden: torch.Tensor) -> torch.Tensor:
        total = int(calibration["natural_total_count"])
        positive = int(calibration["natural_positive_count"])
        if total <= 0 or positive <= 0:
            return hidden.new_zeros((hidden.shape[0],))
        if positive >= total:
            return hidden.new_ones((hidden.shape[0],))
        natural_prior = positive / total
        raw_log_lr = _episode_log_likelihood_ratio(self, hidden)
        return calibrated_route_scores(
            raw_log_lr,
            natural_rescue_prior=natural_prior,
            break_even_probability=float(calibration["break_even_probability"]),
        )

    return residual


def _make_train_step(
    *,
    training_states: list[torch.Tensor],
    training_targets: list[torch.Tensor],
    calibration: dict[str, float | int],
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
        if not isinstance(arm, HybridRoutingArm):
            raise TypeError("ranked rescue calibration requires HybridRoutingArm")

        arm.train()
        optimizer.zero_grad(set_to_none=True)
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
        rescue_target = _base._hybrid_branch_rescue_target(
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
        _append_training_states(training_states, training_targets, routing_state, rescue_target)
        calibration["natural_positive_count"] = int(calibration["natural_positive_count"]) + int(
            (rescue_target > 0.5).sum().item()
        )
        calibration["natural_total_count"] = int(calibration["natural_total_count"]) + int(
            rescue_target.numel()
        )

        output = arm(surface_events, variable_states, incidence)
        final_decision_loss = F.cross_entropy(
            output.decision_logits.reshape(-1, 2),
            targets.reshape(-1),
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

        routing_loss = _balanced_replay_loss(arm, training_states, training_targets)
        routing_parameters = tuple(arm.routing_head.parameters())
        routing_gradients = torch.autograd.grad(
            float(ROUTING_SUPERVISION["weight"]) * routing_loss,
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
        combined = decision_loss.detach() + float(ROUTING_SUPERVISION["weight"]) * routing_loss.detach()
        return float(combined.item())

    return train_step


def validate_exp279_ranked_rescue_development(payload: dict[str, Any]) -> list[str]:
    original_supervision = _base.ROUTING_SUPERVISION
    original_residual = _base.RESIDUAL_STATISTIC
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    _base.RESIDUAL_STATISTIC = RESIDUAL_STATISTIC
    try:
        errors = _base.validate_exp279_paired_development(payload)
    finally:
        _base.ROUTING_SUPERVISION = original_supervision
        _base.RESIDUAL_STATISTIC = original_residual

    receipt = (payload.get("training") or {}).get("ranked_rescue_calibration") or {}
    if receipt.get("source") != "augmentation_training_only":
        errors.append("ranked rescue calibration source must remain augmentation training only")
    for key in ("natural_positive_count", "natural_total_count", "rescue_anchor_episodes", "nonrescue_anchor_episodes"):
        value = receipt.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"ranked rescue calibration {key} is invalid")
    positive = int(receipt.get("natural_positive_count", 0) or 0)
    total = int(receipt.get("natural_total_count", 0) or 0)
    if positive > total:
        errors.append("ranked rescue natural counts are inconsistent")
    if int(receipt.get("rescue_anchor_episodes", -1)) != positive:
        errors.append("ranked rescue positive anchor count drift")
    if int(receipt.get("nonrescue_anchor_episodes", -1)) != total - positive:
        errors.append("ranked rescue nonrescue anchor count drift")
    if receipt.get("evaluation_examples_used") is not False:
        errors.append("evaluation examples cannot enter ranked rescue calibration")
    if receipt.get("evaluation_targets_used") is not False:
        errors.append("evaluation targets cannot enter ranked rescue calibration")
    if receipt.get("decision_threshold_changed") is not False:
        errors.append("ranked rescue calibration cannot change the frozen threshold")
    if receipt.get("tunable_calibration_hyperparameters") is not False:
        errors.append("ranked rescue calibration cannot add tunable calibration hyperparameters")
    stop = receipt.get("stop_accounted_flops_per_episode")
    branch = receipt.get("branch_accounted_flops_per_episode")
    try:
        expected_break_even = branch_rescue_break_even_probability(
            stop_accounted_flops_per_episode=float(stop),
            branch_accounted_flops_per_episode=float(branch),
        )
        observed_break_even = float(receipt.get("break_even_probability"))
    except (TypeError, ValueError):
        errors.append("ranked rescue ledger calibration receipt is invalid")
    else:
        if abs(observed_break_even - expected_break_even) > 1e-15:
            errors.append("ranked rescue break-even probability does not match sealed ledger")
    return errors


def run_exp279_ranked_rescue_development(**kwargs: Any) -> dict[str, Any]:
    training_states: list[torch.Tensor] = []
    training_targets: list[torch.Tensor] = []
    calibration: dict[str, float | int] = {
        "natural_positive_count": 0,
        "natural_total_count": 0,
        "stop_accounted_flops_per_episode": 0.0,
        "branch_accounted_flops_per_episode": 0.0,
        "break_even_probability": 0.5,
    }
    original_build = _base._build_seeded_triplet
    original_train_step = _base._train_step
    original_supervision = _base.ROUTING_SUPERVISION
    original_residual = _base.RESIDUAL_STATISTIC

    def build_triplet(**build_kwargs: Any):
        propagation, branch, hybrid, model_seed = original_build(**build_kwargs)
        ledgers = _compute_ledgers(
            hybrid,
            timesteps=int(kwargs["timesteps"]),
            variables=int(kwargs["variables"]),
            constraints=int(kwargs["constraints"]),
        )
        hybrid_ledger = ledgers["hybrid"]
        stop = float(hybrid_ledger["stop_accounted_flops_per_episode"])
        routed = float(hybrid_ledger["branch_accounted_flops_per_episode"])
        calibration["stop_accounted_flops_per_episode"] = stop
        calibration["branch_accounted_flops_per_episode"] = routed
        calibration["break_even_probability"] = branch_rescue_break_even_probability(
            stop_accounted_flops_per_episode=stop,
            branch_accounted_flops_per_episode=routed,
        )
        hybrid._residual_uncertainty = MethodType(_make_residual_method(calibration), hybrid)
        return propagation, branch, hybrid, model_seed

    _base._build_seeded_triplet = build_triplet
    _base._train_step = _make_train_step(
        training_states=training_states,
        training_targets=training_targets,
        calibration=calibration,
        original_train_step=original_train_step,
    )
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    _base.RESIDUAL_STATISTIC = RESIDUAL_STATISTIC
    try:
        payload = _base.run_exp279_paired_development(**kwargs)
    finally:
        _base._build_seeded_triplet = original_build
        _base._train_step = original_train_step
        _base.ROUTING_SUPERVISION = original_supervision
        _base.RESIDUAL_STATISTIC = original_residual

    positive = int(calibration["natural_positive_count"])
    total = int(calibration["natural_total_count"])
    payload["training"]["ranked_rescue_calibration"] = {
        "source": "augmentation_training_only",
        "discriminator": "balanced_logistic_proper_loss_on_detached_episode_routing_states",
        "discriminator_logit_semantics": "balanced_class_log_likelihood_ratio",
        "natural_prior": "causal_cumulative_raw_augmentation_rescue_frequency",
        "natural_positive_count": positive,
        "natural_total_count": total,
        "natural_positive_fraction": positive / total if total else 0.0,
        "rescue_anchor_episodes": positive,
        "nonrescue_anchor_episodes": total - positive,
        "stop_accounted_flops_per_episode": int(calibration["stop_accounted_flops_per_episode"]),
        "branch_accounted_flops_per_episode": int(calibration["branch_accounted_flops_per_episode"]),
        "break_even_probability": float(calibration["break_even_probability"]),
        "decision_calibration": "analytic_bayes_prior_plus_sealed_ledger_break_even",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "decision_threshold_changed": False,
        "tunable_calibration_hyperparameters": False,
    }
    payload["replay_contract_digest"] = _base.canonical_sha256(_base._replay_contract(payload))
    payload["artifact_digest"] = _base._artifact_digest(payload)
    errors = validate_exp279_ranked_rescue_development(payload)
    if errors:
        raise RuntimeError(
            "invalid EXP-279 ranked rescue DEVELOPMENT artifact: " + "; ".join(errors)
        )
    return payload
