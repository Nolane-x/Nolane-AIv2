from __future__ import annotations

from copy import deepcopy
import math
from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from . import exp279_paired_runner as _base
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import HybridRoutingArm, _compute_ledgers


RESIDUAL_STATISTIC = "economic_rescue_score_from_pairwise_likelihood_ratio"

ROUTING_SUPERVISION = deepcopy(_base.ROUTING_SUPERVISION)
ROUTING_SUPERVISION["hybrid_route_teacher"] = {
    "positive": "stop_exact_failure_and_forced_branch_exact_success",
    "negative": "otherwise",
    "evaluation_targets_used_for_routing": False,
    "decision_threshold_changed": False,
    "rescue_likelihood_ratio": {
        "enabled": True,
        "source": "augmentation_training_only",
        "ranking_loss": "pairwise_logistic_rescue_over_all_non_rescue",
        "ranking_score": "mean_routing_logit_on_propagation_state",
        "calibration": "training_moment_matching_to_natural_rescue_prevalence",
        "economics": "incremental_branch_flops_divided_by_branch_flops",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
        "tunable_calibration_hyperparameters": False,
    },
}


def _branch_outcome_masks(
    stop_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if stop_logits.shape != branch_logits.shape:
        raise ValueError("stop and branch logits must have identical shape")
    if stop_logits.ndim != 3 or stop_logits.shape[-1] != 2:
        raise ValueError("EXP-279 stop/branch logits must be [batch,variables,2]")
    if targets.shape != stop_logits.shape[:-1]:
        raise ValueError("EXP-279 targets must align with stop/branch logits")

    stop_exact = (stop_logits.detach().argmax(dim=-1) == targets).all(dim=-1)
    branch_exact = (branch_logits.detach().argmax(dim=-1) == targets).all(dim=-1)
    return {
        "rescue": (~stop_exact) & branch_exact,
        "harm": stop_exact & (~branch_exact),
        "both_success": stop_exact & branch_exact,
        "both_failure": (~stop_exact) & (~branch_exact),
    }


def _pairwise_rescue_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    if positive_scores.ndim != 1 or negative_scores.ndim != 1:
        raise ValueError("pairwise rescue ranking scores must be rank-1")
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return positive_scores.sum() * 0.0 + negative_scores.sum() * 0.0
    margins = positive_scores[:, None] - negative_scores[None, :]
    return F.softplus(-margins).mean()


def _solve_prevalence_intercept(
    ranking_logits: torch.Tensor,
    *,
    natural_positive_fraction: float,
) -> float:
    if ranking_logits.ndim != 1 or ranking_logits.numel() == 0:
        raise ValueError("ranking logits must be a non-empty rank-1 tensor")
    prevalence = float(natural_positive_fraction)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("natural positive fraction must lie strictly within (0, 1)")
    if not bool(torch.isfinite(ranking_logits).all().item()):
        raise ValueError("ranking logits must be finite")

    logits = ranking_logits.detach().to(dtype=torch.float64, device="cpu")
    low = -128.0
    high = 128.0
    for _ in range(256):
        mid = 0.5 * (low + high)
        mean_probability = float(torch.sigmoid(logits + mid).mean().item())
        if mean_probability < prevalence:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _validate_probability_tensor(value: torch.Tensor, *, name: str) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} must lie within [0, 1]")


def _economic_rescue_score_from_probability(
    natural_rescue_probability: torch.Tensor,
    *,
    incremental_cost_ratio: float,
) -> torch.Tensor:
    _validate_probability_tensor(
        natural_rescue_probability,
        name="natural rescue probability",
    )
    cost_ratio = float(incremental_cost_ratio)
    if not 0.0 < cost_ratio < 1.0:
        raise ValueError("incremental branch cost ratio must lie strictly within (0, 1)")
    numerator = natural_rescue_probability
    denominator = numerator + cost_ratio * (1.0 - numerator)
    return torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


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


def _ranking_logits(
    arm: HybridRoutingArm,
    routing_states: torch.Tensor,
) -> torch.Tensor:
    if routing_states.ndim != 3:
        raise ValueError("routing states must be [batch,variables,hidden]")
    variable_logits = arm.routing_head(routing_states).squeeze(-1)
    return variable_logits.mean(dim=-1)


def _append_partitioned_states(
    positive_states: list[torch.Tensor],
    negative_states: list[torch.Tensor],
    routing_states: torch.Tensor,
    rescue_mask: torch.Tensor,
) -> None:
    if rescue_mask.ndim != 1 or rescue_mask.shape[0] != routing_states.shape[0]:
        raise ValueError("rescue mask must align with routing-state batch dimension")
    detached = routing_states.detach()
    if bool(rescue_mask.any().item()):
        positive_states.append(detached[rescue_mask].clone())
    negative_mask = ~rescue_mask
    if bool(negative_mask.any().item()):
        negative_states.append(detached[negative_mask].clone())


def _concat_states(
    states: list[torch.Tensor],
    *,
    template: torch.Tensor,
) -> torch.Tensor:
    if states:
        return torch.cat(states, dim=0)
    return template.new_empty((0, template.shape[1], template.shape[2]))


def _fit_pairwise_router(
    arm: HybridRoutingArm,
    *,
    root_seed: str,
    train_replicates: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    d_model: int,
    noise_std: float,
    lr: float,
) -> dict[str, Any]:
    generator = Exp279RoutingGenerator(root_seed=root_seed)
    positive_states: list[torch.Tensor] = []
    negative_states: list[torch.Tensor] = []
    outcome_counts = {
        "rescue": 0,
        "harm": 0,
        "both_success": 0,
        "both_failure": 0,
    }
    losses: list[float] = []
    ranking_optimizer = torch.optim.AdamW(
        arm.routing_head.parameters(),
        lr=float(lr),
        weight_decay=0.0,
    )

    arm.train()
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
        with torch.no_grad():
            stop_logits = _base._hybrid_stop_decision_logits(
                arm,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            branch_logits = _base._hybrid_forced_branch_decision_logits(
                arm,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            masks = _branch_outcome_masks(stop_logits, branch_logits, batch.targets)
            routing_states = _hybrid_routing_state(
                arm,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            ).detach()

        for key, mask in masks.items():
            outcome_counts[key] += int(mask.to(torch.int64).sum().item())
        _append_partitioned_states(
            positive_states,
            negative_states,
            routing_states,
            masks["rescue"],
        )

        all_positive = _concat_states(positive_states, template=routing_states)
        all_negative = _concat_states(negative_states, template=routing_states)
        ranking_optimizer.zero_grad(set_to_none=True)
        positive_scores = _ranking_logits(arm, all_positive)
        negative_scores = _ranking_logits(arm, all_negative)
        loss = _pairwise_rescue_ranking_loss(positive_scores, negative_scores)
        loss.backward()
        ranking_optimizer.step()
        losses.append(float(loss.detach().item()))

    template = routing_states
    all_positive = _concat_states(positive_states, template=template)
    all_negative = _concat_states(negative_states, template=template)
    all_states = torch.cat((all_positive, all_negative), dim=0)
    natural_positive_count = int(all_positive.shape[0])
    natural_total_count = int(all_states.shape[0])
    if natural_total_count != train_replicates * batch_size:
        raise RuntimeError("EXP-279 ranking replay did not preserve augmentation-training episode count")

    with torch.no_grad():
        final_ranking_logits = _ranking_logits(arm, all_states)

    if natural_positive_count == 0:
        calibration_method = "degenerate_zero_positive_all_stop"
        calibration_intercept: float | None = None
    elif natural_positive_count == natural_total_count:
        calibration_method = "degenerate_all_positive_all_route"
        calibration_intercept = None
    else:
        calibration_method = "training_moment_matching_to_natural_rescue_prevalence"
        calibration_intercept = _solve_prevalence_intercept(
            final_ranking_logits,
            natural_positive_fraction=natural_positive_count / natural_total_count,
        )

    return {
        "source": "augmentation_training_only",
        "ranking_steps": train_replicates,
        "ranking_optimizer": {
            "type": "AdamW",
            "lr": float(lr),
            "weight_decay": 0.0,
            "parameter_scope": "routing_head_only",
        },
        "ranking_losses": losses,
        "mean_ranking_loss": sum(losses) / len(losses),
        "outcome_counts": outcome_counts,
        "natural_positive_count": natural_positive_count,
        "natural_total_count": natural_total_count,
        "natural_positive_fraction": natural_positive_count / natural_total_count,
        "calibration_method": calibration_method,
        "calibration_intercept": calibration_intercept,
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "decision_threshold_changed": False,
        "tunable_calibration_hyperparameters": False,
    }


def _bind_calibrated_router(
    arm: HybridRoutingArm,
    *,
    calibration_method: str,
    calibration_intercept: float | None,
    incremental_cost_ratio: float,
) -> None:
    def calibrated_residual(self: HybridRoutingArm, hidden: torch.Tensor) -> torch.Tensor:
        ranking_logits = _ranking_logits(self, hidden)
        if calibration_method == "degenerate_zero_positive_all_stop":
            natural_probability = torch.zeros_like(ranking_logits)
        elif calibration_method == "degenerate_all_positive_all_route":
            natural_probability = torch.ones_like(ranking_logits)
        elif calibration_method == "training_moment_matching_to_natural_rescue_prevalence":
            if calibration_intercept is None or not math.isfinite(calibration_intercept):
                raise RuntimeError("finite calibration intercept is required for non-degenerate routing")
            natural_probability = torch.sigmoid(ranking_logits + calibration_intercept)
        else:
            raise RuntimeError(f"unknown EXP-279 calibration method: {calibration_method}")
        return _economic_rescue_score_from_probability(
            natural_probability,
            incremental_cost_ratio=incremental_cost_ratio,
        )

    arm._residual_uncertainty = MethodType(calibrated_residual, arm)


def _evaluate_calibrated_router(
    *,
    propagation: torch.nn.Module,
    branch: torch.nn.Module,
    hybrid: HybridRoutingArm,
    root_seed: str,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    d_model: int,
    noise_std: float,
    prop_cost: float,
    branch_cost: float,
    hybrid_stop_cost: float,
    hybrid_branch_cost: float,
    route_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for arm in (propagation, branch, hybrid):
        arm.eval()
    generator = Exp279RoutingGenerator(root_seed=root_seed)
    rows: list[dict[str, Any]] = []
    discrimination = {
        "posthoc_evaluation_targets_used_for_diagnostic": True,
        "evaluation_targets_used_for_training": False,
        "evaluation_targets_used_for_deployable_routing": False,
        "total_episodes": 0,
        "natural_rescue_count": 0,
        "routed_episodes": 0,
        "routed_rescues": 0,
        "routed_harms": 0,
        "routed_both_success": 0,
        "routed_both_failure": 0,
    }

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
            prop_output = propagation(
                batch.surface_events,
                batch.variable_states,
                batch.incidence,
            )
            branch_output = branch(batch.surface_events, batch.variable_states)
            hybrid_output = hybrid(
                batch.surface_events,
                batch.variable_states,
                batch.incidence,
            )
            stop_logits = _base._hybrid_stop_decision_logits(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            forced_branch_logits = _base._hybrid_forced_branch_decision_logits(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            masks = _branch_outcome_masks(
                stop_logits,
                forced_branch_logits,
                batch.targets,
            )

            route_tensor = hybrid_output.branch_route_mask.to(dtype=torch.bool)
            route_mask = [bool(item) for item in route_tensor.cpu().tolist()]
            routed = int(route_tensor.to(torch.int64).sum().item())
            route_fraction = routed / batch_size
            hybrid_cost = hybrid_stop_cost + route_fraction * (
                hybrid_branch_cost - hybrid_stop_cost
            )
            discrimination["total_episodes"] += batch_size
            discrimination["natural_rescue_count"] += int(
                masks["rescue"].to(torch.int64).sum().item()
            )
            discrimination["routed_episodes"] += routed
            discrimination["routed_rescues"] += int(
                (route_tensor & masks["rescue"]).to(torch.int64).sum().item()
            )
            discrimination["routed_harms"] += int(
                (route_tensor & masks["harm"]).to(torch.int64).sum().item()
            )
            discrimination["routed_both_success"] += int(
                (route_tensor & masks["both_success"]).to(torch.int64).sum().item()
            )
            discrimination["routed_both_failure"] += int(
                (route_tensor & masks["both_failure"]).to(torch.int64).sum().item()
            )

            rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "paired_batch_digest": batch.digest,
                    "world_pairing_closed": True,
                    "propagation_only": _base._external_metrics(
                        prop_output,
                        batch.targets,
                        accounted_flops_per_episode=prop_cost,
                    ),
                    "branch_only": _base._external_metrics(
                        branch_output,
                        batch.targets,
                        accounted_flops_per_episode=branch_cost,
                    ),
                    "hybrid": _base._external_metrics(
                        hybrid_output,
                        batch.targets,
                        accounted_flops_per_episode=hybrid_cost,
                    ),
                    "hybrid_route_receipt": {
                        "strategy": "propagation_then_branch_on_residual_uncertainty",
                        "threshold": float(route_threshold),
                        "batch_size": batch_size,
                        "branch_route_mask": route_mask,
                        "routed_episodes": routed,
                        "route_fraction": route_fraction,
                        "mean_residual_uncertainty": float(
                            hybrid_output.residual_uncertainty.mean().item()
                        ),
                        "stop_accounted_flops_per_episode": hybrid_stop_cost,
                        "branch_accounted_flops_per_episode": hybrid_branch_cost,
                        "charged_accounted_flops_per_episode": hybrid_cost,
                    },
                }
            )

    total = int(discrimination["total_episodes"])
    routed_total = int(discrimination["routed_episodes"])
    rescue_total = int(discrimination["natural_rescue_count"])
    natural_prevalence = rescue_total / total if total else 0.0
    routed_precision = (
        int(discrimination["routed_rescues"]) / routed_total
        if routed_total
        else None
    )
    discrimination["natural_rescue_prevalence"] = natural_prevalence
    discrimination["routed_precision"] = routed_precision
    discrimination["precision_enrichment_vs_natural_prevalence"] = (
        routed_precision / natural_prevalence
        if routed_precision is not None and natural_prevalence > 0.0
        else None
    )
    return rows, discrimination


def validate_exp279_rescue_likelihood_ratio_development(
    payload: dict[str, Any],
) -> list[str]:
    original_supervision = _base.ROUTING_SUPERVISION
    original_statistic = _base.RESIDUAL_STATISTIC
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    _base.RESIDUAL_STATISTIC = RESIDUAL_STATISTIC
    try:
        errors = _base.validate_exp279_paired_development(payload)
    finally:
        _base.ROUTING_SUPERVISION = original_supervision
        _base.RESIDUAL_STATISTIC = original_statistic

    receipt = (payload.get("training") or {}).get("rescue_likelihood_ratio") or {}
    if receipt.get("source") != "augmentation_training_only":
        errors.append("rescue likelihood-ratio source must remain augmentation training only")
    for key in ("ranking_steps", "natural_positive_count", "natural_total_count"):
        value = receipt.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"rescue likelihood-ratio {key} is invalid")
    if int(receipt.get("natural_positive_count", 0)) > int(receipt.get("natural_total_count", 0)):
        errors.append("rescue likelihood-ratio natural counts are inconsistent")
    if receipt.get("evaluation_examples_used") is not False:
        errors.append("evaluation examples cannot enter rescue likelihood-ratio training")
    if receipt.get("evaluation_targets_used") is not False:
        errors.append("evaluation targets cannot enter rescue likelihood-ratio training")
    if receipt.get("decision_threshold_changed") is not False:
        errors.append("rescue likelihood-ratio training cannot change frozen route threshold")
    if receipt.get("tunable_calibration_hyperparameters") is not False:
        errors.append("rescue likelihood-ratio calibration cannot add tunable hyperparameters")
    calibration_method = receipt.get("calibration_method")
    if calibration_method not in {
        "training_moment_matching_to_natural_rescue_prevalence",
        "degenerate_zero_positive_all_stop",
        "degenerate_all_positive_all_route",
    }:
        errors.append("rescue likelihood-ratio calibration method is invalid")
    intercept = receipt.get("calibration_intercept")
    if calibration_method == "training_moment_matching_to_natural_rescue_prevalence":
        if not isinstance(intercept, float) or not math.isfinite(intercept):
            errors.append("rescue likelihood-ratio calibration intercept must be finite")
    elif intercept is not None:
        errors.append("degenerate rescue likelihood-ratio calibration cannot carry an intercept")

    stop_flops = receipt.get("stop_accounted_flops_per_episode")
    branch_flops = receipt.get("branch_accounted_flops_per_episode")
    ratio = receipt.get("incremental_cost_ratio")
    if not isinstance(stop_flops, int) or not isinstance(branch_flops, int):
        errors.append("rescue likelihood-ratio FLOP ledger is incomplete")
    elif not (0 < stop_flops < branch_flops):
        errors.append("rescue likelihood-ratio FLOP ledger ordering is invalid")
    elif not isinstance(ratio, float) or abs(
        ratio - (branch_flops - stop_flops) / branch_flops
    ) > 1e-12:
        errors.append("rescue likelihood-ratio economics ratio does not match sealed FLOP ledger")

    diagnostic = (payload.get("evaluation") or {}).get("routing_discrimination") or {}
    if diagnostic.get("posthoc_evaluation_targets_used_for_diagnostic") is not True:
        errors.append("rescue likelihood-ratio routed precision diagnostic provenance is missing")
    if diagnostic.get("evaluation_targets_used_for_training") is not False:
        errors.append("evaluation targets cannot enter rescue likelihood-ratio training")
    if diagnostic.get("evaluation_targets_used_for_deployable_routing") is not False:
        errors.append("evaluation targets cannot enter deployable rescue likelihood-ratio routing")
    total = diagnostic.get("total_episodes")
    routed = diagnostic.get("routed_episodes")
    routed_rescues = diagnostic.get("routed_rescues")
    if not isinstance(total, int) or total <= 0:
        errors.append("rescue likelihood-ratio diagnostic total episode count is invalid")
    if not isinstance(routed, int) or routed < 0 or (isinstance(total, int) and routed > total):
        errors.append("rescue likelihood-ratio routed episode count is invalid")
    if not isinstance(routed_rescues, int) or routed_rescues < 0 or (
        isinstance(routed, int) and routed_rescues > routed
    ):
        errors.append("rescue likelihood-ratio routed rescue count is invalid")
    if isinstance(routed, int) and routed > 0:
        expected_precision = routed_rescues / routed
        if abs(float(diagnostic.get("routed_precision")) - expected_precision) > 1e-12:
            errors.append("rescue likelihood-ratio routed precision mismatch")
    elif diagnostic.get("routed_precision") is not None:
        errors.append("zero-route rescue likelihood-ratio diagnostic must have null precision")
    return errors


def run_exp279_rescue_likelihood_ratio_development(**kwargs: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original_build = _base._build_seeded_triplet
    original_supervision = _base.ROUTING_SUPERVISION
    original_statistic = _base.RESIDUAL_STATISTIC

    def build_and_capture(**build_kwargs: Any):
        propagation, branch, hybrid, seed = original_build(**build_kwargs)
        captured["propagation"] = propagation
        captured["branch"] = branch
        captured["hybrid"] = hybrid
        return propagation, branch, hybrid, seed

    _base._build_seeded_triplet = build_and_capture
    _base.ROUTING_SUPERVISION = ROUTING_SUPERVISION
    _base.RESIDUAL_STATISTIC = RESIDUAL_STATISTIC
    try:
        payload = _base.run_exp279_paired_development(**kwargs)
    finally:
        _base._build_seeded_triplet = original_build
        _base.ROUTING_SUPERVISION = original_supervision
        _base.RESIDUAL_STATISTIC = original_statistic

    propagation = captured["propagation"]
    branch = captured["branch"]
    hybrid: HybridRoutingArm = captured["hybrid"]
    geometry = payload["world_geometry"]
    training_receipt = _fit_pairwise_router(
        hybrid,
        root_seed=payload["root_seed"],
        train_replicates=int(payload["training"]["replicates"]),
        batch_size=int(geometry["batch_size"]),
        timesteps=int(geometry["timesteps"]),
        variables=int(geometry["variables"]),
        constraints=int(geometry["constraints"]),
        d_model=int(geometry["d_model"]),
        noise_std=float(geometry["noise_std"]),
        lr=float(payload["training"]["optimizer"]["lr"]),
    )

    ledgers = _compute_ledgers(
        hybrid,
        timesteps=int(geometry["timesteps"]),
        variables=int(geometry["variables"]),
        constraints=int(geometry["constraints"]),
    )
    prop_cost = float(ledgers["propagation_only"]["accounted_flops_per_episode"])
    branch_cost = float(ledgers["branch_only"]["accounted_flops_per_episode"])
    hybrid_stop_cost = float(ledgers["hybrid"]["stop_accounted_flops_per_episode"])
    hybrid_branch_cost = float(ledgers["hybrid"]["branch_accounted_flops_per_episode"])
    incremental_cost_ratio = (
        hybrid_branch_cost - hybrid_stop_cost
    ) / hybrid_branch_cost
    training_receipt["stop_accounted_flops_per_episode"] = int(hybrid_stop_cost)
    training_receipt["branch_accounted_flops_per_episode"] = int(hybrid_branch_cost)
    training_receipt["incremental_cost_ratio"] = incremental_cost_ratio

    _bind_calibrated_router(
        hybrid,
        calibration_method=str(training_receipt["calibration_method"]),
        calibration_intercept=training_receipt["calibration_intercept"],
        incremental_cost_ratio=incremental_cost_ratio,
    )

    rows, discrimination = _evaluate_calibrated_router(
        propagation=propagation,
        branch=branch,
        hybrid=hybrid,
        root_seed=payload["root_seed"],
        eval_replicates=int(payload["evaluation"]["replicates"]),
        eval_start_replicate=int(payload["evaluation"]["start_replicate"]),
        batch_size=int(geometry["batch_size"]),
        timesteps=int(geometry["timesteps"]),
        variables=int(geometry["variables"]),
        constraints=int(geometry["constraints"]),
        d_model=int(geometry["d_model"]),
        noise_std=float(geometry["noise_std"]),
        prop_cost=prop_cost,
        branch_cost=branch_cost,
        hybrid_stop_cost=hybrid_stop_cost,
        hybrid_branch_cost=hybrid_branch_cost,
        route_threshold=float(payload["route_config"]["threshold"]),
    )

    payload["training"]["rescue_likelihood_ratio"] = training_receipt
    payload["evaluation"]["per_replicate"] = rows
    payload["evaluation"]["aggregate"] = _base._aggregate_rows(rows)
    payload["evaluation"]["routing_discrimination"] = discrimination
    payload["final_state"]["hybrid_digest"] = _base._functional_state_digest(hybrid)
    payload["final_state_digest"] = canonical_sha256(payload["final_state"])
    payload["replay_contract_digest"] = canonical_sha256(_base._replay_contract(payload))
    payload["artifact_digest"] = _base._artifact_digest(payload)

    errors = validate_exp279_rescue_likelihood_ratio_development(payload)
    if errors:
        raise RuntimeError(
            "invalid EXP-279 rescue likelihood-ratio DEVELOPMENT artifact: "
            + "; ".join(errors)
        )
    return payload
