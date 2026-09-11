from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from . import exp279_paired_runner_core as _core
from .matched_routing_arms import _compute_ledgers

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

ROUTING_SUPERVISION["episode_targets"]["hybrid"] = "cost_aware_soft_marginal_utility"
ROUTING_SUPERVISION["hybrid_route_teacher"] = {
    "target": "soft_exact_utility_branch_share",
    "soft_exact_surrogate": "exp_sum_log_target_probability",
    "utility": "soft_exact_probability_divided_by_accounted_flops",
    "temperature": 1.0,
    "cost_source": "sealed_pair_audit_hybrid_stop_and_branch_ledgers",
    "uses_same_paired_training_targets": True,
    "evaluation_targets_used_for_routing": False,
    "decision_threshold_changed": False,
}

_AUDITED_HYBRID_COSTS: dict[int, tuple[float, float]] = {}
_ORIGINAL_AUDIT_MATCHED_EXP279_ARM_TRIPLET = _core.audit_matched_exp279_arm_triplet


def _hybrid_cost_aware_marginal_utility_teacher(
    stop_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    stop_accounted_flops_per_episode: float,
    branch_accounted_flops_per_episode: float,
) -> torch.Tensor:
    if stop_logits.shape != branch_logits.shape:
        raise ValueError("stop and forced-branch logits must have identical shape")
    if stop_logits.ndim < 3 or stop_logits.shape[-1] != 2:
        raise ValueError("hybrid route teacher expects [..., 2] decision logits")
    if tuple(targets.shape) != tuple(stop_logits.shape[:-1]):
        raise ValueError("hybrid route teacher target shape mismatch")

    stop_cost = float(stop_accounted_flops_per_episode)
    branch_cost = float(branch_accounted_flops_per_episode)
    if (
        not math.isfinite(stop_cost)
        or not math.isfinite(branch_cost)
        or stop_cost <= 0.0
        or branch_cost <= 0.0
    ):
        raise ValueError("hybrid route teacher accounted FLOPs must be finite and positive")

    temperature = float(ROUTING_SUPERVISION["hybrid_route_teacher"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("hybrid route teacher temperature must be finite and positive")

    batch = int(targets.shape[0])
    with torch.no_grad():
        target_index = targets.detach().unsqueeze(-1)
        stop_log_target = F.log_softmax(stop_logits.detach(), dim=-1).gather(
            dim=-1, index=target_index
        ).squeeze(-1)
        branch_log_target = F.log_softmax(branch_logits.detach(), dim=-1).gather(
            dim=-1, index=target_index
        ).squeeze(-1)

        # Exact success is an all-variable event. Multiplying target probabilities
        # directly is numerically fragile, so compare the smooth exact surrogate in
        # log space and divide by the same accounted-FLOP costs used by EXP-279.
        stop_log_utility = stop_log_target.reshape(batch, -1).sum(dim=1) - math.log(stop_cost)
        branch_log_utility = branch_log_target.reshape(batch, -1).sum(dim=1) - math.log(branch_cost)
        return torch.sigmoid((branch_log_utility - stop_log_utility) / temperature)


def audit_matched_exp279_arm_triplet(
    propagation: PropagationOnlyArm,
    branch: BranchOnlyArm,
    hybrid: HybridRoutingArm,
    **kwargs: object,
) -> dict[str, object]:
    audit = _ORIGINAL_AUDIT_MATCHED_EXP279_ARM_TRIPLET(
        propagation,
        branch,
        hybrid,
        **kwargs,
    )
    ledger = audit["compute_ledger"]["hybrid"]
    stop_cost = float(ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(ledger["branch_accounted_flops_per_episode"])
    if stop_cost <= 0.0 or branch_cost <= 0.0:
        raise RuntimeError("EXP-279 audited hybrid route costs must be positive")
    _AUDITED_HYBRID_COSTS[id(hybrid)] = (stop_cost, branch_cost)
    return audit


def _hybrid_route_costs(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[float, float]:
    audited = _AUDITED_HYBRID_COSTS.get(id(arm))
    if audited is not None:
        return audited

    # Unit-level callers can invoke _train_step without the paired runner's audit.
    # Use the identical analytical ledger function as a deterministic fallback;
    # scientific runner executions take the cached sealed pair-audit values above.
    ledger = _compute_ledgers(
        arm,
        timesteps=int(surface_events.shape[1]),
        variables=int(variable_states.shape[1]),
        constraints=int(incidence.shape[1]),
    )["hybrid"]
    return (
        float(ledger["stop_accounted_flops_per_episode"]),
        float(ledger["branch_accounted_flops_per_episode"]),
    )


def _train_step(
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

    final_decision_loss = F.cross_entropy(
        output.decision_logits.reshape(-1, 2), targets.reshape(-1)
    )
    decision_loss = final_decision_loss
    if arm_id == "hybrid":
        stop_logits = _hybrid_stop_decision_logits(
            arm,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
        )
        forced_branch_logits = _hybrid_forced_branch_decision_logits(
            arm,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
        )
        stop_path = ROUTING_SUPERVISION["hybrid_stop_path_supervision"]
        stop_decision_loss = F.cross_entropy(
            stop_logits.reshape(-1, 2), targets.reshape(-1)
        )
        forced_branch = ROUTING_SUPERVISION["hybrid_forced_branch_supervision"]
        forced_branch_decision_loss = F.cross_entropy(
            forced_branch_logits.reshape(-1, 2), targets.reshape(-1)
        )
        decision_loss = (
            float(stop_path["final_path_weight"]) * final_decision_loss
            + float(stop_path["stop_path_weight"]) * stop_decision_loss
            + float(forced_branch["weight"]) * forced_branch_decision_loss
        )
        stop_cost, branch_cost = _hybrid_route_costs(
            arm,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
        )
        routing_target = _hybrid_cost_aware_marginal_utility_teacher(
            stop_logits,
            forced_branch_logits,
            targets,
            stop_accounted_flops_per_episode=stop_cost,
            branch_accounted_flops_per_episode=branch_cost,
        )
    else:
        routing_target = _episode_failure_target(output.decision_logits, targets)

    routing_loss = F.binary_cross_entropy(output.residual_uncertainty, routing_target)
    routing_weight = float(ROUTING_SUPERVISION["weight"])
    routing_parameters = tuple(arm.routing_head.parameters())
    routing_gradients = torch.autograd.grad(
        routing_weight * routing_loss, routing_parameters, retain_graph=True
    )
    decision_loss.backward()
    for parameter, routing_gradient in zip(routing_parameters, routing_gradients):
        if parameter.grad is None:
            parameter.grad = routing_gradient.detach().clone()
        else:
            parameter.grad.add_(routing_gradient)
    optimizer.step()
    loss_value = decision_loss.detach() + routing_weight * routing_loss.detach()
    return float(loss_value.item())


_core.audit_matched_exp279_arm_triplet = audit_matched_exp279_arm_triplet
_core._train_step = _train_step
