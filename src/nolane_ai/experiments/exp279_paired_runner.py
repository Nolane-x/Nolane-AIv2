from __future__ import annotations

import torch
from torch.nn import functional as F

from . import exp279_paired_runner_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

ROUTING_SUPERVISION["episode_targets"]["hybrid"] = "soft_branch_advantage"
ROUTING_SUPERVISION["hybrid_route_teacher"] = {
    "target": "sigmoid_per_episode_stop_ce_minus_forced_branch_ce",
    "temperature": 1.0,
    "uses_same_paired_training_targets": True,
    "evaluation_targets_used_for_routing": False,
    "decision_threshold_changed": False,
}


def _hybrid_branch_advantage_teacher(
    stop_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if stop_logits.shape != branch_logits.shape:
        raise ValueError("stop and forced-branch logits must have identical shape")
    if stop_logits.ndim < 3 or stop_logits.shape[-1] != 2:
        raise ValueError("hybrid route teacher expects [..., 2] decision logits")
    if tuple(targets.shape) != tuple(stop_logits.shape[:-1]):
        raise ValueError("hybrid route teacher target shape mismatch")
    batch = int(targets.shape[0])
    with torch.no_grad():
        flat_targets = targets.detach().reshape(-1)
        stop_ce = F.cross_entropy(
            stop_logits.detach().reshape(-1, 2), flat_targets, reduction="none"
        ).reshape(batch, -1).mean(dim=1)
        branch_ce = F.cross_entropy(
            branch_logits.detach().reshape(-1, 2), flat_targets, reduction="none"
        ).reshape(batch, -1).mean(dim=1)
        temperature = float(ROUTING_SUPERVISION["hybrid_route_teacher"]["temperature"])
        return torch.sigmoid((stop_ce - branch_ce) / temperature)


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
        routing_target = _hybrid_branch_advantage_teacher(
            stop_logits, forced_branch_logits, targets
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


_core._train_step = _train_step
