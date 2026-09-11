from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from . import exp279_paired_runner_core as _core

# Keep the large deterministic runner implementation byte-identical while this
# DEVELOPMENT-only calibration remediation is under falsification. Re-export
# its public/private contract so existing scientific tooling keeps one canonical
# import path; the core runner is patched below only at the training-step seam.
for _name, _value in vars(_core).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

ROUTING_SUPERVISION["hybrid_route_teacher"].update(
    {
        "class_balance": "equal_positive_negative_mass_when_both_present",
        "decision_threshold_changed": False,
    }
)


def _balanced_binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Give rescue/non-rescue classes equal calibration mass when both exist."""
    if probabilities.shape != targets.shape:
        raise ValueError("routing probabilities and targets must have identical shape")
    positive = targets > 0.5
    negative = ~positive
    if bool(positive.any().item()) and bool(negative.any().item()):
        positive_loss = F.binary_cross_entropy(
            probabilities[positive], targets[positive]
        )
        negative_loss = F.binary_cross_entropy(
            probabilities[negative], targets[negative]
        )
        return 0.5 * (positive_loss + negative_loss)
    return F.binary_cross_entropy(probabilities, targets)


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
        output.decision_logits.reshape(-1, 2),
        targets.reshape(-1),
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
        final_path_weight = float(stop_path["final_path_weight"])
        stop_path_weight = float(stop_path["stop_path_weight"])
        stop_decision_loss = F.cross_entropy(
            stop_logits.reshape(-1, 2),
            targets.reshape(-1),
        )
        forced_branch = ROUTING_SUPERVISION["hybrid_forced_branch_supervision"]
        forced_branch_weight = float(forced_branch["weight"])
        forced_branch_decision_loss = F.cross_entropy(
            forced_branch_logits.reshape(-1, 2),
            targets.reshape(-1),
        )
        decision_loss = (
            final_path_weight * final_decision_loss
            + stop_path_weight * stop_decision_loss
            + forced_branch_weight * forced_branch_decision_loss
        )
        routing_target = _hybrid_branch_rescue_target(
            stop_logits,
            forced_branch_logits,
            targets,
        )
        routing_loss = _balanced_binary_cross_entropy(
            output.residual_uncertainty,
            routing_target,
        )
    else:
        routing_target = _episode_failure_target(output.decision_logits, targets)
        routing_loss = F.binary_cross_entropy(
            output.residual_uncertainty,
            routing_target,
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
    loss_value = decision_loss.detach() + routing_weight * routing_loss.detach()
    return float(loss_value.item())


# Functions defined in the byte-identical core resolve globals from that module.
# Patch its one training seam so replay/validation/execution all use the same
# calibrated DEVELOPMENT behavior through the canonical runner import.
_core._train_step = _train_step
