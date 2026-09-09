from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from nolane_ai.experiments.exp279_paired_runner import (
    _build_seeded_triplet,
    _episode_failure_target,
    _hybrid_stop_decision_logits,
    _train_step,
)
from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator
from nolane_ai.training.optimizer import build_functional_optimizer


def _legacy_hybrid_step_without_direct_stop_supervision(
    arm: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Reproduce the pre-fix hybrid objective for a causal A/B test."""
    arm.train()
    optimizer.zero_grad(set_to_none=True)
    output = arm(surface_events, variable_states, incidence)
    decision_loss = F.cross_entropy(
        output.decision_logits.reshape(-1, 2),
        targets.reshape(-1),
    )
    stop_logits = _hybrid_stop_decision_logits(
        arm,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )
    routing_target = _episode_failure_target(stop_logits, targets)
    routing_loss = F.binary_cross_entropy(output.residual_uncertainty, routing_target)
    routing_parameters = tuple(arm.routing_head.parameters())
    routing_gradients = torch.autograd.grad(
        routing_loss,
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


def test_hybrid_training_directly_supervises_stop_decision_even_when_all_episodes_route() -> None:
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-stop-supervision",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
        route_threshold=0.0,
    )
    legacy = deepcopy(hybrid)
    generator = Exp279RoutingGenerator(root_seed="exp279-stop-supervision")
    batch = generator.make_batch(
        replicate=0,
        batch_size=4,
        timesteps=3,
        variables=4,
        constraints=2,
        d_model=16,
        noise_std=0.05,
        rng_stream="augmentation",
        stratum="PROPAGATION_FIT",
    )

    optimizer = build_functional_optimizer(hybrid, lr=0.002, weight_decay=0.0)
    legacy_optimizer = build_functional_optimizer(legacy, lr=0.002, weight_decay=0.0)

    with torch.no_grad():
        assert hybrid(
            batch.surface_events,
            batch.variable_states,
            batch.incidence,
        ).branch_route_mask.all()

    _train_step(
        hybrid,
        optimizer,
        arm_id="hybrid",
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
        targets=batch.targets,
    )
    _legacy_hybrid_step_without_direct_stop_supervision(
        legacy,
        legacy_optimizer,
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
        targets=batch.targets,
    )

    # A hybrid router is trained from stop-path failure. If the stop path itself is
    # not directly supervised, an early weak stop prediction can self-lock the arm
    # into always-branch behavior. The production objective must therefore differ
    # from the legacy branch-only decision objective even when every episode routes.
    assert not torch.equal(
        hybrid.decision_head.weight.detach(),
        legacy.decision_head.weight.detach(),
    )
    assert not torch.equal(
        hybrid.decision_head.bias.detach(),
        legacy.decision_head.bias.detach(),
    )
