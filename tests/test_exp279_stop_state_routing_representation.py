from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import _build_seeded_triplet
from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator


def test_hybrid_routes_from_the_same_stop_state_the_teacher_is_judging() -> None:
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-stop-state-routing-representation",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
        route_threshold=0.5,
    )
    generator = Exp279RoutingGenerator(root_seed="exp279-stop-state-routing-representation")
    batch = generator.make_batch(
        replicate=0,
        batch_size=4,
        timesteps=3,
        variables=4,
        constraints=2,
        d_model=16,
        noise_std=0.05,
        rng_stream="augmentation",
        stratum="MIXED_RESIDUAL",
    )

    routing_inputs: list[torch.Tensor] = []

    def capture_routing_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        routing_inputs.append(inputs[0].detach().clone())

    handle = hybrid.routing_head.register_forward_pre_hook(capture_routing_input)
    try:
        with torch.no_grad():
            hybrid(batch.surface_events, batch.variable_states, batch.incidence)
    finally:
        handle.remove()

    with torch.no_grad():
        _, variables = hybrid._validate_common(batch.surface_events, batch.variable_states)
        incidence = hybrid._validate_incidence(batch.incidence, variables)
        propagated = hybrid._propagate(variables, incidence)
        propagation_state = variables + propagated
        reclaimed = torch.tanh(hybrid.reclaimed_projection(propagation_state))
        stop_state = propagation_state + reclaimed

    assert len(routing_inputs) == 1
    assert not torch.allclose(stop_state, propagation_state)
    assert torch.allclose(routing_inputs[0], stop_state)
