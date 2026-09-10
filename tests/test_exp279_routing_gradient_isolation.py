from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import (
    ROUTING_SUPERVISION,
    _build_seeded_triplet,
    _train_step,
)
from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator
from nolane_ai.training.optimizer import build_functional_optimizer


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], prefix: str) -> bool:
    return any(
        not torch.equal(before[name], after[name])
        for name in before
        if name.startswith(prefix)
    )


def test_exp279_routing_supervision_contract_is_head_only() -> None:
    assert ROUTING_SUPERVISION["gradient_scope"] == "routing_head_only"
    assert ROUTING_SUPERVISION["shared_backbone_receives_routing_loss_gradient"] is False
    assert ROUTING_SUPERVISION["decision_head_receives_routing_loss_gradient"] is False


def test_exp279_routing_loss_does_not_distort_shared_decision_backbone() -> None:
    propagation_a, _, _, _ = _build_seeded_triplet(
        root_seed="exp279-gradient-isolation",
        d_model=16,
        hidden_size=12,
        target_parameters=12000,
        route_threshold=0.5,
    )
    propagation_b = deepcopy(propagation_a)
    generator = Exp279RoutingGenerator(root_seed="exp279-gradient-isolation")
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

    optimizer_a = build_functional_optimizer(propagation_a, lr=0.002, weight_decay=0.0)
    optimizer_b = build_functional_optimizer(propagation_b, lr=0.002, weight_decay=0.0)

    # Production step: decision CE plus routing calibration under the frozen scope.
    _train_step(
        propagation_a,
        optimizer_a,
        arm_id="propagation_only",
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
        targets=batch.targets,
    )

    # Decision-only baseline from identical initialization, world and optimizer.
    propagation_b.train()
    optimizer_b.zero_grad(set_to_none=True)
    output = propagation_b(batch.surface_events, batch.variable_states, batch.incidence)
    loss = torch.nn.functional.cross_entropy(
        output.decision_logits.reshape(-1, 2),
        batch.targets.reshape(-1),
    )
    loss.backward()
    optimizer_b.step()

    production = _state(propagation_a)
    decision_only = _state(propagation_b)

    routing_names = [name for name in production if name.startswith("routing_head.")]
    assert routing_names
    assert any(
        not torch.equal(production[name], decision_only[name])
        for name in routing_names
    ), "routing supervision must update routing_head beyond decision-only baseline"

    protected_prefixes = (
        "event_projection.",
        "variable_projection.",
        "branch_gru.",
        "propagation_projection.",
        "reclaimed_projection.",
        "decision_head.",
        "verifier_head.",
    )
    for prefix in protected_prefixes:
        names = [name for name in production if name.startswith(prefix)]
        assert names
        for name in names:
            assert torch.equal(production[name], decision_only[name]), (
                f"routing calibration leaked gradient into {name}"
            )

    assert torch.equal(production["mix_gate"], decision_only["mix_gate"])
