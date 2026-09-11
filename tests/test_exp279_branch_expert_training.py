from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import _build_seeded_triplet, _train_step
from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator
from nolane_ai.training.optimizer import build_functional_optimizer


def _branch_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("branch_gru.")
    }


def test_hybrid_branch_expert_trains_even_when_current_router_abstains() -> None:
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-branch-expert-training",
        d_model=16,
        hidden_size=12,
        target_parameters=12000,
        route_threshold=1.0,
    )
    generator = Exp279RoutingGenerator(root_seed="exp279-branch-expert-training")
    batch = generator.make_batch(
        replicate=0,
        batch_size=8,
        timesteps=3,
        variables=4,
        constraints=2,
        d_model=16,
        noise_std=0.05,
        rng_stream="augmentation",
        stratum="MIXED_RESIDUAL",
    )

    hybrid.eval()
    with torch.no_grad():
        before_output = hybrid(
            batch.surface_events,
            batch.variable_states,
            batch.incidence,
        )
    assert not bool(before_output.branch_route_mask.any().item())

    before = _branch_state(hybrid)
    optimizer = build_functional_optimizer(hybrid, lr=0.002, weight_decay=0.0)
    _train_step(
        hybrid,
        optimizer,
        arm_id="hybrid",
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
        targets=batch.targets,
    )
    after = _branch_state(hybrid)

    assert before
    assert any(
        not torch.equal(before[name], after[name])
        for name in before
    ), (
        "the forced branch expert must receive DEVELOPMENT decision supervision even "
        "when the current router abstains; otherwise rescue labels self-extinguish"
    )
