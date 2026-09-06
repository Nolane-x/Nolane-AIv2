import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from nolane_ai.training.stage_a import StageAMultitaskBatch, stage_a_multitask_loss


def _batch(d_model: int) -> StageAMultitaskBatch:
    return StageAMultitaskBatch(
        variable_states=torch.randn(2, 3, d_model),
        incidence=torch.tensor(
            [
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        ),
        belief_targets=torch.tensor([[0, 1, 0], [1, 0, 1]]),
        conflict_targets=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        source_semantics=torch.randn(2, d_model),
        candidate_semantics=torch.randn(2, d_model),
        fidelity_targets=torch.tensor([1.0, 0.0]),
    )


def test_functional_parameter_selector_excludes_all_capacity_reserves():
    model = NolaneLivingModel(NLMConfig.stage_a_pilot_tiny_for_tests())
    selected = list(functional_trainable_named_parameters(model))
    assert selected
    assert all("capacity_reserve" not in name for name, _ in selected)
    selected_ids = {id(parameter) for _, parameter in selected}
    reserve_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if "capacity_reserve" in name
    }
    assert selected_ids.isdisjoint(reserve_ids)


def test_optimizer_step_cannot_mutate_capacity_reserve_or_give_it_gradient():
    torch.manual_seed(7)
    model = NolaneLivingModel(NLMConfig.stage_a_pilot_tiny_for_tests())
    optimizer = build_functional_optimizer(model, lr=1e-3, weight_decay=0.0)
    reserves_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "capacity_reserve" in name
    }

    losses = stage_a_multitask_loss(model, _batch(model.config.d_model))
    losses["total"].backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        if "capacity_reserve" not in name:
            continue
        assert parameter.grad is None
        assert torch.equal(parameter.detach(), reserves_before[name])
