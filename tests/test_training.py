import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.training.stage_a import StageAMultitaskBatch, stage_a_multitask_loss


def test_stage_a_multitask_loss_backpropagates_only_through_functional_paths():
    model = NolaneLivingModel(NLMConfig.tiny_for_tests(), device="cpu")
    b, v, c, d = 2, 4, 3, model.config.d_model
    variable_states = torch.randn(b, v, d)
    incidence = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
        ], dtype=torch.float32,
    )
    batch = StageAMultitaskBatch(
        variable_states=variable_states,
        incidence=incidence,
        belief_targets=torch.randint(0, 2, (b, v)),
        conflict_targets=torch.randint(0, 2, (b, c)).float(),
        source_semantics=torch.randn(b, d),
        candidate_semantics=torch.randn(b, d),
        fidelity_targets=torch.randint(0, 2, (b,)).float(),
    )
    losses = stage_a_multitask_loss(model, batch)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert any(p.grad is not None for name, p in model.named_parameters() if "capacity_reserve" not in name)
    assert all(p.grad is None for name, p in model.named_parameters() if "capacity_reserve" in name)
