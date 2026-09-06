import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.training.curriculum import StageACurriculum
from nolane_ai.training.stage_a import stage_a_multitask_loss
from nolane_ai.model.nlm import NolaneLivingModel


def test_curriculum_is_deterministic_for_same_seed_and_replicate():
    curriculum = StageACurriculum(root_seed="20260906")
    a = curriculum.make_batch(replicate=3, batch_size=4, variables=5, constraints=3, d_model=16)
    b = curriculum.make_batch(replicate=3, batch_size=4, variables=5, constraints=3, d_model=16)
    assert a.digest == b.digest
    assert a.metadata == b.metadata
    for field in (
        "variable_states",
        "incidence",
        "belief_targets",
        "conflict_targets",
        "source_semantics",
        "candidate_semantics",
        "fidelity_targets",
    ):
        assert torch.equal(getattr(a.batch, field), getattr(b.batch, field))


def test_curriculum_changes_when_replicate_changes():
    curriculum = StageACurriculum(root_seed="20260906")
    a = curriculum.make_batch(replicate=0, batch_size=4, variables=5, constraints=3, d_model=16)
    b = curriculum.make_batch(replicate=1, batch_size=4, variables=5, constraints=3, d_model=16)
    assert a.digest != b.digest
    assert not torch.equal(a.batch.variable_states, b.batch.variable_states)


def test_curriculum_batch_matches_stage_a_loss_contract():
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    model = NolaneLivingModel(config)
    batch = StageACurriculum(root_seed="loss-contract").make_batch(
        replicate=0,
        batch_size=3,
        variables=4,
        constraints=3,
        d_model=config.d_model,
    )
    losses = stage_a_multitask_loss(model, batch.batch)
    assert set(losses) == {"total", "belief", "conflict", "fidelity"}
    assert torch.isfinite(losses["total"])
    assert batch.metadata["scope"] == "synthetic-stage-a-development-curriculum"


def test_curriculum_digest_does_not_require_numpy(monkeypatch):
    def _forbid_numpy(self):
        raise AssertionError("curriculum digest must not depend on Tensor.numpy")

    monkeypatch.setattr(torch.Tensor, "numpy", _forbid_numpy, raising=True)
    batch = StageACurriculum(root_seed="no-numpy").make_batch(
        replicate=0, batch_size=2, variables=3, constraints=2, d_model=8
    )
    assert len(batch.digest) == 64
