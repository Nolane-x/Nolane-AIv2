import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.model.stage_a_regions import (
    ConflictCoreRegion,
    ConstraintBeliefFabricRegion,
    FidelityCourtRegion,
    RecurrentDeliberationRegion,
)


def test_authoritative_100m_uses_specialized_stage_a_regions_without_budget_drift():
    model = NolaneLivingModel(NLMConfig.authoritative_100m(), device="meta")
    assert sum(p.numel() for p in model.parameters()) == 100_000_000
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 90_000_000
    assert isinstance(model.regions["recurrent_deliberation_core"], RecurrentDeliberationRegion)
    assert isinstance(model.regions["constraint_belief_fabric"], ConstraintBeliefFabricRegion)
    assert isinstance(model.regions["conflict_core_backjump_clause"], ConflictCoreRegion)
    assert isinstance(model.regions["problem_compiler_fidelity_court"], FidelityCourtRegion)


def test_tiny_structured_reasoning_path_returns_beliefs_and_conflict_scores():
    model = NolaneLivingModel(NLMConfig.tiny_for_tests(), device="cpu")
    variable_states = torch.randn(2, 4, model.config.d_model)
    incidence = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
        ],
        dtype=torch.float32,
    )
    output = model.structured_reason(variable_states, incidence)
    assert output.variable_states.shape == variable_states.shape
    assert output.belief_logits.shape == (2, 4, 2)
    assert output.conflict_scores.shape == (2, 3)


def test_tiny_fidelity_score_is_probability():
    model = NolaneLivingModel(NLMConfig.tiny_for_tests(), device="cpu")
    source = torch.randn(3, model.config.d_model)
    candidate = torch.randn(3, model.config.d_model)
    score = model.semantic_fidelity_score(source, candidate)
    assert score.shape == (3,)
    assert torch.all((score >= 0) & (score <= 1))
