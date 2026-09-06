from nolane_ai.model.budget import authoritative_v016_budget
from nolane_ai.model.config import NLMConfig


def test_stage_a_pilot_config_is_exactly_16m_and_uses_only_gate_regions():
    config = NLMConfig.stage_a_pilot_16m()
    assert config.budget.total_parameters == 16_000_000
    assert config.budget.frozen_parameters == 0
    assert tuple(region.name for region in config.budget.regions) == (
        "language_perception_binding",
        "recurrent_deliberation_core",
        "constraint_belief_fabric",
        "conflict_core_backjump_clause",
        "problem_compiler_fidelity_court",
        "verifier_proof_counterexample_heads",
        "developmental_reserve",
    )
    assert config.active_region_names == tuple(
        region.name for region in config.budget.regions if region.name != "developmental_reserve"
    )


def test_stage_a_pilot_does_not_mutate_authoritative_100m_budget():
    before = authoritative_v016_budget()
    _ = NLMConfig.stage_a_pilot_16m()
    after = authoritative_v016_budget()
    assert before == after
    assert after.total_parameters == 100_000_000


def test_stage_a_pilot_model_is_exactly_16m_on_meta():
    torch = __import__("torch")
    from nolane_ai.model.nlm import NolaneLivingModel

    model = NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta")
    assert sum(parameter.numel() for parameter in model.parameters()) == 16_000_000
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 16_000_000
