import pytest

pytest.importorskip("torch")

from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel


def test_model_audit_separates_functional_parameters_from_explicit_reserve():
    model = NolaneLivingModel(NLMConfig.authoritative_100m(), device="meta")
    report = audit_model(model)
    assert report.total_parameters == 100_000_000
    assert report.trainable_parameters == 90_000_000
    assert report.reserved_parameters > 0
    assert report.functional_parameters + report.reserved_parameters == report.total_parameters
    for name in (
        "recurrent_deliberation_core",
        "constraint_belief_fabric",
        "conflict_core_backjump_clause",
        "problem_compiler_fidelity_court",
    ):
        assert report.regions[name].functional_parameters > 0
        assert report.regions[name].reserved_parameters >= 0
