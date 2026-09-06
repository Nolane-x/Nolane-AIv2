from importlib import import_module


def _budget_module():
    module = import_module("nolane_ai.model.budget")
    assert hasattr(module, "authoritative_v016_budget")
    return module


def test_authoritative_budget_is_exactly_100m():
    budget = _budget_module().authoritative_v016_budget()
    assert budget.total_parameters == 100_000_000


def test_budget_region_names_are_unique_and_positive():
    budget = _budget_module().authoritative_v016_budget()
    names = [region.name for region in budget.regions]
    assert len(names) == len(set(names))
    assert all(region.parameters > 0 for region in budget.regions)


def test_frozen_support_is_explicitly_10m():
    budget = _budget_module().authoritative_v016_budget()
    assert budget.frozen_parameters == 10_000_000
    assert budget.trainable_parameters == 90_000_000
