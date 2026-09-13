from __future__ import annotations

from nolane_ai.model.budget import (
    V017_CAPACITY_EXCHANGE_ENVELOPE,
    V017_FROZEN_BASE_PARAMETERS,
    authoritative_v017_10m_budget,
)


def test_v017_budget_closes_exactly_at_ten_million() -> None:
    budget = authoritative_v017_10m_budget()

    assert V017_FROZEN_BASE_PARAMETERS == 9_120_832
    assert V017_CAPACITY_EXCHANGE_ENVELOPE == 879_168
    assert V017_FROZEN_BASE_PARAMETERS + V017_CAPACITY_EXCHANGE_ENVELOPE == 10_000_000
    assert budget.total_parameters == 10_000_000
    assert budget.trainable_parameters == 10_000_000


def test_v017_budget_is_scientifically_distinct_from_v016_regions() -> None:
    budget = authoritative_v017_10m_budget()

    assert budget.version == "v0.17-exp301-10m-native-recursive"
    assert tuple(region.name for region in budget.regions) == (
        "native_recursive_base",
        "capacity_exchange_envelope",
    )
    assert all(not region.frozen for region in budget.regions)
