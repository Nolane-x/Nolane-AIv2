from __future__ import annotations

from dataclasses import dataclass

from .budget import ModelBudget, RegionBudget, authoritative_v016_budget


@dataclass(frozen=True, slots=True)
class NLMConfig:
    vocab_size: int
    d_model: int
    budget: ModelBudget
    active_region_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        known = {region.name for region in self.budget.regions}
        unknown = set(self.active_region_names) - known
        if unknown:
            raise ValueError(f"unknown active regions: {sorted(unknown)}")

    @classmethod
    def authoritative_100m(cls) -> "NLMConfig":
        budget = authoritative_v016_budget()
        return cls(
            vocab_size=8192,
            d_model=512,
            budget=budget,
            active_region_names=tuple(
                region.name for region in budget.regions if region.name != "developmental_reserve"
            ),
        )

    @classmethod
    def tiny_for_tests(cls) -> "NLMConfig":
        regions = tuple(
            RegionBudget(region.name, 5_000, frozen=region.frozen)
            for region in authoritative_v016_budget().regions
        )
        return cls(
            vocab_size=64,
            d_model=16,
            budget=ModelBudget(version="tiny-test", regions=regions),
            active_region_names=tuple(
                region.name for region in regions if region.name != "developmental_reserve"
            ),
        )
