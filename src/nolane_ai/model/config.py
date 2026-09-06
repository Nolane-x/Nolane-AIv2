from __future__ import annotations

from dataclasses import dataclass

from .budget import ModelBudget, RegionBudget, authoritative_v016_budget


_STAGE_A_PILOT_16M_REGIONS: tuple[RegionBudget, ...] = (
    RegionBudget("language_perception_binding", 2_000_000),
    RegionBudget("recurrent_deliberation_core", 4_000_000),
    RegionBudget("constraint_belief_fabric", 4_000_000),
    RegionBudget("conflict_core_backjump_clause", 2_000_000),
    RegionBudget("problem_compiler_fidelity_court", 2_000_000),
    RegionBudget("verifier_proof_counterexample_heads", 1_000_000),
    RegionBudget("developmental_reserve", 1_000_000),
)


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
    def stage_a_pilot_16m(cls) -> "NLMConfig":
        budget = ModelBudget(
            version="stage-a-pilot-16m-v1",
            regions=_STAGE_A_PILOT_16M_REGIONS,
        )
        if budget.total_parameters != 16_000_000:
            raise RuntimeError(f"Stage-A pilot budget drifted: {budget.total_parameters:,}")
        return cls(
            vocab_size=4096,
            d_model=256,
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

    @classmethod
    def stage_a_pilot_tiny_for_tests(cls) -> "NLMConfig":
        region_names = tuple(region.name for region in _STAGE_A_PILOT_16M_REGIONS)
        regions = tuple(RegionBudget(name, 5_000) for name in region_names)
        return cls(
            vocab_size=64,
            d_model=16,
            budget=ModelBudget(version="stage-a-pilot-tiny-test", regions=regions),
            active_region_names=tuple(name for name in region_names if name != "developmental_reserve"),
        )
