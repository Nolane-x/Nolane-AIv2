from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegionBudget:
    name: str
    parameters: int
    frozen: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("region name must be non-empty")
        if self.parameters <= 0:
            raise ValueError(f"region {self.name!r} must have a positive parameter budget")


@dataclass(frozen=True, slots=True)
class ModelBudget:
    version: str
    regions: tuple[RegionBudget, ...]

    def __post_init__(self) -> None:
        names = [region.name for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("region names must be unique")

    @property
    def total_parameters(self) -> int:
        return sum(region.parameters for region in self.regions)

    @property
    def frozen_parameters(self) -> int:
        return sum(region.parameters for region in self.regions if region.frozen)

    @property
    def trainable_parameters(self) -> int:
        return self.total_parameters - self.frozen_parameters

    def by_name(self, name: str) -> RegionBudget:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


AUTHORITATIVE_V016_REGIONS: tuple[RegionBudget, ...] = (
    RegionBudget("language_perception_binding", 7_000_000),
    RegionBudget("stable_mechanism_semantic_cortex", 7_000_000),
    RegionBudget("frozen_hbae_support", 5_000_000, frozen=True),
    RegionBudget("pcsa_mcra_world_state", 10_000_000),
    RegionBudget("recurrent_deliberation_core", 15_000_000),
    RegionBudget("constraint_belief_fabric", 11_000_000),
    RegionBudget("conflict_core_backjump_clause", 6_000_000),
    RegionBudget("lemma_invariant_economy", 5_000_000),
    RegionBudget("latent_hypothesis_bank", 5_000_000),
    RegionBudget("problem_compiler_fidelity_court", 6_000_000),
    RegionBudget("verifier_proof_counterexample_heads", 6_000_000),
    RegionBudget("cognitive_isa_program_bridge", 4_000_000),
    RegionBudget("mvtc_meta_reasoning_controller", 3_000_000),
    RegionBudget("frozen_consolidation_meta_updater", 5_000_000, frozen=True),
    RegionBudget("developmental_reserve", 5_000_000),
)


def authoritative_v016_budget() -> ModelBudget:
    budget = ModelBudget(version="V0.16.1-authority-map", regions=AUTHORITATIVE_V016_REGIONS)
    if budget.total_parameters != 100_000_000:
        raise RuntimeError(f"authoritative budget drifted: {budget.total_parameters:,}")
    return budget
