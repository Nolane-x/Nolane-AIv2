from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import NLMConfig
from .regions import BudgetedResidualRegion
from .stage_a_regions import (
    ConflictCoreRegion,
    ConstraintBeliefFabricRegion,
    FidelityCourtRegion,
    RecurrentDeliberationRegion,
    StructuredReasoningOutput,
)


class NolaneLivingModel(nn.Module):
    """Exact-budget NLM V0.16.1 candidate with executable Stage-A regions.

    Specialized modules expose real neural interfaces but remain UNVERIFIED until
    the frozen neural experiments run. Unearned regional capacity stays explicit.
    """

    def __init__(
        self,
        config: NLMConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        language = config.budget.by_name("language_perception_binding")
        embedding_parameters = config.vocab_size * config.d_model
        language_remaining = language.parameters - embedding_parameters
        if language_remaining <= 0:
            raise ValueError("language budget cannot fit token embedding")

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            device=device,
        )
        self.language_adapter = BudgetedResidualRegion(
            config.d_model,
            language_remaining,
            device=device,
            frozen=language.frozen,
        )

        regions: dict[str, nn.Module] = {}
        for spec in config.budget.regions:
            if spec.name == "language_perception_binding":
                continue
            region = self._build_region(spec.name, spec.parameters, spec.frozen, device)
            regions[spec.name] = region
        self.regions = nn.ModuleDict(regions)
        self.output_norm = nn.LayerNorm(
            config.d_model,
            elementwise_affine=False,
            device=device,
        )

        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != config.budget.total_parameters:
            raise RuntimeError(
                f"model count mismatch: expected {config.budget.total_parameters:,}, "
                f"got {actual:,}"
            )

    def _build_region(
        self,
        name: str,
        parameters: int,
        frozen: bool,
        device: str | torch.device | None,
    ) -> nn.Module:
        kwargs = {
            "d_model": self.config.d_model,
            "target_parameters": parameters,
            "device": device,
            "frozen": frozen,
        }
        if name == "recurrent_deliberation_core":
            return RecurrentDeliberationRegion(**kwargs)
        if name == "constraint_belief_fabric":
            return ConstraintBeliefFabricRegion(**kwargs)
        if name == "conflict_core_backjump_clause":
            return ConflictCoreRegion(**kwargs)
        if name == "problem_compiler_fidelity_court":
            return FidelityCourtRegion(**kwargs)
        return BudgetedResidualRegion(
            self.config.d_model,
            parameters,
            device=device,
            frozen=frozen,
            functional=name != "developmental_reserve",
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(token_ids)
        hidden = self.language_adapter(hidden)
        for name in self.config.active_region_names:
            if name == "language_perception_binding":
                continue
            hidden = self.regions[name](hidden)
        hidden = self.output_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)

    def structured_reason(
        self,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
    ) -> StructuredReasoningOutput:
        fabric = self.regions["constraint_belief_fabric"]
        conflict = self.regions["conflict_core_backjump_clause"]
        assert isinstance(fabric, ConstraintBeliefFabricRegion)
        assert isinstance(conflict, ConflictCoreRegion)
        updated, belief_logits, constraint_states = fabric.reason(
            variable_states,
            incidence,
        )
        conflict_scores = conflict.score_conflicts(constraint_states)
        return StructuredReasoningOutput(
            variable_states=updated,
            belief_logits=belief_logits,
            conflict_scores=conflict_scores,
            constraint_states=constraint_states,
        )

    def semantic_fidelity_score(
        self,
        source: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        court = self.regions["problem_compiler_fidelity_court"]
        assert isinstance(court, FidelityCourtRegion)
        return court.fidelity_score(source, candidate)
