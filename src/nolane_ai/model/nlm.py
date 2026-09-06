from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import NLMConfig
from .regions import BudgetedResidualRegion


class NolaneLivingModel(nn.Module):
    """Executable parameter-budget-faithful NLM V0.16.1 scaffold.

    This is a research substrate, not an empirical capability claim. Region reserves
    are progressively replaced by evidence-earning neural mechanisms as Stage-A gates pass.
    """

    def __init__(self, config: NLMConfig, *, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        language = config.budget.by_name("language_perception_binding")
        embedding_parameters = config.vocab_size * config.d_model
        language_remaining = language.parameters - embedding_parameters
        if language_remaining <= 0:
            raise ValueError("language budget cannot fit token embedding")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, device=device)
        self.language_adapter = BudgetedResidualRegion(
            config.d_model,
            language_remaining,
            device=device,
            frozen=language.frozen,
        )

        regions: dict[str, BudgetedResidualRegion] = {}
        for spec in config.budget.regions:
            if spec.name == "language_perception_binding":
                continue
            regions[spec.name] = BudgetedResidualRegion(
                config.d_model,
                spec.parameters,
                device=device,
                frozen=spec.frozen,
                functional=spec.name != "developmental_reserve",
            )
        self.regions = nn.ModuleDict(regions)
        self.output_norm = nn.LayerNorm(config.d_model, elementwise_affine=False, device=device)

        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != config.budget.total_parameters:
            raise RuntimeError(
                f"model count mismatch: expected {config.budget.total_parameters:,}, got {actual:,}"
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
