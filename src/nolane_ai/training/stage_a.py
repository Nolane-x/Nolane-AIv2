from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nolane_ai.model.nlm import NolaneLivingModel


@dataclass(slots=True)
class StageAMultitaskBatch:
    variable_states: torch.Tensor
    incidence: torch.Tensor
    belief_targets: torch.Tensor
    conflict_targets: torch.Tensor
    source_semantics: torch.Tensor
    candidate_semantics: torch.Tensor
    fidelity_targets: torch.Tensor

    def validate(self, d_model: int) -> None:
        if self.variable_states.ndim != 3 or self.variable_states.shape[-1] != d_model:
            raise ValueError("variable_states must be [batch, variables, d_model]")
        b, v, _ = self.variable_states.shape
        if self.incidence.ndim != 3 or self.incidence.shape[0] != b or self.incidence.shape[2] != v:
            raise ValueError("incidence must be [batch, constraints, variables]")
        c = self.incidence.shape[1]
        if self.belief_targets.shape != (b, v):
            raise ValueError("belief_targets must be [batch, variables]")
        if self.conflict_targets.shape != (b, c):
            raise ValueError("conflict_targets must be [batch, constraints]")
        if self.source_semantics.shape != (b, d_model) or self.candidate_semantics.shape != (b, d_model):
            raise ValueError("semantic tensors must be [batch, d_model]")
        if self.fidelity_targets.shape != (b,):
            raise ValueError("fidelity_targets must be [batch]")


def stage_a_multitask_loss(
    model: NolaneLivingModel,
    batch: StageAMultitaskBatch,
    *,
    belief_weight: float = 1.0,
    conflict_weight: float = 1.0,
    fidelity_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    batch.validate(model.config.d_model)
    output = model.structured_reason(batch.variable_states, batch.incidence)
    belief = F.cross_entropy(output.belief_logits.reshape(-1, 2), batch.belief_targets.reshape(-1))
    conflict = F.binary_cross_entropy_with_logits(output.conflict_scores, batch.conflict_targets)
    fidelity_prob = model.semantic_fidelity_score(batch.source_semantics, batch.candidate_semantics)
    fidelity = F.binary_cross_entropy(fidelity_prob, batch.fidelity_targets)
    total = belief_weight * belief + conflict_weight * conflict + fidelity_weight * fidelity
    return {"total": total, "belief": belief, "conflict": conflict, "fidelity": fidelity}
