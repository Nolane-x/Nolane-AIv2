from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.optimizer import functional_trainable_named_parameters


TARGET_MODES = (
    "LOCAL_ONLY_CONTROL",
    "LEARNED_STRUCTURAL_TRANSFER",
    "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
)


@dataclass(frozen=True, slots=True)
class Exp290ReasonerOutput:
    branch_logits: torch.Tensor
    verifier_confidence: torch.Tensor
    variable_representations: torch.Tensor
    recurrent_state: torch.Tensor


class Exp290ClauseTransferModel(nn.Module):
    """One matched reasoner plus a model-visible source→target transfer scorer."""

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        target_parameters: int,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-290 dimensions and target_parameters must be positive")
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.deliberation_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.local_memory_adapter = nn.Linear(1, hidden_size, device=device)
        self.branch_head = nn.Linear(hidden_size, 1, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)

        self.literal_embedding = nn.Embedding(2, hidden_size, device=device)
        self.transfer_source = nn.Linear(hidden_size * 2, hidden_size, device=device)
        self.transfer_target = nn.Linear(hidden_size, hidden_size, device=device)
        self.transfer_bias = nn.Parameter(torch.zeros((), device=device))
        self.reasoner_mix_gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=False)

    def encode_reasoner_state(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        *,
        local_memory_hit: torch.Tensor | None = None,
    ) -> Exp290ReasonerOutput:
        if surface_events.ndim != 3 or surface_events.shape[-1] != self.d_model:
            raise ValueError("surface_events must be [batch,time,d_model]")
        if variable_states.ndim != 3 or variable_states.shape[-1] != self.d_model:
            raise ValueError("variable_states must be [batch,variables,d_model]")
        if surface_events.shape[0] != variable_states.shape[0]:
            raise ValueError("surface_events and variable_states batch dimensions must match")
        events = F.silu(self.event_projection(surface_events))
        variables = F.silu(self.variable_projection(variable_states))
        _, hidden = self.deliberation_gru(events)
        recurrent = hidden[-1]
        recurrent_expanded = recurrent.unsqueeze(1).expand(-1, variables.shape[1], -1)

        if local_memory_hit is None:
            hit = torch.zeros(
                variables.shape[0],
                variables.shape[1],
                1,
                device=variables.device,
                dtype=variables.dtype,
            )
        else:
            if local_memory_hit.ndim == 1:
                raw = local_memory_hit[:, None, None]
            elif local_memory_hit.ndim == 2 and local_memory_hit.shape[1] == variables.shape[1]:
                raw = local_memory_hit.unsqueeze(-1)
            else:
                raise ValueError("local_memory_hit must be [batch] or [batch,variables]")
            hit = raw.to(device=variables.device, dtype=variables.dtype)
            hit = hit.expand(-1, variables.shape[1], 1) if hit.shape[1] == 1 else hit
        memory_state = F.silu(self.local_memory_adapter(hit))
        mixed = variables + recurrent_expanded + torch.sigmoid(self.reasoner_mix_gate) * memory_state
        return Exp290ReasonerOutput(
            branch_logits=self.branch_head(mixed).squeeze(-1),
            verifier_confidence=torch.sigmoid(self.verifier_head(mixed).squeeze(-1)),
            variable_representations=variables,
            recurrent_state=recurrent,
        )

    def score_transfer(
        self,
        *,
        source_variable_states: torch.Tensor,
        target_variable_states: torch.Tensor,
        source_variable_index: torch.Tensor,
        literal_value: torch.Tensor,
    ) -> torch.Tensor:
        if source_variable_states.ndim != 3 or source_variable_states.shape[-1] != self.d_model:
            raise ValueError("source_variable_states must be [batch,variables,d_model]")
        if target_variable_states.ndim != 3 or target_variable_states.shape[-1] != self.d_model:
            raise ValueError("target_variable_states must be [batch,variables,d_model]")
        if source_variable_states.shape[0] != target_variable_states.shape[0]:
            raise ValueError("source and target batch dimensions must match")
        batch = source_variable_states.shape[0]
        if source_variable_index.shape != (batch,):
            raise ValueError("source_variable_index must be [batch]")
        if literal_value.shape != (batch,):
            raise ValueError("literal_value must be [batch]")
        if bool(((literal_value < 0) | (literal_value > 1)).any().item()):
            raise ValueError("literal_value must use frozen binary alphabet {0,1}")
        if bool(((source_variable_index < 0) | (source_variable_index >= source_variable_states.shape[1])).any().item()):
            raise ValueError("source_variable_index is out of bounds")

        source_repr = F.silu(self.variable_projection(source_variable_states))
        target_repr = F.silu(self.variable_projection(target_variable_states))
        gather = source_variable_index.to(device=source_repr.device, dtype=torch.long).view(batch, 1, 1)
        gather = gather.expand(-1, 1, self.hidden_size)
        chosen_source = source_repr.gather(1, gather).squeeze(1)
        literal = self.literal_embedding(literal_value.to(device=source_repr.device, dtype=torch.long))
        query = F.silu(self.transfer_source(torch.cat([chosen_source, literal], dim=-1)))
        keys = F.silu(self.transfer_target(target_repr))
        scale = float(self.hidden_size) ** -0.5
        return torch.einsum("bh,bvh->bv", query, keys) * scale + self.transfer_bias


def build_matched_exp290_model(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> Exp290ClauseTransferModel:
    return Exp290ClauseTransferModel(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        device=device,
    )


def deterministic_transfer_top1(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] <= 0:
        raise ValueError("transfer scores must be [batch,target_variables]")
    if not torch.isfinite(scores).all():
        raise ValueError("transfer scores must be finite")
    # torch.argmax returns the first maximal index, implementing the frozen
    # ascending-target-index tie break without randomized jitter.
    return torch.argmax(scores, dim=1)


def _parameter_audit(model: Exp290ClauseTransferModel) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    reserve = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "capacity_reserve" in name
    )
    functional = total - reserve
    optimizer_visible = sum(
        parameter.numel() for _, parameter in functional_trainable_named_parameters(model)
    )
    return {
        "total_parameters": int(total),
        "functional_parameters": int(functional),
        "active_functional_parameters": int(functional),
        "optimizer_visible_parameters": int(optimizer_visible),
        "reserved_parameters": int(reserve),
    }


def _linear_flops(inputs: int, outputs: int) -> int:
    return 2 * inputs * outputs + outputs


def audit_exp290_transfer_model(
    model: Exp290ClauseTransferModel,
    *,
    variables: int,
    source_clause_slots: int,
) -> dict[str, Any]:
    if variables <= 0 or source_clause_slots <= 0:
        raise ValueError("EXP-290 audit geometry must be positive")
    audit = _parameter_audit(model)
    h = model.hidden_size
    d = model.d_model
    source_and_target_projection = (variables * 2) * _linear_flops(d, h)
    source_literal_query = _linear_flops(h * 2, h)
    target_key_projection = variables * _linear_flops(h, h)
    dot_scores = variables * (2 * h)
    scorer_flops = source_and_target_projection + source_literal_query + target_key_projection + dot_scores
    return {
        **audit,
        "schema": "NLM-EXP-290-MATCHED-CLAUSE-TRANSFER-MODEL-V1",
        "target_modes": list(TARGET_MODES),
        "mode_parameter_inventory_equal": True,
        "transfer_scorer_executed_in_all_modes": True,
        "oracle_mapping_is_model_input": False,
        "evaluation_mapping_is_learned_input": False,
        "per_source_clause_transfer_scoring_flops": int(scorer_flops),
        "transferred_slot_comparisons_per_target_query": int(source_clause_slots),
        "fixed_slot_accounting": True,
        "value_labels_remapped": False,
        "variables": int(variables),
    }
