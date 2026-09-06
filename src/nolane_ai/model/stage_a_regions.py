from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .regions import finalize_region_budget


class RecurrentDeliberationRegion(nn.Module):
    def __init__(self, d_model: int, target_parameters: int, *, device=None, frozen: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, device=device)
        self.gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.gru(self.norm(x))
        return x + torch.sigmoid(self.gate) * recurrent

    def deliberate(self, state: torch.Tensor, *, steps: int) -> torch.Tensor:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        current = state
        hidden = state.unsqueeze(0)
        for _ in range(steps):
            out, hidden = self.gru(self.norm(current).unsqueeze(1), hidden)
            current = out[:, 0]
        return current


class ConstraintBeliefFabricRegion(nn.Module):
    def __init__(self, d_model: int, target_parameters: int, *, device=None, frozen: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.residual = nn.Linear(d_model, d_model, device=device)
        self.constraint_projection = nn.Linear(d_model, d_model, device=device)
        self.variable_projection = nn.Linear(d_model, d_model, device=device)
        self.belief_head = nn.Linear(d_model, 2, device=device)
        self.gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.gate) * F.silu(self.residual(self.norm(x)))

    def reason(self, variable_states: torch.Tensor, incidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if variable_states.ndim != 3 or incidence.ndim != 3:
            raise ValueError("variable_states and incidence must be rank-3")
        if variable_states.shape[0] != incidence.shape[0] or variable_states.shape[1] != incidence.shape[2]:
            raise ValueError("incidence must have shape [batch,constraints,variables]")
        normalized = self.norm(variable_states)
        c_degree = incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)
        constraint_summary = torch.bmm(incidence, normalized) / c_degree
        constraint_states = torch.tanh(self.constraint_projection(constraint_summary))
        transpose = incidence.transpose(1, 2)
        v_degree = transpose.sum(dim=-1, keepdim=True).clamp_min(1.0)
        back_message = torch.bmm(transpose, constraint_states) / v_degree
        update = torch.tanh(self.variable_projection(back_message))
        updated = variable_states + torch.sigmoid(self.gate) * update
        belief_logits = self.belief_head(updated)
        return updated, belief_logits, constraint_states


class ConflictCoreRegion(nn.Module):
    def __init__(self, d_model: int, target_parameters: int, *, device=None, frozen: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.residual = nn.Linear(d_model, d_model, device=device)
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model, device=device), nn.SiLU(), nn.Linear(d_model, 1, device=device))
        self.gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.gate) * F.silu(self.residual(self.norm(x)))

    def score_conflicts(self, constraint_states: torch.Tensor) -> torch.Tensor:
        return self.scorer(self.norm(constraint_states)).squeeze(-1)


class FidelityCourtRegion(nn.Module):
    def __init__(self, d_model: int, target_parameters: int, *, device=None, frozen: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.residual = nn.Linear(d_model, d_model, device=device)
        self.pair_head = nn.Sequential(nn.Linear(4 * d_model, d_model, device=device), nn.SiLU(), nn.Linear(d_model, 1, device=device))
        self.gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.gate) * F.silu(self.residual(self.norm(x)))

    def fidelity_score(self, source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if source.shape != candidate.shape or source.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("source and candidate must have matching [..., d_model] shape")
        source_n = self.norm(source)
        candidate_n = self.norm(candidate)
        features = torch.cat((source_n, candidate_n, torch.abs(source_n - candidate_n), source_n * candidate_n), dim=-1)
        return torch.sigmoid(self.pair_head(features).squeeze(-1))


@dataclass(slots=True)
class StructuredReasoningOutput:
    variable_states: torch.Tensor
    belief_logits: torch.Tensor
    conflict_scores: torch.Tensor
    constraint_states: torch.Tensor
