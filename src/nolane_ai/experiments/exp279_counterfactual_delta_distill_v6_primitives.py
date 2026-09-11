from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256

from .matched_routing_arms import HybridRoutingArm

SCHEMA = "NLM-EXP-279-COUNTERFACTUAL-DELTA-DISTILL-COURT-V6"
PRIMARY_FAMILY = "CDD_DELTA_LINEAR"
CONTROL_FAMILY = "RAW_CHEAP_LINEAR"
PROBE_ROOT_SUFFIX = "::independent-augmentation-cdd-v6"


def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("pairwise selector scores and labels must align")
    positive = scores[labels > 0]
    negative = scores[labels <= 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.sum() * 0.0
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def class_counts(labels: torch.Tensor) -> dict[str, int]:
    labels = labels.to(torch.long)
    positive = int((labels == 1).sum().item())
    total = int(labels.numel())
    return {"positive": positive, "negative": total - positive, "total": total}


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.long)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    diff = positive[:, None] - negative[None, :]
    pairs = positive.numel() * negative.numel()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()).item() / pairs)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.long)
    positives = int((labels == 1).sum().item())
    if positives == 0:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order]
    cumulative = torch.cumsum(ranked.to(torch.float64), dim=0)
    precision = cumulative / torch.arange(1, ranked.numel() + 1, dtype=torch.float64)
    return float(precision[ranked == 1].sum().item() / positives)


def cheap_features(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    events, variables = arm._validate_common(surface_events, variable_states)
    checked = arm._validate_incidence(incidence, variables)
    propagation_state = variables + arm._propagate(variables, checked)
    p_mean = propagation_state.mean(dim=1)
    e_mean = events.mean(dim=1)
    e_delta = events[:, -1] - events[:, 0]
    return torch.cat((p_mean, e_mean, e_delta), dim=-1), p_mean


def stop_and_branch_states(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    events, variables = arm._validate_common(surface_events, variable_states)
    checked = arm._validate_incidence(incidence, variables)
    propagation_state = variables + arm._propagate(variables, checked)
    reclaimed = torch.tanh(arm.reclaimed_projection(propagation_state))
    stop_state = propagation_state + reclaimed
    branch_context = arm._branch_context(events).unsqueeze(1).expand(-1, variables.shape[1], -1)
    branch_state = propagation_state + torch.sigmoid(arm.mix_gate) * (branch_context + reclaimed)
    return stop_state, branch_state


def teacher_delta(stop_state: torch.Tensor, branch_state: torch.Tensor) -> torch.Tensor:
    if stop_state.ndim != 3 or branch_state.shape != stop_state.shape:
        raise ValueError("V6 stop/branch states must align")
    return (branch_state.detach() - stop_state.detach()).mean(dim=1)


def cdd_costs(*, hidden_size: int, variables: int, timesteps: int) -> dict[str, int]:
    if min(hidden_size, variables, timesteps) <= 0:
        raise ValueError("V6 CDD cost geometry must be positive")
    summary = variables * hidden_size + timesteps * hidden_size + hidden_size
    linear1 = 2 * 3 * hidden_size * hidden_size + hidden_size
    silu = 4 * hidden_size
    linear2 = 2 * hidden_size * hidden_size + hidden_size
    distiller = linear1 + silu + linear2
    selector = 4 * hidden_size + 1
    return {
        "cheap_summary_flops": summary,
        "distiller_flops": distiller,
        "selector_flops": selector,
        "total_cdd_inference_flops": summary + distiller + selector,
    }


def control_costs(*, hidden_size: int, variables: int, timesteps: int) -> dict[str, int]:
    summary = variables * hidden_size + timesteps * hidden_size + hidden_size
    selector = 6 * hidden_size + 1
    return {"cheap_summary_flops": summary, "selector_flops": selector, "total_control_inference_flops": summary + selector}


class CDD(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearSelector(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.readout = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(x).squeeze(-1)


def score_outcome_digest(scores: torch.Tensor, stop: torch.Tensor, branch: torch.Tensor) -> str:
    return canonical_sha256({"scores": scores.tolist(), "stop_exact": stop.to(torch.long).tolist(), "branch_exact": branch.to(torch.long).tolist()})
