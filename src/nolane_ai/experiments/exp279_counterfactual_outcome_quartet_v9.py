from __future__ import annotations

from enum import IntEnum
from typing import Any

import torch
from torch import nn

from .matched_routing_arms import HybridRoutingArm

SCHEMA_SHARD = "NLM-EXP-279-COUNTERFACTUAL-OUTCOME-QUARTET-SHARD-V9"
SCHEMA_BUDGET = "NLM-EXP-279-COUNTERFACTUAL-OUTCOME-QUARTET-BUDGET-V9"
ROOT_PREFIX = "20260912-exp279-counterfactual-outcome-quartet-v9-dev"
PRIMARY_FAMILY = "OUTCOME_QUARTET_MLP"
CONTROL_FAMILY = "RESCUE_ONLY_MLP_CONTROL"
TRAIN_BUDGETS = (60, 120)
CANONICAL_INDICES = (0, 1, 2, 3)
FIT_INDICES = (0, 1)
DECISION_INDICES = (0, 1)
DIAGNOSTIC_FOLDS = 3
FIT_REPLICATES = 1332
DECISION_REPLICATES = 1332
FROZEN_PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
STUDENT_GEOMETRY = {"input_size": 144, "hidden_size": 64, "classes": 4}
STUDENT_OPTIMIZER = {
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 512,
    "steps": 200,
}


class QuartetClass(IntEnum):
    RESCUE = 0
    HARM = 1
    BOTH_SUCCESS = 2
    BOTH_FAILURE = 3


def _require_budget(train_replicates: int) -> int:
    value = int(train_replicates)
    if value not in TRAIN_BUDGETS:
        raise ValueError(f"V9 train budget must be one of {TRAIN_BUDGETS}, got {train_replicates!r}")
    return value


def _require_index(value: int, allowed: tuple[int, ...], *, label: str) -> int:
    index = int(value)
    if index not in allowed:
        raise ValueError(f"V9 {label} must be one of {allowed}, got {value!r}")
    return index


def canonical_root(train_replicates: int, canonical_index: int) -> str:
    budget = _require_budget(train_replicates)
    canonical = _require_index(canonical_index, CANONICAL_INDICES, label="canonical index")
    return f"{ROOT_PREFIX}::train::{budget}::canonical::{canonical}"


def fit_root(train_replicates: int, canonical_index: int, fit_index: int) -> str:
    budget = _require_budget(train_replicates)
    canonical = _require_index(canonical_index, CANONICAL_INDICES, label="canonical index")
    fit = _require_index(fit_index, FIT_INDICES, label="fit index")
    return f"{ROOT_PREFIX}::train::{budget}::fit::{canonical}::{fit}"


def decision_root(train_replicates: int, canonical_index: int, decision_index: int) -> str:
    budget = _require_budget(train_replicates)
    canonical = _require_index(canonical_index, CANONICAL_INDICES, label="canonical index")
    decision = _require_index(decision_index, DECISION_INDICES, label="decision index")
    return f"{ROOT_PREFIX}::train::{budget}::decision::{canonical}::{decision}"


def expected_root_map(train_replicates: int) -> dict[str, list[str]]:
    budget = _require_budget(train_replicates)
    roots = {
        "canonical_roots": [canonical_root(budget, c) for c in CANONICAL_INDICES],
        "fit_roots": [fit_root(budget, c, i) for c in CANONICAL_INDICES for i in FIT_INDICES],
        "decision_roots": [decision_root(budget, c, i) for c in CANONICAL_INDICES for i in DECISION_INDICES],
    }
    flat = roots["canonical_roots"] + roots["fit_roots"] + roots["decision_roots"]
    if len(flat) != len(set(flat)):
        raise RuntimeError("V9 frozen root map is not unique")
    return roots


def diagnostic_fold(replicate: int) -> int:
    value = int(replicate)
    if value < 0:
        raise ValueError("V9 diagnostic replicate must be non-negative")
    return value % DIAGNOSTIC_FOLDS


def quartet_labels(stop_exact: torch.Tensor, branch_exact: torch.Tensor) -> torch.Tensor:
    if stop_exact.ndim != 1 or branch_exact.ndim != 1 or stop_exact.shape != branch_exact.shape:
        raise ValueError("V9 stop/branch exact tensors must be aligned rank-1 tensors")
    stop = stop_exact.to(torch.bool)
    branch = branch_exact.to(torch.bool)
    labels = torch.full(stop.shape, int(QuartetClass.BOTH_FAILURE), dtype=torch.long, device=stop.device)
    labels[(~stop) & branch] = int(QuartetClass.RESCUE)
    labels[stop & (~branch)] = int(QuartetClass.HARM)
    labels[stop & branch] = int(QuartetClass.BOTH_SUCCESS)
    return labels


def cheap_prebranch_features(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    events, variables = arm._validate_common(surface_events, variable_states)
    checked = arm._validate_incidence(incidence, variables)
    propagation_state = variables + arm._propagate(variables, checked)
    p_mean = propagation_state.mean(dim=1)
    e_mean = events.mean(dim=1)
    e_delta = events[:, -1] - events[:, 0]
    return torch.cat((p_mean, e_mean, e_delta), dim=-1)


class OutcomeQuartetMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STUDENT_GEOMETRY["input_size"], STUDENT_GEOMETRY["hidden_size"]),
            nn.SiLU(),
            nn.Linear(STUDENT_GEOMETRY["hidden_size"], STUDENT_GEOMETRY["classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RescueOnlyMLPControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STUDENT_GEOMETRY["input_size"], STUDENT_GEOMETRY["hidden_size"]),
            nn.SiLU(),
            nn.Linear(STUDENT_GEOMETRY["hidden_size"], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def student_inference_flops(*, hidden_size: int, variables: int, timesteps: int) -> int:
    h = int(hidden_size)
    v = int(variables)
    t = int(timesteps)
    if min(h, v, t) <= 0:
        raise ValueError("V9 student FLOP geometry must be positive")
    input_size = 3 * h
    student_hidden = STUDENT_GEOMETRY["hidden_size"]
    classes = STUDENT_GEOMETRY["classes"]
    summary = v * h + t * h + h
    linear1 = 2 * input_size * student_hidden + student_hidden
    silu = 4 * student_hidden
    linear2 = 2 * student_hidden * classes + classes
    return int(summary + linear1 + silu + linear2)


def marginal_route_score(
    quartet_probabilities: torch.Tensor,
    *,
    fit_stop_utility: float,
    stop_accounted_flops: int | float,
    branch_accounted_flops: int | float,
) -> torch.Tensor:
    if quartet_probabilities.ndim != 2 or quartet_probabilities.shape[1] != 4:
        raise ValueError("V9 quartet probabilities must be [episodes,4]")
    stop_cost = float(stop_accounted_flops)
    branch_cost = float(branch_accounted_flops)
    if fit_stop_utility < 0.0 or stop_cost <= 0.0 or branch_cost <= stop_cost:
        raise ValueError("V9 route economics are invalid")
    p_rescue = quartet_probabilities[:, int(QuartetClass.RESCUE)]
    p_harm = quartet_probabilities[:, int(QuartetClass.HARM)]
    return (p_rescue - p_harm) - float(fit_stop_utility) * (branch_cost - stop_cost)


def apply_quartet_policy(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("V9 route scores must be rank-1")
    return scores > 0


def direct_utility(
    chosen_branch: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    *,
    stop_accounted_flops: int | float,
    branch_accounted_flops: int | float,
    student_accounted_flops: int | float,
) -> dict[str, Any]:
    if chosen_branch.ndim != 1 or stop_exact.ndim != 1 or branch_exact.ndim != 1:
        raise ValueError("V9 direct utility inputs must be rank-1")
    if chosen_branch.shape != stop_exact.shape or stop_exact.shape != branch_exact.shape:
        raise ValueError("V9 direct utility inputs must align")
    stop_cost = float(stop_accounted_flops)
    branch_cost = float(branch_accounted_flops)
    student_cost = float(student_accounted_flops)
    if stop_cost <= 0.0 or branch_cost <= 0.0 or student_cost < 0.0:
        raise ValueError("V9 direct utility costs are invalid")
    route = chosen_branch.to(torch.bool)
    stop = stop_exact.to(torch.bool)
    branch = branch_exact.to(torch.bool)
    realized = torch.where(route, branch, stop)
    episodes = int(route.numel())
    solutions = int(realized.sum().item())
    routed = int(route.sum().item())
    total_flops = (episodes - routed) * (stop_cost + student_cost) + routed * (branch_cost + student_cost)
    return {
        "episodes": episodes,
        "solutions": solutions,
        "routed_episodes": routed,
        "route_fraction": routed / episodes if episodes else 0.0,
        "total_accounted_flops": int(total_flops) if float(total_flops).is_integer() else total_flops,
        "utility": solutions / total_flops if total_flops > 0.0 else 0.0,
    }
