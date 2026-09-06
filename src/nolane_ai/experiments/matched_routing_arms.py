from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget


STRATA = ("PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL")


@dataclass(frozen=True, slots=True)
class Exp279ArmOutput:
    decision_logits: torch.Tensor
    verifier_confidence: torch.Tensor
    residual_uncertainty: torch.Tensor
    branch_route_mask: torch.Tensor
    representation_semantics: str


class _MatchedExp279ArmBase(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        target_parameters: int,
        route_threshold: float,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-279 arm dimensions and target_parameters must be positive")
        if not 0.0 <= route_threshold <= 1.0:
            raise ValueError("route_threshold must be in [0, 1]")
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)
        self.route_threshold = float(route_threshold)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.event_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.structure_projection = nn.Linear(hidden_size, hidden_size, device=device)
        self.recurrent_core = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.route_head = nn.Linear(hidden_size, 1, device=device)
        self.decision_head = nn.Linear(hidden_size, 2, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        finalize_region_budget(self, target_parameters, device=device, frozen=False)

    def _encode_common(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if surface_events.ndim != 3 or surface_events.shape[-1] != self.d_model:
            raise ValueError("surface_events must be [batch,time,d_model]")
        if variable_states.ndim != 3 or variable_states.shape[-1] != self.d_model:
            raise ValueError("variable_states must be [batch,variables,d_model]")
        if surface_events.shape[0] != variable_states.shape[0]:
            raise ValueError("surface_events and variable_states batch dimensions must match")
        events = F.silu(self.event_projection(surface_events))
        _, event_hidden = self.event_gru(events)
        context = event_hidden[-1]
        variables = F.silu(self.variable_projection(variable_states))
        return context, variables

    @staticmethod
    def _validate_incidence(incidence: torch.Tensor, variables: torch.Tensor) -> torch.Tensor:
        if incidence.ndim != 3:
            raise ValueError("incidence must be rank-3 [batch,constraints,variables]")
        if incidence.shape[0] != variables.shape[0] or incidence.shape[2] != variables.shape[1]:
            raise ValueError("incidence must have shape [batch,constraints,variables]")
        if incidence.shape[1] <= 0:
            raise ValueError("incidence must contain at least one constraint")
        return incidence.to(device=variables.device, dtype=variables.dtype)

    def _propagate(self, variables: torch.Tensor, incidence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        incidence = self._validate_incidence(incidence, variables)
        constraint_degree = incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)
        constraint_summary = torch.bmm(incidence, variables) / constraint_degree
        constraint_states = torch.tanh(self.structure_projection(constraint_summary + context.unsqueeze(1)))
        transpose = incidence.transpose(1, 2)
        variable_degree = transpose.sum(dim=-1, keepdim=True).clamp_min(1.0)
        message = torch.bmm(transpose, constraint_states) / variable_degree
        return variables + message

    def _heads(
        self,
        hidden: torch.Tensor,
        *,
        residual_uncertainty: torch.Tensor,
        branch_route_mask: torch.Tensor,
        semantics: str,
    ) -> Exp279ArmOutput:
        return Exp279ArmOutput(
            decision_logits=self.decision_head(hidden),
            verifier_confidence=torch.sigmoid(self.verifier_head(hidden).squeeze(-1)),
            residual_uncertainty=residual_uncertainty,
            branch_route_mask=branch_route_mask,
            representation_semantics=semantics,
        )


class PropagationOnlyArm(_MatchedExp279ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Exp279ArmOutput:
        context, variables = self._encode_common(surface_events, variable_states)
        propagated = self._propagate(variables, incidence, context)
        # Reclaimed recurrent capacity is active as iterative propagation refinement,
        # not as branch search.
        refined, _ = self.recurrent_core(propagated)
        confidence_gate = torch.sigmoid(self.route_head(refined))
        hidden = propagated + confidence_gate * refined
        residual = 1.0 - confidence_gate.squeeze(-1).mean(dim=1)
        route_mask = torch.zeros(propagated.shape[0], dtype=torch.bool, device=propagated.device)
        return self._heads(
            hidden,
            residual_uncertainty=residual,
            branch_route_mask=route_mask,
            semantics="incidence_propagation_with_reclaimed_iterative_refinement_no_branch_search",
        )


class BranchOnlyArm(_MatchedExp279ArmBase):
    def forward(self, surface_events: torch.Tensor, variable_states: torch.Tensor) -> Exp279ArmOutput:
        context, variables = self._encode_common(surface_events, variable_states)
        # Reclaimed propagation transform is active as a branch preconditioner.
        preconditioned = torch.tanh(self.structure_projection(variables + context.unsqueeze(1)))
        refined, _ = self.recurrent_core(preconditioned)
        confidence_gate = torch.sigmoid(self.route_head(refined))
        hidden = variables + confidence_gate * (preconditioned + refined)
        residual = 1.0 - confidence_gate.squeeze(-1).mean(dim=1)
        route_mask = torch.ones(variables.shape[0], dtype=torch.bool, device=variables.device)
        return self._heads(
            hidden,
            residual_uncertainty=residual,
            branch_route_mask=route_mask,
            semantics="surface_recurrent_branch_with_reclaimed_structure_preconditioner_no_incidence",
        )


class HybridRoutingArm(_MatchedExp279ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Exp279ArmOutput:
        context, variables = self._encode_common(surface_events, variable_states)
        propagated = self._propagate(variables, incidence, context)
        per_variable_uncertainty = torch.sigmoid(self.route_head(propagated)).squeeze(-1)
        residual = per_variable_uncertainty.mean(dim=1)
        route_mask = residual > self.route_threshold
        hidden = propagated.clone()
        if bool(route_mask.any().item()):
            routed, _ = self.recurrent_core(propagated[route_mask])
            hidden[route_mask] = propagated[route_mask] + routed
        return self._heads(
            hidden,
            residual_uncertainty=residual,
            branch_route_mask=route_mask,
            semantics="incidence_propagation_then_conditional_recurrent_branch_on_residual_uncertainty",
        )


def build_matched_exp279_arm_triplet(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float = 0.5,
    device: str | torch.device | None = None,
) -> tuple[PropagationOnlyArm, BranchOnlyArm, HybridRoutingArm]:
    propagation = PropagationOnlyArm(
        d_model,
        hidden_size,
        target_parameters,
        route_threshold,
        device=device,
    )
    branch = BranchOnlyArm(d_model, hidden_size, target_parameters, route_threshold, device=device)
    hybrid = HybridRoutingArm(d_model, hidden_size, target_parameters, route_threshold, device=device)
    branch.load_state_dict(propagation.state_dict())
    hybrid.load_state_dict(propagation.state_dict())
    return propagation, branch, hybrid


def _parameter_audit(arm: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in arm.parameters())
    reserve = sum(
        parameter.numel()
        for name, parameter in arm.named_parameters()
        if "capacity_reserve" in name
    )
    functional = total - reserve
    return {
        "total_parameters": total,
        "functional_parameters": functional,
        "active_functional_parameters": functional,
        "reserved_parameters": reserve,
    }


def _linear_flops(inputs: int, outputs: int) -> int:
    return 2 * inputs * outputs + outputs


def _gru_step_flops(width: int) -> int:
    # Conservative analytical scalar arithmetic count for three GRU gates.
    return 12 * width * width + 18 * width


def _common_flops(arm: _MatchedExp279ArmBase, *, timesteps: int, variables: int) -> int:
    d = arm.d_model
    h = arm.hidden_size
    return (
        timesteps * _linear_flops(d, h)
        + timesteps * _gru_step_flops(h)
        + variables * _linear_flops(d, h)
    )


def _head_flops(arm: _MatchedExp279ArmBase, *, variables: int) -> int:
    h = arm.hidden_size
    return variables * (
        _linear_flops(h, 2)
        + _linear_flops(h, 1)
        + 2
    )


def _propagation_flops(arm: _MatchedExp279ArmBase, *, variables: int, constraints: int) -> int:
    h = arm.hidden_size
    incidence_routing = 4 * constraints * variables * h
    normalization = (constraints + variables) * h
    structure = constraints * _linear_flops(h, h)
    return incidence_routing + normalization + structure


def _recurrent_flops(arm: _MatchedExp279ArmBase, *, variables: int) -> int:
    return variables * _gru_step_flops(arm.hidden_size)


def _route_flops(arm: _MatchedExp279ArmBase, *, variables: int) -> int:
    return variables * (_linear_flops(arm.hidden_size, 1) + 2)


def _compute_ledger(
    arm: _MatchedExp279ArmBase,
    *,
    arm_id: str,
    timesteps: int,
    variables: int,
    constraints: int,
) -> dict[str, Any]:
    if min(timesteps, variables, constraints) <= 0:
        raise ValueError("EXP-279 compute geometry must be positive")
    common = _common_flops(arm, timesteps=timesteps, variables=variables)
    heads = _head_flops(arm, variables=variables)
    route = _route_flops(arm, variables=variables)
    propagation = _propagation_flops(arm, variables=variables, constraints=constraints)
    recurrent = _recurrent_flops(arm, variables=variables)
    structure_preconditioner = variables * _linear_flops(arm.hidden_size, arm.hidden_size)

    if arm_id == "propagation_only":
        components = {
            "common_encoding": common,
            "constraint_propagation": propagation,
            "reclaimed_iterative_propagation_refinement": recurrent,
            "propagation_confidence_gate": route,
            "decision_and_verifier_heads": heads,
        }
        base_without_branch = sum(components.values())
        branch_increment = 0
    elif arm_id == "branch_only":
        components = {
            "common_encoding": common,
            "reclaimed_structure_branch_preconditioner": structure_preconditioner,
            "recurrent_branch_refinement": recurrent,
            "branch_confidence_gate": route,
            "decision_and_verifier_heads": heads,
        }
        base_without_branch = sum(components.values())
        branch_increment = 0
    elif arm_id == "hybrid":
        base_components = {
            "common_encoding": common,
            "constraint_propagation": propagation,
            "residual_uncertainty_route": route,
            "decision_and_verifier_heads": heads,
        }
        base_without_branch = sum(base_components.values())
        branch_increment = recurrent
        components = dict(base_components)
        components["conditional_recurrent_branch_max"] = recurrent
    else:
        raise ValueError(f"unknown EXP-279 arm id: {arm_id}")

    maximum = base_without_branch + branch_increment
    return {
        "arm_id": arm_id,
        "accounting_semantics": "analytical scalar arithmetic FLOPs for declared EXP-279 neural geometry; not hardware-profiler FLOPs",
        "geometry": {
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": arm.d_model,
            "hidden_size": arm.hidden_size,
        },
        "components": components,
        "base_without_branch_flops": base_without_branch,
        "branch_increment_flops": branch_increment,
        "accounted_flops_per_episode": maximum,
        "max_accounted_flops_per_episode": maximum,
        "hardware_profiler_flops_claimed": False,
    }


def _reclaimed_assignments() -> dict[str, dict[str, str]]:
    return {
        "propagation_only": {
            "event_projection+event_gru": "surface evidence encoding",
            "variable_projection": "variable state encoding",
            "structure_projection": "constraint propagation transform",
            "recurrent_core": "reclaimed iterative propagation refinement; not branch search",
            "route_head": "reclaimed propagation-confidence gate",
            "decision_head+verifier_head": "decision and external-verification support heads",
        },
        "branch_only": {
            "event_projection+event_gru": "surface evidence encoding",
            "variable_projection": "variable state encoding",
            "structure_projection": "reclaimed branch preconditioner; no incidence input",
            "recurrent_core": "recurrent branch refinement",
            "route_head": "reclaimed branch-confidence gate",
            "decision_head+verifier_head": "decision and external-verification support heads",
        },
        "hybrid": {
            "event_projection+event_gru": "surface evidence encoding",
            "variable_projection": "variable state encoding",
            "structure_projection": "constraint propagation transform",
            "recurrent_core": "conditional recurrent branch refinement",
            "route_head": "residual-uncertainty branch router",
            "decision_head+verifier_head": "decision and external-verification support heads",
        },
    }


def audit_matched_exp279_arm_triplet(
    propagation: PropagationOnlyArm,
    branch: BranchOnlyArm,
    hybrid: HybridRoutingArm,
    *,
    timesteps: int,
    variables: int,
    constraints: int,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    arms: dict[str, _MatchedExp279ArmBase] = {
        "propagation_only": propagation,
        "branch_only": branch,
        "hybrid": hybrid,
    }
    audits = {arm_id: _parameter_audit(arm) for arm_id, arm in arms.items()}
    totals = {audit["total_parameters"] for audit in audits.values()}
    functionals = {audit["functional_parameters"] for audit in audits.values()}
    active = {audit["active_functional_parameters"] for audit in audits.values()}
    parameter_match = len(totals) == 1
    functional_match = len(functionals) == 1
    active_match = len(active) == 1 and active == functionals

    ledgers = {
        arm_id: _compute_ledger(
            arm,
            arm_id=arm_id,
            timesteps=timesteps,
            variables=variables,
            constraints=constraints,
        )
        for arm_id, arm in arms.items()
    }
    required = max(int(ledger["max_accounted_flops_per_episode"]) for ledger in ledgers.values())
    declared = required if max_accounted_flops_per_episode is None else int(max_accounted_flops_per_episode)
    if declared <= 0 or required > declared:
        raise ValueError(f"EXP-279 compute budget exceeded: required {required}, declared {declared}")
    if not parameter_match or not functional_match or not active_match:
        raise ValueError("EXP-279 matched triplet parameter contract is not closed")

    assignments = _reclaimed_assignments()
    if any(not assignment for assignment in assignments.values()):
        raise ValueError("EXP-279 reclaimed parameter assignment is incomplete")

    return {
        "schema": "NLM-EXP-279-MATCHED-ROUTING-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        **audits,
        "parameter_match": True,
        "functional_parameter_match": True,
        "active_functional_parameter_match": True,
        "reclaimed_parameter_assignment_closed": True,
        "reclaimed_parameter_assignment": assignments,
        "structure_fit_strata": list(STRATA),
        "declared_max_accounted_flops_per_episode": declared,
        "compute_budget_closed": True,
        "compute_ledger": {
            "schema": "NLM-EXP-279-COMPUTE-LEDGER-V1",
            "evidence_level": "EV-E2",
            "decision": "UNVERIFIED",
            **ledgers,
        },
        "remaining_blockers": [
            "paired structure-fit evaluator lineage remains DEVELOPMENT-only until execution evidence is attached",
            "confirmatory-open and post-freeze challenge execution remain unrun",
        ],
    }
