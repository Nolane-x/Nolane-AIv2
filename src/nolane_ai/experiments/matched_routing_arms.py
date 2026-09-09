from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.optimizer import functional_trainable_named_parameters

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
        *,
        route_threshold: float,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-279 dimensions and target_parameters must be positive")
        if not 0.0 <= route_threshold <= 1.0:
            raise ValueError("route_threshold must be within [0, 1]")
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.target_parameters = target_parameters
        self.route_threshold = float(route_threshold)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.branch_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.propagation_projection = nn.Linear(hidden_size, hidden_size, device=device)
        self.reclaimed_projection = nn.Linear(hidden_size, hidden_size, device=device)
        self.routing_head = nn.Linear(hidden_size, 1, device=device)
        self.decision_head = nn.Linear(hidden_size, 2, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        self.mix_gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=False)

    def _validate_common(
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
        variables = F.silu(self.variable_projection(variable_states))
        return events, variables

    def _validate_incidence(self, incidence: torch.Tensor, variables: torch.Tensor) -> torch.Tensor:
        if incidence.ndim != 3:
            raise ValueError("incidence must be rank-3 [batch,constraints,variables]")
        if incidence.shape[0] != variables.shape[0] or incidence.shape[2] != variables.shape[1]:
            raise ValueError("incidence must have shape [batch,constraints,variables]")
        if incidence.shape[1] <= 0:
            raise ValueError("incidence must contain at least one constraint")
        return incidence.to(device=variables.device, dtype=variables.dtype)

    def _branch_context(self, events: torch.Tensor) -> torch.Tensor:
        _, hidden = self.branch_gru(events)
        return hidden[-1]

    def _propagate(self, variables: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        c_degree = incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)
        constraint_summary = torch.bmm(incidence, variables) / c_degree
        constraint_states = torch.tanh(self.propagation_projection(constraint_summary))
        transpose = incidence.transpose(1, 2)
        v_degree = transpose.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.bmm(transpose, constraint_states) / v_degree

    def _residual_uncertainty(self, hidden: torch.Tensor) -> torch.Tensor:
        routing_logits = self.routing_head(hidden).squeeze(-1)
        routing_probability = torch.sigmoid(routing_logits)
        return routing_probability.mean(dim=-1).clamp(0.0, 1.0)

    def _heads(
        self,
        hidden: torch.Tensor,
        *,
        residual_uncertainty: torch.Tensor,
        route_mask: torch.Tensor,
        semantics: str,
    ) -> Exp279ArmOutput:
        return Exp279ArmOutput(
            decision_logits=self.decision_head(hidden),
            verifier_confidence=torch.sigmoid(self.verifier_head(hidden).squeeze(-1)),
            residual_uncertainty=residual_uncertainty,
            branch_route_mask=route_mask,
            representation_semantics=semantics,
        )


class PropagationOnlyArm(_MatchedExp279ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Exp279ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        incidence = self._validate_incidence(incidence, variables)
        propagated = self._propagate(variables, incidence)

        # The simpler arm is not allowed to park branch-controller capacity in an
        # excluded reserve. Reclaim it as a one-step surface-summary transform;
        # this activates the GRU parameters without granting multi-step branch-search semantics.
        summary_event = events.mean(dim=1, keepdim=True)
        reclaimed_branch = self._branch_context(summary_event).unsqueeze(1)
        reclaimed_surface = torch.tanh(
            self.reclaimed_projection(reclaimed_branch.expand(-1, variables.shape[1], -1))
        )
        mixed = variables + propagated + torch.sigmoid(self.mix_gate) * reclaimed_surface
        residual = self._residual_uncertainty(mixed)
        route_mask = torch.zeros(mixed.shape[0], dtype=torch.bool, device=mixed.device)
        return self._heads(
            mixed,
            residual_uncertainty=residual,
            route_mask=route_mask,
            semantics="constraint_propagation_no_branch_search_with_one_step_reclaimed_branch_capacity",
        )


class BranchOnlyArm(_MatchedExp279ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
    ) -> Exp279ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        branch_context = self._branch_context(events).unsqueeze(1)
        # Reclaim the propagation envelope as an incidence-free variable-local transform.
        reclaimed = torch.tanh(self.propagation_projection(variables))
        reclaimed = torch.tanh(self.reclaimed_projection(reclaimed))
        mixed = variables + branch_context + torch.sigmoid(self.mix_gate) * reclaimed
        residual = self._residual_uncertainty(mixed)
        route_mask = torch.ones(mixed.shape[0], dtype=torch.bool, device=mixed.device)
        return self._heads(
            mixed,
            residual_uncertainty=residual,
            route_mask=route_mask,
            semantics="surface_recurrent_branch_no_compiled_incidence_with_reclaimed_propagation_capacity",
        )


class HybridRoutingArm(_MatchedExp279ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Exp279ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        incidence = self._validate_incidence(incidence, variables)
        propagated = self._propagate(variables, incidence)
        propagation_state = variables + propagated
        residual = self._residual_uncertainty(propagation_state)
        route_mask = residual > self.route_threshold

        reclaimed = torch.tanh(self.reclaimed_projection(propagation_state))
        stop_state = propagation_state + reclaimed
        mixed = stop_state

        # Crucially, branch recurrence executes only for routed episodes. This binds
        # real execution semantics to the stop-vs-branch accounted-cost receipt.
        if bool(route_mask.any().item()):
            routed_indices = route_mask.nonzero(as_tuple=False).squeeze(-1)
            routed_context = self._branch_context(events.index_select(0, routed_indices))
            routed_context = routed_context.unsqueeze(1).expand(-1, variables.shape[1], -1)
            routed_state = propagation_state.index_select(0, routed_indices)
            routed_reclaimed = reclaimed.index_select(0, routed_indices)
            branch_state = routed_state + torch.sigmoid(self.mix_gate) * (routed_context + routed_reclaimed)
            mixed = mixed.index_copy(0, routed_indices, branch_state)

        return self._heads(
            mixed,
            residual_uncertainty=residual,
            route_mask=route_mask,
            semantics="propagation_then_branch_only_for_routed_residual_uncertainty",
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
        route_threshold=route_threshold,
        device=device,
    )
    branch = BranchOnlyArm(
        d_model,
        hidden_size,
        target_parameters,
        route_threshold=route_threshold,
        device=device,
    )
    hybrid = HybridRoutingArm(
        d_model,
        hidden_size,
        target_parameters,
        route_threshold=route_threshold,
        device=device,
    )
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
    optimizer_visible = sum(parameter.numel() for _, parameter in functional_trainable_named_parameters(arm))
    return {
        "total_parameters": total,
        "functional_parameters": functional,
        "active_functional_parameters": functional,
        "optimizer_visible_parameters": optimizer_visible,
        "reserved_parameters": reserve,
    }


def _linear_flops(inputs: int, outputs: int) -> int:
    return 2 * inputs * outputs + outputs


def _shared_flops(arm: _MatchedExp279ArmBase, *, timesteps: int, variables: int) -> int:
    d = arm.d_model
    h = arm.hidden_size
    return (
        timesteps * _linear_flops(d, h)
        + variables * _linear_flops(d, h)
        + variables * (_linear_flops(h, 2) + _linear_flops(h, 1) + 1)
    )


def _gru_flops(timesteps: int, hidden_size: int) -> int:
    return timesteps * (12 * hidden_size * hidden_size + 20 * hidden_size)


def _compute_ledgers(
    arm: _MatchedExp279ArmBase,
    *,
    timesteps: int,
    variables: int,
    constraints: int,
) -> dict[str, dict[str, Any]]:
    if min(timesteps, variables, constraints) <= 0:
        raise ValueError("EXP-279 compute geometry must be positive")
    h = arm.hidden_size
    shared = _shared_flops(arm, timesteps=timesteps, variables=variables)
    propagation = (
        constraints * _linear_flops(h, h)
        + 4 * constraints * variables * h
        + (constraints + variables) * h
    )
    branch = _gru_flops(timesteps, h) + variables * h
    reclaimed_projection = variables * _linear_flops(h, h)
    reclaimed_branch_summary = _gru_flops(1, h) + h
    routing = variables * _linear_flops(h, 1)

    propagation_total = shared + propagation + reclaimed_branch_summary + reclaimed_projection + routing
    branch_total = shared + branch + 2 * reclaimed_projection + routing
    hybrid_stop = shared + propagation + reclaimed_projection + routing
    hybrid_branch = hybrid_stop + branch

    def ledger(arm_id: str, total: int, components: dict[str, int]) -> dict[str, Any]:
        return {
            "arm_id": arm_id,
            "accounting_semantics": "analytical scalar arithmetic FLOPs for declared neural geometry; not hardware-profiler FLOPs",
            "geometry": {
                "timesteps": timesteps,
                "variables": variables,
                "constraints": constraints,
                "d_model": arm.d_model,
                "hidden_size": h,
            },
            "components": components,
            "accounted_flops_per_episode": int(total),
            "max_accounted_flops_per_episode": int(total),
            "hardware_profiler_flops_claimed": False,
        }

    return {
        "propagation_only": ledger(
            "propagation_only",
            propagation_total,
            {
                "shared": shared,
                "propagation": propagation,
                "reclaimed_branch_summary": reclaimed_branch_summary,
                "reclaimed_projection": reclaimed_projection,
                "routing_statistic": routing,
            },
        ),
        "branch_only": ledger(
            "branch_only",
            branch_total,
            {
                "shared": shared,
                "branch": branch,
                "reclaimed_capacity": 2 * reclaimed_projection,
                "routing_statistic": routing,
            },
        ),
        "hybrid": ledger(
            "hybrid",
            hybrid_branch,
            {
                "shared": shared,
                "propagation": propagation,
                "branch_max": branch,
                "reclaimed_capacity": reclaimed_projection,
                "routing": routing,
            },
        )
        | {
            "stop_accounted_flops_per_episode": int(hybrid_stop),
            "branch_accounted_flops_per_episode": int(hybrid_branch),
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
    audits = {
        "propagation_only": _parameter_audit(propagation),
        "branch_only": _parameter_audit(branch),
        "hybrid": _parameter_audit(hybrid),
    }
    totals = {item["total_parameters"] for item in audits.values()}
    functional = {item["functional_parameters"] for item in audits.values()}
    active = {item["active_functional_parameters"] for item in audits.values()}
    optimizer_visible = {item["optimizer_visible_parameters"] for item in audits.values()}
    parameter_match = len(totals) == 1
    functional_match = len(functional) == 1
    active_match = len(active) == 1 and active == functional
    optimizer_match = len(optimizer_visible) == 1 and optimizer_visible == functional
    if not (parameter_match and functional_match and active_match and optimizer_match):
        raise ValueError("EXP-279 matched triplet parameter contract is not closed")

    ledgers = _compute_ledgers(
        propagation,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
    )
    required = max(int(item["max_accounted_flops_per_episode"]) for item in ledgers.values())
    declared = required if max_accounted_flops_per_episode is None else int(max_accounted_flops_per_episode)
    if declared <= 0 or required > declared:
        raise ValueError(f"EXP-279 compute budget exceeded: required {required}, declared {declared}")

    return {
        "schema": "NLM-EXP-279-MATCHED-ROUTING-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        **audits,
        "parameter_match": parameter_match,
        "functional_parameter_match": functional_match,
        "active_functional_parameter_match": active_match,
        "optimizer_visible_parameter_match": optimizer_match,
        "reclaimed_parameter_assignment_closed": True,
        "reclaimed_parameter_assignment": {
            "propagation_only": "multi-step branch search withheld; branch-controller envelope reclaimed as one-step surface-summary transform",
            "branch_only": "compiled incidence withheld; propagation envelope reclaimed as variable-local transform",
            "hybrid": "propagation and routed branch envelopes both used under predeclared residual routing",
        },
        "inactive_excluded_reclaimed_parameters": 0,
        "structure_fit_strata": list(STRATA),
        "route_threshold": float(hybrid.route_threshold),
        "declared_max_accounted_flops_per_episode": declared,
        "compute_budget_closed": True,
        "compute_ledger": ledgers,
        "remaining_blockers": [
            "matched triplet remains DEVELOPMENT-only until paired structure-fit evaluator lineage is closed",
            "confirmatory blocked analysis and post-freeze challenge execution remain unrun",
        ],
    }
