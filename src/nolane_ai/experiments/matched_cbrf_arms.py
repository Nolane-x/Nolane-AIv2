from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget


@dataclass(frozen=True, slots=True)
class Exp277ArmOutput:
    decision_logits: torch.Tensor
    verifier_confidence: torch.Tensor
    representation_semantics: str


class _MatchedExp277ArmBase(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, target_parameters: int, *, device=None) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-277 arm dimensions and target_parameters must be positive")
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.target_parameters = target_parameters
        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.event_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.structure_projection = nn.Linear(hidden_size, hidden_size, device=device)
        self.decision_head = nn.Linear(hidden_size, 2, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        self.gate = nn.Parameter(torch.zeros((), device=device))
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
        _, hidden = self.event_gru(events)
        context = hidden[-1]
        variables = F.silu(self.variable_projection(variable_states))
        return context, variables

    def _heads(self, hidden: torch.Tensor, *, semantics: str) -> Exp277ArmOutput:
        return Exp277ArmOutput(
            decision_logits=self.decision_head(hidden),
            verifier_confidence=torch.sigmoid(self.verifier_head(hidden).squeeze(-1)),
            representation_semantics=semantics,
        )


class ARCSBranchArm(_MatchedExp277ArmBase):
    def forward(self, surface_events: torch.Tensor, variable_states: torch.Tensor) -> Exp277ArmOutput:
        context, variables = self._encode_common(surface_events, variable_states)
        branch_state = torch.tanh(self.structure_projection(variables + context.unsqueeze(1)))
        mixed = variables + torch.sigmoid(self.gate) * branch_state
        return self._heads(mixed, semantics="surface_recurrent_branch_no_oracle_incidence")


class OracleCBRFArm(_MatchedExp277ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Exp277ArmOutput:
        context, variables = self._encode_common(surface_events, variable_states)
        if incidence.ndim != 3:
            raise ValueError("incidence must be rank-3 [batch,constraints,variables]")
        if incidence.shape[0] != variables.shape[0] or incidence.shape[2] != variables.shape[1]:
            raise ValueError("incidence must have shape [batch,constraints,variables]")
        incidence = incidence.to(device=variables.device, dtype=variables.dtype)
        c_degree = incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)
        constraint_summary = torch.bmm(incidence, variables) / c_degree
        constraint_states = torch.tanh(self.structure_projection(constraint_summary))
        transpose = incidence.transpose(1, 2)
        v_degree = transpose.sum(dim=-1, keepdim=True).clamp_min(1.0)
        structural_message = torch.bmm(transpose, constraint_states) / v_degree
        mixed = variables + context.unsqueeze(1) + torch.sigmoid(self.gate) * structural_message
        return self._heads(mixed, semantics="ground_truth_factor_incidence_message_passing")


def build_matched_exp277_arm_pair(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> tuple[ARCSBranchArm, OracleCBRFArm]:
    arcs = ARCSBranchArm(d_model, hidden_size, target_parameters, device=device)
    oracle = OracleCBRFArm(d_model, hidden_size, target_parameters, device=device)
    oracle.load_state_dict(arcs.state_dict())
    return arcs, oracle


def _arm_parameter_audit(arm: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in arm.parameters())
    reserve = sum(
        parameter.numel()
        for name, parameter in arm.named_parameters()
        if "capacity_reserve" in name
    )
    return {
        "total_parameters": total,
        "functional_parameters": total - reserve,
        "reserved_parameters": reserve,
    }


def _linear_flops(inputs: int, outputs: int) -> int:
    return 2 * inputs * outputs + outputs


def _compute_ledger(
    arm: _MatchedExp277ArmBase,
    *,
    arm_id: str,
    timesteps: int,
    variables: int,
    constraints: int,
) -> dict[str, Any]:
    if min(timesteps, variables, constraints) <= 0:
        raise ValueError("EXP-277 compute geometry must be positive")
    d = arm.d_model
    h = arm.hidden_size
    event_projection = timesteps * _linear_flops(d, h)
    gru = timesteps * (12 * h * h + 20 * h)
    variable_projection = variables * _linear_flops(d, h)
    heads = variables * (_linear_flops(h, 2) + _linear_flops(h, 1) + 1)
    if arm_id == "arcs_branch":
        structure = variables * _linear_flops(h, h)
        routing = variables * (4 * h)
    elif arm_id == "oracle_cbrf":
        structure = constraints * _linear_flops(h, h)
        routing = 4 * constraints * variables * h + (constraints + variables) * h
    else:
        raise ValueError(f"unknown EXP-277 arm id: {arm_id}")
    total = event_projection + gru + variable_projection + structure + routing + heads
    return {
        "arm_id": arm_id,
        "accounting_semantics": "analytical scalar arithmetic FLOPs for declared neural geometry; not hardware-profiler FLOPs",
        "geometry": {
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d,
            "hidden_size": h,
        },
        "components": {
            "event_projection": event_projection,
            "recurrent_event_encoding": gru,
            "variable_projection": variable_projection,
            "structure_or_branch_processing": structure,
            "routing_or_incidence_message_passing": routing,
            "decision_and_verifier_heads": heads,
        },
        "accounted_flops_per_episode": total,
        "hardware_profiler_flops_claimed": False,
    }


def audit_matched_exp277_arm_pair(
    arcs: ARCSBranchArm,
    oracle: OracleCBRFArm,
    *,
    timesteps: int,
    variables: int,
    constraints: int,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    arcs_audit = _arm_parameter_audit(arcs)
    oracle_audit = _arm_parameter_audit(oracle)
    parameter_match = arcs_audit["total_parameters"] == oracle_audit["total_parameters"]
    functional_match = arcs_audit["functional_parameters"] == oracle_audit["functional_parameters"]
    arcs_compute = _compute_ledger(
        arcs,
        arm_id="arcs_branch",
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
    )
    oracle_compute = _compute_ledger(
        oracle,
        arm_id="oracle_cbrf",
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
    )
    required_budget = max(
        int(arcs_compute["accounted_flops_per_episode"]),
        int(oracle_compute["accounted_flops_per_episode"]),
    )
    declared_budget = required_budget if max_accounted_flops_per_episode is None else int(max_accounted_flops_per_episode)
    if declared_budget <= 0 or required_budget > declared_budget:
        raise ValueError(
            f"EXP-277 compute budget exceeded: required {required_budget}, declared {declared_budget}"
        )
    if not parameter_match or not functional_match:
        raise ValueError("EXP-277 matched arm parameter contract is not closed")
    return {
        "schema": "NLM-EXP-277-MATCHED-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "arcs_branch": arcs_audit,
        "oracle_cbrf": oracle_audit,
        "parameter_match": parameter_match,
        "functional_parameter_match": functional_match,
        "oracle_information_separation": True,
        "declared_max_accounted_flops_per_episode": declared_budget,
        "compute_budget_closed": True,
        "compute_ledger": {
            "schema": "NLM-EXP-277-COMPUTE-LEDGER-V1",
            "evidence_level": "EV-E2",
            "decision": "UNVERIFIED",
            "arcs_branch": arcs_compute,
            "oracle_cbrf": oracle_compute,
        },
        "remaining_blockers": [
            "matched arms remain DEVELOPMENT-only until paired world/evaluation lineage is closed",
            "confirmatory-open and post-freeze challenge execution remain unrun",
        ],
    }
