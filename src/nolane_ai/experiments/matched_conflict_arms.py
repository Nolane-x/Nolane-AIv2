from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.optimizer import functional_trainable_named_parameters


@dataclass(frozen=True, slots=True)
class Exp286ArmOutput:
    rollback_logits: torch.Tensor
    verifier_confidence: torch.Tensor
    conflict_conditioning_used: bool
    representation_semantics: str


class _MatchedExp286ArmBase(nn.Module):
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
            raise ValueError("EXP-286 dimensions and target_parameters must be positive")
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.deliberation_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.core_adapter = nn.Linear(1, hidden_size, device=device)
        self.rollback_head = nn.Linear(hidden_size, 1, device=device)
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
        if surface_events.shape[1] <= 0 or variable_states.shape[1] <= 0:
            raise ValueError("surface_events and variable_states must be non-empty")
        events = F.silu(self.event_projection(surface_events))
        variables = F.silu(self.variable_projection(variable_states))
        return events, variables

    def _shared_state(
        self,
        events: torch.Tensor,
        variables: torch.Tensor,
        core_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.deliberation_gru(events)
        recurrent = hidden[-1].unsqueeze(1).expand(-1, variables.shape[1], -1)
        core_state = F.silu(self.core_adapter(core_token))
        mixed = variables + recurrent + torch.sigmoid(self.mix_gate) * core_state
        return (
            self.rollback_head(mixed).squeeze(-1),
            torch.sigmoid(self.verifier_head(mixed).squeeze(-1)),
        )


class ChronologicalFailureArm(_MatchedExp286ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        *,
        contradiction_observed: bool,
    ) -> Exp286ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        del contradiction_observed  # No privileged information enters this arm.
        null_core = torch.zeros(
            variables.shape[0],
            variables.shape[1],
            1,
            device=variables.device,
            dtype=variables.dtype,
        )
        rollback_logits, verifier_confidence = self._shared_state(events, variables, null_core)
        return Exp286ArmOutput(
            rollback_logits=rollback_logits,
            verifier_confidence=verifier_confidence,
            conflict_conditioning_used=False,
            representation_semantics="chronological_rollback_with_canonical_null_conflict_token",
        )


class OracleConflictCoreArm(_MatchedExp286ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        *,
        conflict_core_mask: torch.Tensor,
        contradiction_observed: bool,
    ) -> Exp286ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        if not contradiction_observed:
            raise ValueError("conflict core cannot be delivered before the current contradiction")
        if conflict_core_mask.ndim != 2:
            raise ValueError("conflict_core_mask must be [batch,variables]")
        if (
            conflict_core_mask.shape[0] != variables.shape[0]
            or conflict_core_mask.shape[1] != variables.shape[1]
        ):
            raise ValueError("conflict_core_mask must match [batch,variables]")
        mask = conflict_core_mask.to(device=variables.device, dtype=variables.dtype)
        if not torch.isfinite(mask).all():
            raise ValueError("conflict_core_mask must be finite")
        if bool(((mask < 0.0) | (mask > 1.0)).any().item()):
            raise ValueError("conflict_core_mask membership must be within [0,1]")
        if bool((mask.sum(dim=1) <= 0.0).any().item()):
            raise ValueError("conflict_core_mask must contain a current conflict member per episode")

        rollback_logits, verifier_confidence = self._shared_state(
            events,
            variables,
            mask.unsqueeze(-1),
        )
        return Exp286ArmOutput(
            rollback_logits=rollback_logits,
            verifier_confidence=verifier_confidence,
            conflict_conditioning_used=True,
            representation_semantics="current_contradiction_ground_truth_conflict_core_conditioned_rollback",
        )


def build_matched_exp286_arm_pair(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> tuple[ChronologicalFailureArm, OracleConflictCoreArm]:
    chronological = ChronologicalFailureArm(
        d_model,
        hidden_size,
        target_parameters,
        device=device,
    )
    oracle = OracleConflictCoreArm(
        d_model,
        hidden_size,
        target_parameters,
        device=device,
    )
    oracle.load_state_dict(chronological.state_dict())
    return chronological, oracle


def _parameter_audit(arm: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in arm.parameters())
    reserve = sum(
        parameter.numel()
        for name, parameter in arm.named_parameters()
        if "capacity_reserve" in name
    )
    functional = total - reserve
    optimizer_visible = sum(
        parameter.numel() for _, parameter in functional_trainable_named_parameters(arm)
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


def _gru_flops(timesteps: int, hidden_size: int) -> int:
    return timesteps * (12 * hidden_size * hidden_size + 20 * hidden_size)


def _compute_ledger(
    arm: _MatchedExp286ArmBase,
    *,
    timesteps: int,
    variables: int,
    max_search_steps: int,
) -> dict[str, Any]:
    if min(timesteps, variables, max_search_steps) <= 0:
        raise ValueError("EXP-286 compute geometry must be positive")
    d = arm.d_model
    h = arm.hidden_size

    event_projection = timesteps * _linear_flops(d, h)
    variable_projection = variables * _linear_flops(d, h)
    recurrent_deliberation = _gru_flops(timesteps, h)
    core_adapter = variables * _linear_flops(1, h)
    rollback_head = variables * _linear_flops(h, 1)
    verifier_head = variables * _linear_flops(h, 1)
    fusion_and_gate = variables * (3 * h + 1)

    per_search_step = (
        event_projection
        + variable_projection
        + recurrent_deliberation
        + core_adapter
        + rollback_head
        + verifier_head
        + fusion_and_gate
    )
    maximum = max_search_steps * per_search_step
    return {
        "accounting_semantics": (
            "analytical scalar arithmetic FLOPs for the declared neural search geometry; "
            "oracle information is free information but every neural operation consuming it is charged"
        ),
        "geometry": {
            "timesteps": int(timesteps),
            "variables": int(variables),
            "max_search_steps": int(max_search_steps),
            "d_model": int(d),
            "hidden_size": int(h),
        },
        "components_per_search_step": {
            "event_projection": int(event_projection),
            "variable_projection": int(variable_projection),
            "recurrent_deliberation": int(recurrent_deliberation),
            "conflict_core_or_null_adapter": int(core_adapter),
            "rollback_head": int(rollback_head),
            "verifier_head": int(verifier_head),
            "fusion_and_gate": int(fusion_and_gate),
        },
        "accounted_flops_per_search_step": int(per_search_step),
        "max_accounted_flops_per_episode": int(maximum),
        "hardware_profiler_flops_claimed": False,
    }


def audit_matched_exp286_arm_pair(
    chronological: ChronologicalFailureArm,
    oracle: OracleConflictCoreArm,
    *,
    timesteps: int,
    variables: int,
    max_search_steps: int,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    chronological_audit = _parameter_audit(chronological)
    oracle_audit = _parameter_audit(oracle)

    parameter_match = (
        chronological_audit["total_parameters"] == oracle_audit["total_parameters"]
    )
    functional_match = (
        chronological_audit["functional_parameters"] == oracle_audit["functional_parameters"]
    )
    active_match = (
        chronological_audit["active_functional_parameters"]
        == oracle_audit["active_functional_parameters"]
        == chronological_audit["functional_parameters"]
    )
    optimizer_match = (
        chronological_audit["optimizer_visible_parameters"]
        == oracle_audit["optimizer_visible_parameters"]
        == chronological_audit["functional_parameters"]
    )
    reserve_match = (
        chronological_audit["reserved_parameters"] == oracle_audit["reserved_parameters"]
    )
    if not all((parameter_match, functional_match, active_match, optimizer_match, reserve_match)):
        raise ValueError("EXP-286 matched pair parameter contract is not closed")

    chronological_ledger = _compute_ledger(
        chronological,
        timesteps=timesteps,
        variables=variables,
        max_search_steps=max_search_steps,
    )
    oracle_ledger = _compute_ledger(
        oracle,
        timesteps=timesteps,
        variables=variables,
        max_search_steps=max_search_steps,
    )
    required = max(
        int(chronological_ledger["max_accounted_flops_per_episode"]),
        int(oracle_ledger["max_accounted_flops_per_episode"]),
    )
    declared = required if max_accounted_flops_per_episode is None else int(
        max_accounted_flops_per_episode
    )
    if declared <= 0 or required > declared:
        raise ValueError(f"EXP-286 compute budget exceeded: required {required}, declared {declared}")

    return {
        "schema": "NLM-EXP-286-MATCHED-CONFLICT-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "chronological_failure": chronological_audit,
        "oracle_conflict_core": oracle_audit,
        "parameter_match": parameter_match,
        "functional_parameter_match": functional_match,
        "active_functional_parameter_match": active_match,
        "optimizer_visible_parameter_match": optimizer_match,
        "reserve_parameter_match": reserve_match,
        "oracle_information_separation": True,
        "oracle_information_receipt": {
            "artifact": "ground_truth_conflict_core",
            "ground_truth": True,
            "delivery_event": "after_current_contradiction_only",
            "delivered_to": ["oracle_conflict_core"],
            "withheld_from": ["chronological_failure"],
            "chronological_failure_received_conflict_core": False,
            "future_conflict_core_leakage": False,
            "solution_leakage": False,
        },
        "compute_budget_closed": True,
        "compute_ledger": {
            "chronological_failure": chronological_ledger,
            "oracle_conflict_core": oracle_ledger,
        },
        "declared_max_accounted_flops_per_episode": int(declared),
    }
