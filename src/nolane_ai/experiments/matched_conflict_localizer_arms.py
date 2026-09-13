from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.optimizer import functional_trainable_named_parameters


NULL_MODE = "NULL_CORE_CONTROL"
LEARNED_MODE = "LEARNED_CORE_TOP2"
ORACLE_MODE = "ORACLE_CORE_UPPER_BOUND"
MODES = (NULL_MODE, LEARNED_MODE, ORACLE_MODE)


@dataclass(frozen=True, slots=True)
class Exp287ModeOutput:
    rollback_logits: torch.Tensor
    verifier_confidence: torch.Tensor
    localizer_logits: torch.Tensor
    conflict_token: torch.Tensor
    mode: str
    contradiction_observed: bool
    oracle_information_delivered: bool


def learned_top2_mask(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("localizer logits must be [batch,variables] with at least two variables")
    if not torch.isfinite(logits).all():
        raise ValueError("localizer logits must be finite")
    # Stable ordering means equal logits retain ascending variable-index order.
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    selected = order[:, :2]
    mask = torch.zeros_like(logits)
    return mask.scatter(1, selected, 1.0)


class Exp287ConflictLocalizer(nn.Module):
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
            raise ValueError("EXP-287 dimensions and target_parameters must be positive")
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.deliberation_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.localizer_head = nn.Linear(hidden_size, 1, device=device)
        self.core_adapter = nn.Linear(1, hidden_size, device=device)
        self.rollback_head = nn.Linear(hidden_size, 1, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        self.mix_gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=False)

    def _precore_state(
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
        _, hidden = self.deliberation_gru(events)
        recurrent = hidden[-1].unsqueeze(1).expand(-1, variables.shape[1], -1)
        precore = variables + recurrent
        localizer_logits = self.localizer_head(precore).squeeze(-1)
        return precore, localizer_logits

    @staticmethod
    def _validated_oracle_mask(
        mask: torch.Tensor,
        *,
        batch: int,
        variables: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mask.ndim != 2 or mask.shape != (batch, variables):
            raise ValueError("oracle_core_mask must match [batch,variables]")
        result = mask.to(device=device, dtype=dtype)
        if not torch.isfinite(result).all():
            raise ValueError("oracle_core_mask must be finite")
        if bool(((result < 0.0) | (result > 1.0)).any().item()):
            raise ValueError("oracle_core_mask membership must be within [0,1]")
        if not torch.equal(result.sum(dim=1), torch.full((batch,), 2.0, device=device, dtype=dtype)):
            raise ValueError("EXP-287 oracle_core_mask must contain exactly two members")
        return result

    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        *,
        mode: str,
        contradiction_observed: bool,
        oracle_core_mask: torch.Tensor | None = None,
    ) -> Exp287ModeOutput:
        if mode not in MODES:
            raise ValueError(f"unknown EXP-287 information mode: {mode}")
        precore, localizer_logits = self._precore_state(surface_events, variable_states)
        batch, variables, _ = precore.shape

        if mode == LEARNED_MODE and oracle_core_mask is not None:
            raise ValueError("learned mode cannot accept oracle core information")
        if mode == NULL_MODE and oracle_core_mask is not None:
            raise ValueError("null control cannot accept oracle core information")
        if not contradiction_observed and oracle_core_mask is not None:
            raise ValueError("oracle core cannot be delivered before the current contradiction")

        oracle_information_delivered = False
        if not contradiction_observed or mode == NULL_MODE:
            conflict_token = torch.zeros(
                batch,
                variables,
                device=precore.device,
                dtype=precore.dtype,
            )
        elif mode == LEARNED_MODE:
            conflict_token = learned_top2_mask(localizer_logits)
        else:
            if oracle_core_mask is None:
                raise ValueError("oracle mode requires oracle_core_mask after contradiction")
            conflict_token = self._validated_oracle_mask(
                oracle_core_mask,
                batch=batch,
                variables=variables,
                device=precore.device,
                dtype=precore.dtype,
            )
            oracle_information_delivered = True

        core_state = F.silu(self.core_adapter(conflict_token.unsqueeze(-1)))
        mixed = precore + torch.sigmoid(self.mix_gate) * core_state
        rollback_logits = self.rollback_head(mixed).squeeze(-1)
        verifier_confidence = torch.sigmoid(self.verifier_head(mixed).squeeze(-1))
        return Exp287ModeOutput(
            rollback_logits=rollback_logits,
            verifier_confidence=verifier_confidence,
            localizer_logits=localizer_logits,
            conflict_token=conflict_token,
            mode=mode,
            contradiction_observed=bool(contradiction_observed),
            oracle_information_delivered=oracle_information_delivered,
        )


def build_exp287_conflict_localizer(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> Exp287ConflictLocalizer:
    return Exp287ConflictLocalizer(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        device=device,
    )


def _linear_flops(inputs: int, outputs: int) -> int:
    return 2 * inputs * outputs + outputs


def _gru_flops(timesteps: int, hidden_size: int) -> int:
    return timesteps * (12 * hidden_size * hidden_size + 20 * hidden_size)


def _parameter_audit(model: Exp287ConflictLocalizer) -> dict[str, int]:
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


def _compute_ledger(
    model: Exp287ConflictLocalizer,
    *,
    timesteps: int,
    variables: int,
    max_search_steps: int,
) -> dict[str, Any]:
    if min(timesteps, variables, max_search_steps) <= 0:
        raise ValueError("EXP-287 compute geometry must be positive")
    d = model.d_model
    h = model.hidden_size
    event_projection = timesteps * _linear_flops(d, h)
    variable_projection = variables * _linear_flops(d, h)
    recurrent_deliberation = _gru_flops(timesteps, h)
    localizer_head = variables * _linear_flops(h, 1)
    core_adapter = variables * _linear_flops(1, h)
    rollback_head = variables * _linear_flops(h, 1)
    verifier_head = variables * _linear_flops(h, 1)
    fusion_and_gate = variables * (3 * h + 1)
    per_search_step = (
        event_projection
        + variable_projection
        + recurrent_deliberation
        + localizer_head
        + core_adapter
        + rollback_head
        + verifier_head
        + fusion_and_gate
    )
    return {
        "accounting_semantics": (
            "analytical scalar arithmetic FLOPs for the declared neural search geometry; "
            "localizer computation is charged in null, learned, and oracle information modes"
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
            "localizer_head": int(localizer_head),
            "conflict_core_or_null_adapter": int(core_adapter),
            "rollback_head": int(rollback_head),
            "verifier_head": int(verifier_head),
            "fusion_and_gate": int(fusion_and_gate),
        },
        "accounted_flops_per_search_step": int(per_search_step),
        "max_accounted_flops_per_episode": int(per_search_step * max_search_steps),
        "hardware_profiler_flops_claimed": False,
    }


def audit_exp287_information_modes(
    model: Exp287ConflictLocalizer,
    *,
    timesteps: int,
    variables: int,
    max_search_steps: int,
) -> dict[str, Any]:
    parameters = _parameter_audit(model)
    ledger = _compute_ledger(
        model,
        timesteps=timesteps,
        variables=variables,
        max_search_steps=max_search_steps,
    )
    return {
        "schema": "NLM-EXP-287-MATCHED-INFORMATION-MODES-DEV-V1",
        "experiment_id": "EXP-287",
        "modes": list(MODES),
        "parameters": parameters,
        "compute_ledger": ledger,
        "parameter_inventory_shared": True,
        "active_functional_parameters_shared": True,
        "optimizer_visible_parameters_shared": True,
        "accounted_flops_per_search_step_shared": True,
        "localizer_flops_charged_in_all_modes": True,
        "oracle_mode_deployable": False,
        "learned_mode_oracle_input_allowed": False,
        "precontradiction_conflict_token": "NULL_IN_ALL_MODES",
    }
