from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget


@dataclass(frozen=True, slots=True)
class BeliefArmResult:
    decision_logits: torch.Tensor
    state: torch.Tensor
    state_semantics: str


class _MatchedBeliefArmBase(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, target_parameters: int, *, device=None) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("belief arm dimensions and target_parameters must be positive")
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(d_model, hidden_size, device=device)
        self.update_in = nn.Linear(hidden_size, 3 * hidden_size, device=device)
        self.update_out = nn.Linear(3 * hidden_size, hidden_size, device=device)
        self.state_scale = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.state_bias = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.decision_head = nn.Linear(hidden_size, 2, device=device)
        finalize_region_budget(self, target_parameters, device=device, frozen=False)

    @property
    def affine_madds_per_observation(self) -> int:
        h = self.hidden_size
        return self.d_model * h + 6 * h * h + 2 * h

    def _proposal(self, observation: torch.Tensor) -> torch.Tensor:
        encoded = F.silu(self.input_projection(observation))
        update = self.update_out(F.silu(self.update_in(encoded)))
        return update * torch.sigmoid(self.state_scale) + self.state_bias


class RecurrentHiddenBeliefArm(_MatchedBeliefArmBase):
    def run_with_state(self, observations: torch.Tensor) -> BeliefArmResult:
        if observations.ndim != 4 or observations.shape[-1] != self.d_model:
            raise ValueError("observations must be [batch,time,variables,d_model]")
        batch, timesteps, variables, _ = observations.shape
        hidden = torch.zeros(batch * variables, self.hidden_size, device=observations.device, dtype=observations.dtype)
        logits = torch.zeros(batch * variables, 2, device=observations.device, dtype=observations.dtype)
        for step in range(timesteps):
            proposal = self._proposal(observations[:, step].reshape(batch * variables, self.d_model))
            hidden = torch.tanh(hidden + proposal)
            evidence = self.decision_head(hidden)
            # Deliberately charge the same output-accumulation primitive as the explicit arm
            # without turning the recurrent comparator into an explicit belief accumulator.
            logits = torch.zeros_like(evidence) + evidence
        return BeliefArmResult(
            decision_logits=logits.reshape(batch, variables, 2),
            state=hidden.reshape(batch, variables, self.hidden_size),
            state_semantics="opaque_recurrent_hidden",
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.run_with_state(observations).decision_logits


class ExplicitBeliefArm(_MatchedBeliefArmBase):
    def run_with_state(self, observations: torch.Tensor) -> BeliefArmResult:
        if observations.ndim != 4 or observations.shape[-1] != self.d_model:
            raise ValueError("observations must be [batch,time,variables,d_model]")
        batch, timesteps, variables, _ = observations.shape
        belief_logits = torch.zeros(batch * variables, 2, device=observations.device, dtype=observations.dtype)
        zero_hidden = torch.zeros(batch * variables, self.hidden_size, device=observations.device, dtype=observations.dtype)
        for step in range(timesteps):
            proposal = self._proposal(observations[:, step].reshape(batch * variables, self.d_model))
            # Transient hidden-shaped transform matches recurrent compute but carries no cross-time state.
            transient = torch.tanh(zero_hidden + proposal)
            evidence = self.decision_head(transient)
            belief_logits = belief_logits + evidence
        belief_logits = belief_logits.reshape(batch, variables, 2)
        return BeliefArmResult(
            decision_logits=belief_logits,
            state=belief_logits,
            state_semantics="explicit_binary_belief_logits",
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.run_with_state(observations).decision_logits


def build_matched_belief_arm_pair(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> tuple[RecurrentHiddenBeliefArm, ExplicitBeliefArm]:
    recurrent = RecurrentHiddenBeliefArm(d_model, hidden_size, target_parameters, device=device)
    explicit = ExplicitBeliefArm(d_model, hidden_size, target_parameters, device=device)
    return recurrent, explicit


def _arm_audit(arm: nn.Module) -> dict[str, Any]:
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
        "affine_madds_per_observation": int(getattr(arm, "affine_madds_per_observation")),
    }


def _primitive_counts(arm: _MatchedBeliefArmBase, *, timesteps: int, variables: int) -> dict[str, int]:
    if min(timesteps, variables) <= 0:
        raise ValueError("timesteps and variables must be positive")
    d = arm.d_model
    h = arm.hidden_size
    per_variable_step = {
        "linear_multiply": d * h + h * (3 * h) + (3 * h) * h + h * 2,
        "linear_accumulate": h * (d - 1) + (3 * h) * (h - 1) + h * (3 * h - 1) + 2 * (h - 1),
        "bias_add": h + 3 * h + h + 2,
        "elementwise_multiply": h,
        "elementwise_add": 2 * h + 2,
        "sigmoid_elements": h,
        "tanh_elements": h,
        "silu_elements": 4 * h,
        "decision_head_calls": 1,
    }
    scale = timesteps * variables
    return {name: value * scale for name, value in per_variable_step.items()}


def _accounted_arithmetic_flops(counts: dict[str, int]) -> int:
    return (
        counts["linear_multiply"]
        + counts["linear_accumulate"]
        + counts["bias_add"]
        + counts["elementwise_multiply"]
        + counts["elementwise_add"]
    )


def account_matched_belief_arm_pair(
    recurrent: RecurrentHiddenBeliefArm,
    explicit: ExplicitBeliefArm,
    *,
    timesteps: int,
    variables: int,
) -> dict[str, Any]:
    if min(timesteps, variables) <= 0:
        raise ValueError("timesteps and variables must be positive")
    recurrent_counts = _primitive_counts(recurrent, timesteps=timesteps, variables=variables)
    explicit_counts = _primitive_counts(explicit, timesteps=timesteps, variables=variables)
    recurrent_flops = _accounted_arithmetic_flops(recurrent_counts)
    explicit_flops = _accounted_arithmetic_flops(explicit_counts)
    denominator = max(recurrent_flops, explicit_flops, 1)
    return {
        "schema": "NLM-EXP-282-COMPUTE-LEDGER-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "accounting_semantics": "analytical scalar arithmetic FLOPs plus exact nonlinear primitive signature; not hardware-profiler FLOPs",
        "geometry": {"timesteps": timesteps, "variables": variables},
        "recurrent_hidden": {
            "primitive_counts": recurrent_counts,
            "accounted_flops_per_episode": recurrent_flops,
        },
        "explicit_belief": {
            "primitive_counts": explicit_counts,
            "accounted_flops_per_episode": explicit_flops,
        },
        "primitive_operation_match": recurrent_counts == explicit_counts,
        "accounted_flops_match": recurrent_flops == explicit_flops,
        "relative_accounted_flop_difference": abs(recurrent_flops - explicit_flops) / denominator,
        "hardware_profiler_flops_claimed": False,
    }


def audit_matched_belief_arm_pair(
    recurrent: RecurrentHiddenBeliefArm,
    explicit: ExplicitBeliefArm,
    *,
    timesteps: int = 1,
    variables: int = 1,
) -> dict[str, Any]:
    recurrent_audit = _arm_audit(recurrent)
    explicit_audit = _arm_audit(explicit)
    compute = account_matched_belief_arm_pair(
        recurrent,
        explicit,
        timesteps=timesteps,
        variables=variables,
    )
    parameter_match = (
        recurrent_audit["total_parameters"] == explicit_audit["total_parameters"]
        and recurrent_audit["functional_parameters"] == explicit_audit["functional_parameters"]
    )
    affine_match = (
        recurrent_audit["affine_madds_per_observation"]
        == explicit_audit["affine_madds_per_observation"]
    )
    full_match = parameter_match and affine_match and compute["primitive_operation_match"] and compute["accounted_flops_match"]
    blockers = [] if full_match else ["matched parameter or compute ledger does not close"]
    blockers.append("matched arms are experiment-local components and are not yet integrated into confirmatory checkpoint/evaluator lineage")
    return {
        "schema": "NLM-EXP-282-MATCHED-BELIEF-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "recurrent_hidden": recurrent_audit,
        "explicit_belief": explicit_audit,
        "parameter_match": parameter_match,
        "observation_history_match": True,
        "affine_madd_match": affine_match,
        "primitive_operation_match": compute["primitive_operation_match"],
        "full_accounted_flop_match": compute["accounted_flops_match"],
        "relative_accounted_flop_difference": compute["relative_accounted_flop_difference"],
        "compute_ledger": compute,
        "remaining_blockers": blockers,
    }
