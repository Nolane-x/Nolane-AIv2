from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.optimizer import functional_trainable_named_parameters


CanonicalNogood = tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Exp289ArmOutput:
    branch_logits: torch.Tensor
    verifier_confidence: torch.Tensor
    memory_query_logit: torch.Tensor
    memory_conditioning_used: bool
    representation_semantics: str


class EpisodeScopedNogoodStore:
    """Exact, episode-local nogood store with deterministic subset semantics.

    The store deliberately knows nothing about ground-truth validity, oracle
    conflict cores, future solutions, or cross-episode retrieval.  Scientific
    safety truth is evaluated *after* arm actions by the EXP-289 evaluator.
    """

    def __init__(self, *, episode_digest: str, problem_digest: str) -> None:
        if not isinstance(episode_digest, str) or not episode_digest:
            raise ValueError("episode_digest must be a non-empty string")
        if not isinstance(problem_digest, str) or not problem_digest:
            raise ValueError("problem_digest must be a non-empty string")
        self._episode_digest = episode_digest
        self._problem_digest = problem_digest
        self._nogoods: set[frozenset[tuple[str, int]]] = set()
        self._insertion_count = 0
        self._query_count = 0
        self._comparison_count = 0
        self._hit_count = 0
        self._canonicalization_operations = 0

    @property
    def episode_digest(self) -> str:
        return self._episode_digest

    @property
    def problem_digest(self) -> str:
        return self._problem_digest

    def __len__(self) -> int:
        return len(self._nogoods)

    def _assert_scope(
        self,
        *,
        episode_digest: str | None,
        problem_digest: str | None,
    ) -> None:
        if episode_digest is not None and episode_digest != self._episode_digest:
            raise ValueError("episode scope mismatch for episode-local nogood store")
        if problem_digest is not None and problem_digest != self._problem_digest:
            raise ValueError("problem scope mismatch for episode-local nogood store")

    def _canonicalize(self, assignment: Mapping[str, int]) -> CanonicalNogood:
        if not isinstance(assignment, Mapping):
            raise TypeError("assignment must be a mapping")
        canonical: list[tuple[str, int]] = []
        for name, value in assignment.items():
            if not isinstance(name, str) or not name:
                raise ValueError("assignment variable names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("assignment values must be integers")
            canonical.append((name, int(value)))
        canonical.sort()
        self._canonicalization_operations += max(len(canonical), 1)
        return tuple(canonical)

    def add(
        self,
        assignment: Mapping[str, int],
        *,
        dead_end_observed: bool,
        episode_digest: str | None = None,
        problem_digest: str | None = None,
    ) -> bool:
        self._assert_scope(
            episode_digest=episode_digest,
            problem_digest=problem_digest,
        )
        if dead_end_observed is not True:
            raise ValueError("nogood insertion requires an observed dead end")
        canonical = self._canonicalize(assignment)
        if not canonical:
            raise ValueError("episode-local nogood must be non-empty")
        key = frozenset(canonical)
        if key in self._nogoods:
            return False
        self._nogoods.add(key)
        self._insertion_count += 1
        return True

    def matches(
        self,
        assignment: Mapping[str, int],
        *,
        episode_digest: str | None = None,
        problem_digest: str | None = None,
    ) -> bool:
        self._assert_scope(
            episode_digest=episode_digest,
            problem_digest=problem_digest,
        )
        current = frozenset(self._canonicalize(assignment))
        self._query_count += 1
        ordered = sorted(
            self._nogoods,
            key=lambda key: tuple(sorted(key)),
        )
        for nogood in ordered:
            self._comparison_count += 1
            if nogood.issubset(current):
                self._hit_count += 1
                return True
        return False

    def canonical_keys(self) -> tuple[CanonicalNogood, ...]:
        return tuple(
            sorted(
                (tuple(sorted(key)) for key in self._nogoods),
            )
        )

    def snapshot_counters(self) -> dict[str, int]:
        return {
            "insertion_count": int(self._insertion_count),
            "query_count": int(self._query_count),
            "comparison_count": int(self._comparison_count),
            "hit_count": int(self._hit_count),
            "canonicalization_operations": int(self._canonicalization_operations),
        }


class _MatchedExp289ArmBase(nn.Module):
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
            raise ValueError("EXP-289 dimensions and target_parameters must be positive")
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.deliberation_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.memory_adapter = nn.Linear(1, hidden_size, device=device)
        self.branch_head = nn.Linear(hidden_size, 1, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        self.memory_query_head = nn.Linear(hidden_size, 1, device=device)
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
        memory_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, hidden = self.deliberation_gru(events)
        recurrent = hidden[-1].unsqueeze(1).expand(-1, variables.shape[1], -1)
        memory_state = F.silu(self.memory_adapter(memory_token))
        mixed = variables + recurrent + torch.sigmoid(self.mix_gate) * memory_state
        return (
            self.branch_head(mixed).squeeze(-1),
            torch.sigmoid(self.verifier_head(mixed).squeeze(-1)),
            self.memory_query_head(mixed).squeeze(-1),
        )


class NoNogoodArm(_MatchedExp289ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
    ) -> Exp289ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        null_memory = torch.zeros(
            variables.shape[0],
            variables.shape[1],
            1,
            device=variables.device,
            dtype=variables.dtype,
        )
        branch_logits, verifier_confidence, memory_query_logit = self._shared_state(
            events,
            variables,
            null_memory,
        )
        return Exp289ArmOutput(
            branch_logits=branch_logits,
            verifier_confidence=verifier_confidence,
            memory_query_logit=memory_query_logit,
            memory_conditioning_used=False,
            representation_semantics="hybrid_reasoner_with_canonical_null_episode_memory_token",
        )


class LocalNogoodArm(_MatchedExp289ArmBase):
    def forward(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
        *,
        memory_hit: torch.Tensor,
    ) -> Exp289ArmOutput:
        events, variables = self._validate_common(surface_events, variable_states)
        if memory_hit.ndim != 1 or memory_hit.shape[0] != variables.shape[0]:
            raise ValueError("memory_hit must be [batch]")
        hit = memory_hit.to(device=variables.device, dtype=variables.dtype)
        if not torch.isfinite(hit).all():
            raise ValueError("memory_hit must be finite and binary")
        if bool(((hit != 0.0) & (hit != 1.0)).any().item()):
            raise ValueError("memory_hit must be binary")
        memory_token = hit[:, None, None].expand(-1, variables.shape[1], 1)
        branch_logits, verifier_confidence, memory_query_logit = self._shared_state(
            events,
            variables,
            memory_token,
        )
        return Exp289ArmOutput(
            branch_logits=branch_logits,
            verifier_confidence=verifier_confidence,
            memory_query_logit=memory_query_logit,
            memory_conditioning_used=bool((hit > 0.0).any().item()),
            representation_semantics="hybrid_reasoner_conditioned_on_exact_current_episode_nogood_query_hit",
        )


def build_matched_exp289_arm_pair(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> tuple[NoNogoodArm, LocalNogoodArm]:
    no_nogood = NoNogoodArm(
        d_model,
        hidden_size,
        target_parameters,
        device=device,
    )
    local = LocalNogoodArm(
        d_model,
        hidden_size,
        target_parameters,
        device=device,
    )
    local.load_state_dict(no_nogood.state_dict())
    return no_nogood, local


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
    arm: _MatchedExp289ArmBase,
    *,
    restarts: int,
    variables: int,
    max_search_steps: int,
    local_memory_enabled: bool,
) -> dict[str, Any]:
    if min(restarts, variables, max_search_steps) <= 0:
        raise ValueError("EXP-289 compute geometry must be positive")
    d = arm.d_model
    h = arm.hidden_size
    total_steps = restarts * max_search_steps

    event_projection = _linear_flops(d, h)
    variable_projection = variables * _linear_flops(d, h)
    recurrent_deliberation = _gru_flops(1, h)
    memory_adapter = variables * _linear_flops(1, h)
    branch_head = variables * _linear_flops(h, 1)
    verifier_head = variables * _linear_flops(h, 1)
    memory_query_head = variables * _linear_flops(h, 1)
    fusion_and_gate = variables * (3 * h + 1)
    neural_per_step = (
        event_projection
        + variable_projection
        + recurrent_deliberation
        + memory_adapter
        + branch_head
        + verifier_head
        + memory_query_head
        + fusion_and_gate
    )
    neural_maximum = total_steps * neural_per_step

    if local_memory_enabled:
        canonicalization = total_steps * variables * 2
        insertion = total_steps
        subset_comparisons = total_steps * total_steps
        memory_maximum = canonicalization + insertion + subset_comparisons
    else:
        canonicalization = 0
        insertion = 0
        subset_comparisons = 0
        memory_maximum = 0

    total_maximum = neural_maximum + memory_maximum
    return {
        "accounting_semantics": (
            "analytical neural FLOPs plus exact symbolic episode-local memory operations; "
            "memory information is not counted as neural parameters and its maintenance/retrieval work is charged"
        ),
        "geometry": {
            "restarts": int(restarts),
            "variables": int(variables),
            "max_search_steps": int(max_search_steps),
            "d_model": int(d),
            "hidden_size": int(h),
        },
        "components_per_search_step": {
            "event_projection": int(event_projection),
            "variable_projection": int(variable_projection),
            "recurrent_deliberation": int(recurrent_deliberation),
            "nogood_or_null_memory_adapter": int(memory_adapter),
            "branch_head": int(branch_head),
            "verifier_head": int(verifier_head),
            "memory_query_head": int(memory_query_head),
            "fusion_and_gate": int(fusion_and_gate),
        },
        "accounted_neural_flops_per_search_step": int(neural_per_step),
        "max_neural_accounted_flops_per_episode": int(neural_maximum),
        "max_memory_components_per_episode": {
            "canonicalization_operations": int(canonicalization),
            "store_insertion_operations": int(insertion),
            "subset_match_comparison_operations": int(subset_comparisons),
        },
        "max_memory_accounted_operations_per_episode": int(memory_maximum),
        "max_accounted_cost_per_episode": int(total_maximum),
        "hardware_profiler_flops_claimed": False,
    }


def audit_matched_exp289_arm_pair(
    no_nogood: NoNogoodArm,
    local_nogood: LocalNogoodArm,
    *,
    restarts: int,
    variables: int,
    max_search_steps: int,
    max_accounted_cost_per_episode: int | None = None,
) -> dict[str, Any]:
    no_audit = _parameter_audit(no_nogood)
    local_audit = _parameter_audit(local_nogood)

    parameter_match = no_audit["total_parameters"] == local_audit["total_parameters"]
    functional_match = no_audit["functional_parameters"] == local_audit["functional_parameters"]
    active_match = (
        no_audit["active_functional_parameters"]
        == local_audit["active_functional_parameters"]
        == no_audit["functional_parameters"]
    )
    optimizer_match = (
        no_audit["optimizer_visible_parameters"]
        == local_audit["optimizer_visible_parameters"]
        == no_audit["functional_parameters"]
    )
    reserve_match = no_audit["reserved_parameters"] == local_audit["reserved_parameters"]
    if not all((parameter_match, functional_match, active_match, optimizer_match, reserve_match)):
        raise ValueError("EXP-289 matched pair parameter contract is not closed")

    no_ledger = _compute_ledger(
        no_nogood,
        restarts=restarts,
        variables=variables,
        max_search_steps=max_search_steps,
        local_memory_enabled=False,
    )
    local_ledger = _compute_ledger(
        local_nogood,
        restarts=restarts,
        variables=variables,
        max_search_steps=max_search_steps,
        local_memory_enabled=True,
    )
    required = max(
        int(no_ledger["max_accounted_cost_per_episode"]),
        int(local_ledger["max_accounted_cost_per_episode"]),
    )
    declared = required if max_accounted_cost_per_episode is None else int(max_accounted_cost_per_episode)
    if declared <= 0 or required > declared:
        raise ValueError(f"EXP-289 compute budget exceeded: required {required}, declared {declared}")

    return {
        "schema": "NLM-EXP-289-MATCHED-NOGOOD-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "no_nogood": no_audit,
        "local_nogood": local_audit,
        "parameter_match": parameter_match,
        "functional_parameter_match": functional_match,
        "active_functional_parameter_match": active_match,
        "optimizer_visible_parameter_match": optimizer_match,
        "reserve_parameter_match": reserve_match,
        "memory_scope_closed": True,
        "memory_scope_receipt": {
            "scope": "episode_local",
            "cross_episode_reuse": False,
            "cross_problem_reuse": False,
            "exact_subset_matching_only": True,
            "oracle_conflict_core_used": False,
            "ground_truth_gates_arm_action": False,
            "learned_clause_generalization_claimed": False,
        },
        "compute_budget_closed": True,
        "hardware_profiler_flops_claimed": False,
        "compute_ledger": {
            "no_nogood": no_ledger,
            "local_nogood": local_ledger,
        },
        "declared_max_accounted_cost_per_episode": int(declared),
    }
