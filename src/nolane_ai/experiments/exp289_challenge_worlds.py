from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


_SCHEMA = "NLM-EXP-289-POST-FREEZE-CHALLENGE-CONTRACT-V1"
_EXPERIMENT_ID = "EXP-289"

_CONTRACT: dict[str, Any] = {
    "schema": _SCHEMA,
    "experiment_id": _EXPERIMENT_ID,
    "lane": "POST_FREEZE_CHALLENGE",
    "arms": ["no_nogood", "local_nogood"],
    "primary_endpoint": "repeat_dead_end_rate",
    "primary_direction": "lower",
    "mesi_relative_reduction": 0.25,
    "paired": True,
    "zero_opportunity_policy": "retain_raw_episode_exclude_from_rder_denominator",
    "epsilon_denominator_rescue": False,
    "protected_overprune_ceiling": 0.005,
    "protected_solution_floor_delta": -0.01,
    "memory_scope": "episode_local",
    "cross_episode_reuse": False,
    "cross_problem_reuse": False,
    "exact_subset_matching_only": True,
    "seed_authority": "FUTURE_PUBLIC_BEACON_AFTER_IMMUTABLE_GATE_A_FREEZE",
}


def frozen_exp289_challenge_contract() -> dict[str, Any]:
    """Return frozen challenge semantics without consuming future public entropy."""

    return deepcopy(_CONTRACT)


def challenge_contract_digest() -> str:
    """Return the canonical digest bound by pre-beacon authorization and seal."""

    return canonical_sha256(frozen_exp289_challenge_contract())
