from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


_SCHEMA = "NLM-EXP-286-POST-FREEZE-CHALLENGE-CONTRACT-V1"
_EXPERIMENT_ID = "EXP-286"

_CONTRACT: dict[str, Any] = {
    "schema": _SCHEMA,
    "experiment_id": _EXPERIMENT_ID,
    "lane": "POST_FREEZE_CHALLENGE",
    "arms": ["chronological_failure", "oracle_conflict_core"],
    "primary_endpoint": "accounted_reasoning_flops_to_verified_solution",
    "direction": "lower",
    "paired_design": True,
    "candidate_oracle_conflict_labels_visible": False,
    "control_oracle_conflict_labels_visible": True,
    "oracle_labeling_flops_charged_online": True,
    "seed_authority": "future_public_beacon_after_freeze",
}


def challenge_contract() -> dict[str, Any]:
    """Return the immutable scientific semantics for the future EXP-286 challenge lane.

    This contract intentionally contains no seed, beacon, generated world, or
    confirmatory observation.  It exists only so pre-beacon authorization can
    bind the challenge semantics before future public entropy is consumed.
    """

    return deepcopy(_CONTRACT)


def challenge_contract_digest() -> str:
    """Return the canonical digest of the frozen post-freeze challenge contract."""

    return canonical_sha256(challenge_contract())
