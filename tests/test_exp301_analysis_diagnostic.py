from __future__ import annotations

from dataclasses import replace

from nolane_ai.experiments.exp301_analysis import (
    DECISION_KILL,
    DECISION_STABILITY,
    reduce_exp301,
)
from tests.test_exp301_analysis import _full_rows


def _set_candidate_success(rows, *, efforts, families, success_count: int):
    updated = []
    for row in rows:
        if (
            row.arm_id == "C_NRS_CORE"
            and row.effort_multiplier in efforts
            and row.family in families
        ):
            index = int(row.content_id.rsplit("i", 1)[1])
            row = replace(row, verified_success=index < success_count)
        updated.append(row)
    return updated


def test_unseen_only_gain_with_representation_instability_is_diagnostic_only() -> None:
    rows = _full_rows(c_reasoning_by_root=(2, 2, 2, 2), a_reasoning=2)
    rows = _set_candidate_success(
        rows,
        efforts=(12, 16),
        families=(
            "iterative-grid-and-maze",
            "algorithmic-sequence-transform",
            "generator-heldout-abstract-transformation",
        ),
        success_count=8,
    )
    result = reduce_exp301(
        rows,
        bootstrap_samples=80,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=False,
    )
    assert result.decision == DECISION_STABILITY


def test_trained_depth_gain_must_not_be_mislabeled_as_unseen_depth_instability() -> None:
    rows = _full_rows(c_reasoning_by_root=(2, 2, 2, 2), a_reasoning=2)
    rows = _set_candidate_success(
        rows,
        efforts=(1, 2, 4, 8),
        families=("algorithmic-sequence-transform",),
        success_count=8,
    )
    result = reduce_exp301(
        rows,
        bootstrap_samples=80,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=False,
    )
    assert result.decision == DECISION_KILL
