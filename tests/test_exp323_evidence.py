from __future__ import annotations

from copy import deepcopy

import pytest

from nolane_ai.experiments.exp323_contract import CHECKPOINTS, InterventionSnapshot
from nolane_ai.experiments.exp323_evidence import (
    build_arm_evidence,
    build_final_evidence,
    expected_reconstruction_payload,
    validate_arm_evidence_digest,
)
from nolane_ai.experiments.exp323_identity import build_exp323_execution_identity


def _identity():
    return build_exp323_execution_identity(
        source_commit_sha="a" * 40,
        git_tree_sha="b" * 40,
        workflow_sha256="c" * 64,
    )


def _snap(arm: str, step: int, *, token: float = 0.96, full: float = 0.69):
    return InterventionSnapshot(
        arm=arm,
        step=step,
        teacher_forced_answer_token_accuracy=token,
        teacher_forced_full_answer_exact=full,
        greedy_exact=0.69,
        eos_correctness=1.0,
        invalid_output_rate=0.0,
        answer_only_loss=0.44,
        gradient_norm_preclip=1.0,
        parameter_update_norm_ratio=0.01,
        nonfinite_events=0,
    )


def _arm(arm: str, *, passing: bool = False):
    snapshots = tuple(
        _snap(
            arm,
            step,
            token=(0.995 if passing and step == 3072 else 0.96),
            full=(0.95 if passing and step == 3072 else 0.69),
        )
        for step in CHECKPOINTS
    )
    return build_arm_evidence(
        arm=arm,
        execution_identity=_identity(),
        reconstruction=expected_reconstruction_payload(),
        snapshots=snapshots,
        family_summaries={str(step): {} for step in CHECKPOINTS},
        completed_step=3072,
        final_model_state_digest="1" * 64,
        final_optimizer_state_digest="2" * 64,
        final_rng_state_digest="3" * 64,
        invalid_reason=None,
    )


def test_arm_evidence_is_self_digesting_and_replay_bound() -> None:
    payload = _arm("HOLD_5E5")
    validate_arm_evidence_digest(payload)

    forged = deepcopy(payload)
    forged["reconstruction"]["model_state_digest"] = "0" * 64
    materialized = dict(forged)
    materialized.pop("arm_evidence_digest")
    from nolane_ai.experiments.exp323_evidence import _digest
    forged["arm_evidence_digest"] = _digest(materialized)
    with pytest.raises(ValueError, match="reconstruction"):
        validate_arm_evidence_digest(forged)


def test_final_evidence_reduces_hold_then_second_decay() -> None:
    result = build_final_evidence(
        _arm("HOLD_5E5", passing=True),
        _arm("DECAY_2P5E5", passing=True),
    )
    assert result["decision"] == "REDUCED_LR_BUDGET_SUFFICIENT"

    result = build_final_evidence(
        _arm("HOLD_5E5"),
        _arm("DECAY_2P5E5", passing=True),
    )
    assert result["decision"] == "SECOND_DECAY_SUFFICIENT"
    assert len(result["evidence_digest"]) == 64


def test_final_evidence_rejects_duplicate_or_mismatched_identity() -> None:
    with pytest.raises(ValueError, match="one artifact"):
        build_final_evidence(_arm("HOLD_5E5"), _arm("HOLD_5E5"))

    hold = _arm("HOLD_5E5")
    decay = deepcopy(_arm("DECAY_2P5E5"))
    decay["execution_identity"]["source_commit_sha"] = "d" * 40
    from nolane_ai.experiments.exp323_evidence import _digest
    materialized = dict(decay)
    materialized.pop("arm_evidence_digest")
    decay["arm_evidence_digest"] = _digest(materialized)
    with pytest.raises(ValueError):
        build_final_evidence(hold, decay)
