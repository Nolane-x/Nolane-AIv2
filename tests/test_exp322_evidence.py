from __future__ import annotations

from copy import deepcopy

import pytest

from nolane_ai.experiments.exp322_contract import CHECKPOINTS, InterventionSnapshot
from nolane_ai.experiments.exp322_evidence import (
    build_arm_evidence,
    build_final_evidence,
    validate_arm_evidence_digest,
)


def _snap(arm: str, step: int, *, token: float = 0.8, full: float = 0.3):
    return InterventionSnapshot(
        arm=arm,
        step=step,
        teacher_forced_answer_token_accuracy=token,
        teacher_forced_full_answer_exact=full,
        greedy_exact=0.3,
        eos_correctness=1.0,
        invalid_output_rate=0.0,
        answer_only_loss=2.0,
        gradient_norm_preclip=1.0,
        parameter_update_norm_ratio=0.01,
        nonfinite_events=0,
    )


def _arm(arm: str, *, passing: bool = False):
    snapshots = tuple(
        _snap(
            arm,
            step,
            token=(0.995 if passing and step == 2048 else 0.8),
            full=(0.95 if passing and step == 2048 else 0.3),
        )
        for step in CHECKPOINTS
    )
    return build_arm_evidence(
        arm=arm,
        snapshots=snapshots,
        family_summaries={str(step): {} for step in CHECKPOINTS},
        completed_step=2048,
        final_model_state_digest="1" * 64,
        final_optimizer_state_digest="2" * 64,
        final_rng_state_digest="3" * 64,
        invalid_reason=None,
    )


def test_arm_evidence_is_self_digesting_and_tamper_evident() -> None:
    payload = _arm("HOLD_1E4")
    validate_arm_evidence_digest(payload)
    forged = deepcopy(payload)
    forged["completed_step"] = 1536
    with pytest.raises(ValueError):
        validate_arm_evidence_digest(forged)


def test_final_evidence_reduces_budget_then_lr() -> None:
    result = build_final_evidence(
        _arm("HOLD_1E4", passing=True),
        _arm("DECAY_5E5", passing=True),
    )
    assert result["decision"] == "BUDGET_INSUFFICIENCY_EVIDENT"

    result = build_final_evidence(
        _arm("HOLD_1E4"),
        _arm("DECAY_5E5", passing=True),
    )
    assert result["decision"] == "LR_SCHEDULE_INSUFFICIENCY_EVIDENT"
    assert len(result["evidence_digest"]) == 64


def test_final_evidence_rejects_duplicate_arm_artifacts() -> None:
    with pytest.raises(ValueError, match="one artifact"):
        build_final_evidence(_arm("HOLD_1E4"), _arm("HOLD_1E4"))


def test_invalid_arm_reason_forces_invalid_intervention() -> None:
    hold = _arm("HOLD_1E4")
    hold = dict(hold)
    hold.pop("arm_evidence_digest")
    hold["invalid_reason"] = "NONFINITE_EVENT"
    hold["snapshots"] = []
    hold["family_summaries"] = {}
    hold["completed_step"] = 1100
    from nolane_ai.experiments.exp322_evidence import _digest
    hold["arm_evidence_digest"] = _digest(hold)

    result = build_final_evidence(hold, _arm("DECAY_5E5"))
    assert result["decision"] == "INVALID_INTERVENTION"
