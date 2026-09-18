from __future__ import annotations

from dataclasses import replace

from nolane_ai.experiments.exp322_contract import (
    AUTHORIZATION_FLAGS,
    CHECKPOINTS,
    PARENT_EXP321_EVIDENCE_DIGEST,
    PARENT_EXP321_RUN_ID,
    InterventionSnapshot,
    preregistration_digest,
    preregistration_payload,
    reduce_intervention,
)


def _row(
    arm: str,
    step: int,
    *,
    token: float = 0.80,
    full: float = 0.30,
    greedy: float = 0.30,
    eos: float = 1.0,
    invalid: float = 0.0,
    loss: float = 2.0,
    grad: float = 1.0,
    update: float = 0.01,
    nonfinite: int = 0,
) -> InterventionSnapshot:
    return InterventionSnapshot(
        arm=arm,
        step=step,
        teacher_forced_answer_token_accuracy=token,
        teacher_forced_full_answer_exact=full,
        greedy_exact=greedy,
        eos_correctness=eos,
        invalid_output_rate=invalid,
        answer_only_loss=loss,
        gradient_norm_preclip=grad,
        parameter_update_norm_ratio=update,
        nonfinite_events=nonfinite,
    )


def _grid(
    *,
    hold: tuple[InterventionSnapshot, ...] | None = None,
    decay: tuple[InterventionSnapshot, ...] | None = None,
):
    return {
        "HOLD_1E4": hold
        or tuple(_row("HOLD_1E4", step) for step in CHECKPOINTS),
        "DECAY_5E5": decay
        or tuple(_row("DECAY_5E5", step) for step in CHECKPOINTS),
    }


def test_preregistration_binds_exp321_authority_and_digest() -> None:
    payload = preregistration_payload()
    boundary = payload["scientific_boundary"]
    assert PARENT_EXP321_RUN_ID == 35331243762
    assert PARENT_EXP321_EVIDENCE_DIGEST == (
        "5f3bd6cbacef0af4883e33f14bafd56b49a307df88641e837bb9ecdf004d5091"
    )
    assert boundary["parent_exp321_disposition"] == (
        "TEACHER_FORCED_FOUNDATION_INSUFFICIENT"
    )
    assert boundary["architecture_change"] is False
    assert boundary["tokenizer_change"] is False
    assert boundary["dataset_change"] is False
    assert boundary["objective_change"] is False
    assert boundary["scale_change"] is False
    assert preregistration_digest() == (
        "c56f1cabbba2bcb14a916f13f500d0bbbe630788e52a8689028d017cbced22f0"
    )


def test_every_authorization_remains_false() -> None:
    assert AUTHORIZATION_FLAGS
    assert not any(AUTHORIZATION_FLAGS.values())


def test_hold_floor_pass_has_budget_precedence_even_if_decay_also_passes() -> None:
    hold = tuple(
        _row(
            "HOLD_1E4",
            step,
            token=(0.995 if step == 1536 else 0.80),
            full=(0.91 if step == 1536 else 0.30),
        )
        for step in CHECKPOINTS
    )
    decay = tuple(
        _row(
            "DECAY_5E5",
            step,
            token=(0.995 if step == 2048 else 0.80),
            full=(0.95 if step == 2048 else 0.30),
        )
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(hold=hold, decay=decay)) == (
        "BUDGET_INSUFFICIENCY_EVIDENT"
    )


def test_decay_floor_pass_requires_hold_failure() -> None:
    decay = tuple(
        _row(
            "DECAY_5E5",
            step,
            token=(0.99 if step == 2048 else 0.80),
            full=(0.90 if step == 2048 else 0.30),
        )
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(decay=decay)) == (
        "LR_SCHEDULE_INSUFFICIENCY_EVIDENT"
    )


def test_partial_progress_uses_registered_teacher_forced_gains() -> None:
    hold = tuple(
        _row(
            "HOLD_1E4",
            step,
            token=(0.88 if step == 2048 else 0.80),
            full=0.30,
        )
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(hold=hold)) == (
        "PARTIAL_CONTINUATION_PROGRESS"
    )


def test_no_registered_rescue_when_no_floor_or_partial_gain() -> None:
    assert reduce_intervention(_grid()) == "NO_REGISTERED_RESCUE"


def test_greedy_improvement_alone_cannot_change_disposition() -> None:
    hold = tuple(
        _row(
            "HOLD_1E4",
            step,
            token=0.80,
            full=0.30,
            greedy=(1.0 if step == 2048 else 0.30),
        )
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(hold=hold)) == "NO_REGISTERED_RESCUE"


def test_missing_arm_or_checkpoint_fails_closed() -> None:
    assert reduce_intervention({"HOLD_1E4": _grid()["HOLD_1E4"]}) == (
        "INVALID_INTERVENTION"
    )
    broken = _grid()
    broken["DECAY_5E5"] = broken["DECAY_5E5"][:-1]
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"


def test_wrong_checkpoint_order_or_arm_identity_fails_closed() -> None:
    broken = _grid()
    broken["HOLD_1E4"] = tuple(reversed(broken["HOLD_1E4"]))
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"

    broken = _grid()
    rows = list(broken["DECAY_5E5"])
    rows[1] = replace(rows[1], arm="HOLD_1E4")
    broken["DECAY_5E5"] = tuple(rows)
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"


def test_nonfinite_or_nonfinite_event_fails_closed() -> None:
    broken = _grid()
    rows = list(broken["HOLD_1E4"])
    rows[0] = replace(
        rows[0],
        teacher_forced_answer_token_accuracy=float("nan"),
    )
    broken["HOLD_1E4"] = tuple(rows)
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"

    broken = _grid()
    rows = list(broken["DECAY_5E5"])
    rows[-1] = replace(rows[-1], nonfinite_events=1)
    broken["DECAY_5E5"] = tuple(rows)
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"


def test_floor_requires_both_primary_metrics_at_same_checkpoint() -> None:
    hold = (
        _row("HOLD_1E4", 1280, token=0.995, full=0.50),
        _row("HOLD_1E4", 1536, token=0.80, full=0.95),
        _row("HOLD_1E4", 2048, token=0.80, full=0.50),
    )
    assert reduce_intervention(_grid(hold=hold)) != (
        "BUDGET_INSUFFICIENCY_EVIDENT"
    )
