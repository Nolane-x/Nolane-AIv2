from __future__ import annotations

from dataclasses import replace

from nolane_ai.experiments.exp323_contract import (
    AUTHORIZATION_FLAGS,
    CHECKPOINTS,
    PARENT_EXP322_EVIDENCE_DIGEST,
    PARENT_EXP322_RUN_ID,
    InterventionSnapshot,
    preregistration_digest,
    preregistration_payload,
    reduce_intervention,
)


def _row(
    arm: str,
    step: int,
    *,
    token: float = 0.96,
    full: float = 0.69,
    greedy: float = 0.69,
    eos: float = 1.0,
    invalid: float = 0.0,
    loss: float = 0.44,
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


def _grid(*, hold=None, decay=None):
    return {
        "HOLD_5E5": hold or tuple(_row("HOLD_5E5", step) for step in CHECKPOINTS),
        "DECAY_2P5E5": decay or tuple(_row("DECAY_2P5E5", step) for step in CHECKPOINTS),
    }


def test_preregistration_binds_exp322_authority_and_digest() -> None:
    payload = preregistration_payload()
    boundary = payload["scientific_boundary"]
    replay = payload["reconstruction_authority"]
    assert PARENT_EXP322_RUN_ID == 35339004168
    assert PARENT_EXP322_EVIDENCE_DIGEST == (
        "1e3ea213718d1612a6370c0fb4124c869733747dc9992ca58345d16dcfb5aee2"
    )
    assert boundary["parent_exp322_disposition"] == "PARTIAL_CONTINUATION_PROGRESS"
    assert replay["exact_replay_required"] is True
    assert replay["replay_start_step"] == 1024
    assert replay["replay_end_step"] == 2048
    assert boundary["architecture_change"] is False
    assert boundary["scale_change"] is False
    assert preregistration_digest() == "27355c953330dea08cbd28e32d602c4a937722414b736bd774ad24676fb0fbc4"


def test_every_authorization_remains_false() -> None:
    assert AUTHORIZATION_FLAGS
    assert not any(AUTHORIZATION_FLAGS.values())


def test_hold_pass_has_precedence_even_if_both_pass() -> None:
    hold = tuple(
        _row("HOLD_5E5", step, token=(0.995 if step == 2560 else 0.96), full=(0.91 if step == 2560 else 0.69))
        for step in CHECKPOINTS
    )
    decay = tuple(
        _row("DECAY_2P5E5", step, token=(0.995 if step == 3072 else 0.96), full=(0.94 if step == 3072 else 0.69))
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(hold=hold, decay=decay)) == "REDUCED_LR_BUDGET_SUFFICIENT"


def test_second_decay_pass_requires_hold_failure() -> None:
    decay = tuple(
        _row("DECAY_2P5E5", step, token=(0.99 if step == 3072 else 0.96), full=(0.90 if step == 3072 else 0.69))
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(decay=decay)) == "SECOND_DECAY_SUFFICIENT"


def test_material_progress_uses_registered_post_2048_gains() -> None:
    hold = tuple(
        _row("HOLD_5E5", step, token=(0.978 if step == 3072 else 0.96), full=0.69)
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(hold=hold)) == "CONTINUATION_PROGRESS"

    decay = tuple(
        _row("DECAY_2P5E5", step, token=0.96, full=(0.8125 if step == 3072 else 0.69))
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(decay=decay)) == "CONTINUATION_PROGRESS"


def test_no_registered_rescue_below_all_thresholds() -> None:
    assert reduce_intervention(_grid()) == "NO_REGISTERED_RESCUE"


def test_greedy_improvement_alone_cannot_change_disposition() -> None:
    hold = tuple(
        _row("HOLD_5E5", step, token=0.96, full=0.69, greedy=(1.0 if step == 3072 else 0.69))
        for step in CHECKPOINTS
    )
    assert reduce_intervention(_grid(hold=hold)) == "NO_REGISTERED_RESCUE"


def test_missing_duplicate_or_misordered_geometry_fails_closed() -> None:
    assert reduce_intervention({"HOLD_5E5": _grid()["HOLD_5E5"]}) == "INVALID_INTERVENTION"

    broken = _grid()
    broken["DECAY_2P5E5"] = broken["DECAY_2P5E5"][:-1]
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"

    broken = _grid()
    broken["HOLD_5E5"] = tuple(reversed(broken["HOLD_5E5"]))
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"


def test_wrong_arm_nonfinite_or_split_floor_fails_or_does_not_pass() -> None:
    broken = _grid()
    rows = list(broken["DECAY_2P5E5"])
    rows[0] = replace(rows[0], arm="HOLD_5E5")
    broken["DECAY_2P5E5"] = tuple(rows)
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"

    broken = _grid()
    rows = list(broken["HOLD_5E5"])
    rows[-1] = replace(rows[-1], nonfinite_events=1)
    broken["HOLD_5E5"] = tuple(rows)
    assert reduce_intervention(broken) == "INVALID_INTERVENTION"

    hold = (
        _row("HOLD_5E5", 2304, token=0.995, full=0.80),
        _row("HOLD_5E5", 2560, token=0.97, full=0.95),
        _row("HOLD_5E5", 3072, token=0.97, full=0.80),
    )
    assert reduce_intervention(_grid(hold=hold)) != "REDUCED_LR_BUDGET_SUFFICIENT"
