from __future__ import annotations

import importlib

import pytest


def _selection_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_selection")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 selection rules are not implemented yet: {exc}")


def _stage_a(selection, **overrides):
    values = {
        "arm_id": "A_FIXED",
        "root": 0,
        "learning_rate": 1e-4,
        "step": 1024,
        "initial_answer_only_loss": 4.0,
        "answer_only_loss": 0.8,
        "answer_token_accuracy": 0.995,
        "exact_match": 0.95,
        "eos_correctness": 0.97,
        "invalid_output_rate": 0.0,
        "nonfinite_events": 0,
    }
    values.update(overrides)
    return selection.StageASelectionRecord(**values)


def _stage_b(selection, **overrides):
    values = {
        "arm_id": "A_FIXED",
        "root": 1,
        "step": 512,
        "train_exact_match": 0.92,
        "iid_answer_only_loss": 0.7,
        "iid_answer_token_accuracy": 0.96,
        "iid_family_balanced_exact_match": 0.30,
        "iid_family_exact": (
            ("algorithmic-sequence-transform", 0.30),
            ("generator-heldout-abstract-transformation", 0.30),
            ("iterative-grid-and-maze", 0.30),
            ("language-sequence-control", 0.30),
        ),
        "eos_correctness": 0.96,
        "invalid_output_rate": 0.0,
    }
    values.update(overrides)

    balanced_overridden = "iid_family_balanced_exact_match" in overrides
    families_overridden = "iid_family_exact" in overrides
    if balanced_overridden and not families_overridden:
        balanced = float(values["iid_family_balanced_exact_match"])
        values["iid_family_exact"] = tuple(
            (family, balanced)
            for family in (
                "algorithmic-sequence-transform",
                "generator-heldout-abstract-transformation",
                "iterative-grid-and-maze",
                "language-sequence-control",
            )
        )
    elif families_overridden and not balanced_overridden:
        family_exact = tuple(values["iid_family_exact"])
        values["iid_family_balanced_exact_match"] = sum(score for _, score in family_exact) / len(
            family_exact
        )

    return selection.StageBSelectionRecord(**values)


def test_stage_a_floor_is_exactly_preregistered() -> None:
    selection = _selection_module()
    assert selection.stage_a_passes(_stage_a(selection)) is True
    assert selection.stage_a_passes(_stage_a(selection, exact_match=0.899)) is False
    assert selection.stage_a_passes(_stage_a(selection, answer_token_accuracy=0.989)) is False
    assert selection.stage_a_passes(_stage_a(selection, eos_correctness=0.949)) is False
    assert selection.stage_a_passes(_stage_a(selection, answer_only_loss=1.001)) is False
    assert selection.stage_a_passes(_stage_a(selection, invalid_output_rate=0.011)) is False
    assert selection.stage_a_passes(_stage_a(selection, nonfinite_events=1)) is False


def test_stage_a_lr_selection_uses_frozen_tie_order() -> None:
    selection = _selection_module()

    low = _stage_a(selection, learning_rate=1e-4)
    high = _stage_a(selection, learning_rate=3e-4, exact_match=0.96)
    assert selection.select_stage_a_learning_rate((low, high)).learning_rate == pytest.approx(3e-4)

    high = _stage_a(
        selection,
        learning_rate=3e-4,
        answer_token_accuracy=0.999,
    )
    assert selection.select_stage_a_learning_rate((low, high)).learning_rate == pytest.approx(3e-4)

    high = _stage_a(selection, learning_rate=3e-4, answer_only_loss=0.7)
    assert selection.select_stage_a_learning_rate((low, high)).learning_rate == pytest.approx(3e-4)

    exact_tie_high = _stage_a(selection, learning_rate=3e-4)
    assert selection.select_stage_a_learning_rate((low, exact_tie_high)).learning_rate == pytest.approx(1e-4)


def test_stage_a_selection_requires_both_frozen_candidates_and_one_arm() -> None:
    selection = _selection_module()
    low = _stage_a(selection, learning_rate=1e-4)
    with pytest.raises(ValueError, match="two"):
        selection.select_stage_a_learning_rate((low,))
    with pytest.raises(ValueError, match="learning-rate"):
        selection.select_stage_a_learning_rate((low, _stage_a(selection, learning_rate=1e-4)))
    with pytest.raises(ValueError, match="arm"):
        selection.select_stage_a_learning_rate(
            (low, _stage_a(selection, arm_id="C_NRS_CORE", learning_rate=3e-4))
        )


def test_stage_b_floor_is_exactly_preregistered() -> None:
    selection = _selection_module()
    assert selection.stage_b_checkpoint_passes(_stage_b(selection)) is True
    assert selection.stage_b_checkpoint_passes(_stage_b(selection, train_exact_match=0.899)) is False
    assert selection.stage_b_checkpoint_passes(_stage_b(selection, iid_answer_token_accuracy=0.949)) is False
    assert selection.stage_b_checkpoint_passes(
        _stage_b(selection, iid_family_balanced_exact_match=0.249)
    ) is False
    assert selection.stage_b_checkpoint_passes(_stage_b(selection, eos_correctness=0.949)) is False
    assert selection.stage_b_checkpoint_passes(_stage_b(selection, invalid_output_rate=0.011)) is False
    only_two = (
        ("algorithmic-sequence-transform", 0.2),
        ("generator-heldout-abstract-transformation", 0.2),
        ("iterative-grid-and-maze", 0.09),
        ("language-sequence-control", 0.09),
    )
    assert selection.stage_b_checkpoint_passes(_stage_b(selection, iid_family_exact=only_two)) is False


def test_stage_b_selects_earliest_checkpoint_that_passes_all_floors() -> None:
    selection = _selection_module()
    records = (
        _stage_b(selection, step=512, iid_family_balanced_exact_match=0.24),
        _stage_b(selection, step=1024, iid_family_balanced_exact_match=0.30),
        _stage_b(selection, step=2048, iid_family_balanced_exact_match=0.80),
    )
    selected = selection.select_stage_b_checkpoint(records)
    assert selected.step == 1024
    assert selection.stage_b_checkpoint_passes(selected) is True


def test_stage_b_fallback_ranking_is_frozen() -> None:
    selection = _selection_module()
    base = _stage_b(
        selection,
        step=512,
        train_exact_match=0.2,
        iid_family_balanced_exact_match=0.10,
        iid_answer_token_accuracy=0.50,
        iid_answer_only_loss=1.2,
    )
    records = (
        base,
        _stage_b(
            selection,
            step=1024,
            train_exact_match=0.2,
            iid_family_balanced_exact_match=0.20,
            iid_answer_token_accuracy=0.40,
            iid_answer_only_loss=1.5,
        ),
        _stage_b(
            selection,
            step=2048,
            train_exact_match=0.2,
            iid_family_balanced_exact_match=0.20,
            iid_answer_token_accuracy=0.60,
            iid_answer_only_loss=1.7,
        ),
    )
    assert selection.select_stage_b_checkpoint(records).step == 2048

    records = (
        base,
        _stage_b(
            selection,
            step=1024,
            train_exact_match=0.2,
            iid_family_balanced_exact_match=0.20,
            iid_answer_token_accuracy=0.60,
            iid_answer_only_loss=1.1,
        ),
        _stage_b(
            selection,
            step=2048,
            train_exact_match=0.2,
            iid_family_balanced_exact_match=0.20,
            iid_answer_token_accuracy=0.60,
            iid_answer_only_loss=1.0,
        ),
    )
    assert selection.select_stage_b_checkpoint(records).step == 2048

    records = (
        base,
        _stage_b(
            selection,
            step=1024,
            train_exact_match=0.2,
            iid_family_balanced_exact_match=0.20,
            iid_answer_token_accuracy=0.60,
            iid_answer_only_loss=1.0,
        ),
        _stage_b(
            selection,
            step=2048,
            train_exact_match=0.2,
            iid_family_balanced_exact_match=0.20,
            iid_answer_token_accuracy=0.60,
            iid_answer_only_loss=1.0,
        ),
    )
    assert selection.select_stage_b_checkpoint(records).step == 1024


def test_stage_b_selection_rejects_missing_checkpoint_or_root() -> None:
    selection = _selection_module()
    two = (_stage_b(selection, step=512), _stage_b(selection, step=1024))
    with pytest.raises(ValueError, match="checkpoints"):
        selection.select_stage_b_checkpoint(two)

    all_records = []
    for root in (1, 2, 3):
        for step in (512, 1024, 2048):
            all_records.append(_stage_b(selection, root=root, step=step))
    with pytest.raises(ValueError, match="roots"):
        selection.select_stage_b_checkpoints_by_root(tuple(all_records), arm_id="A_FIXED")


def test_stage_b_cross_root_helper_requires_three_of_four_floor_passes() -> None:
    selection = _selection_module()
    selected = tuple(
        _stage_b(
            selection,
            root=root,
            step=512,
            iid_family_balanced_exact_match=(0.30 if root != 4 else 0.10),
        )
        for root in (1, 2, 3, 4)
    )
    assert selection.stage_b_cross_root_passes(selected) is True

    selected = tuple(
        _stage_b(
            selection,
            root=root,
            step=512,
            iid_family_balanced_exact_match=(0.30 if root in (1, 2) else 0.10),
        )
        for root in (1, 2, 3, 4)
    )
    assert selection.stage_b_cross_root_passes(selected) is False
