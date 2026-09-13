from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from nolane_ai.experiments.exp301_training import EXP301_LR_CANDIDATES
from nolane_ai.experiments.exp301_training_runtime import (
    DECISION_TIE_BREAK_RULE,
    Exp301TrialResult,
    build_trial_plans,
    checkpoint_state_digest,
    deterministic_training_examples,
    run_training_trial,
    select_best_trial,
)


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")


class TinyArm(nn.Module):
    def __init__(self, vocab_size: int = 4608, width: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.projection = nn.Linear(width, vocab_size, bias=False)

    def _forward(self, tokens: torch.Tensor, effort: int) -> torch.Tensor:
        x = self.embedding(tokens)
        # Make the effort argument part of the real graph without creating an
        # architecture-specific training shortcut.
        x = x + float(effort) * 0.0
        return self.projection(x)

    def forward(self, tokens: torch.Tensor, *, loops: int | None = None, restarts: int | None = None) -> torch.Tensor:
        effort = loops if loops is not None else restarts
        if effort is None:
            raise ValueError("effort is required")
        return self._forward(tokens, int(effort))


def test_trial_plans_freeze_identical_search_budget_across_arms() -> None:
    by_arm = {
        arm: build_trial_plans(
            arm,
            root=2,
            max_steps=4,
            token_budget=4096,
            test_only=True,
        )
        for arm in ARMS
    }

    for plans in by_arm.values():
        assert tuple(plan.learning_rate for plan in plans) == EXP301_LR_CANDIDATES
        assert tuple(plan.trial_index for plan in plans) == (0, 1)
        assert all(plan.max_steps == 4 for plan in plans)
        assert all(plan.token_budget == 4096 for plan in plans)
        assert all(plan.scientific_evidence_eligible is False for plan in plans)

    reference = by_arm["A_FIXED"]
    for arm in ("B_LOOP_SIMPLE", "C_NRS_CORE"):
        for left, right in zip(reference, by_arm[arm]):
            assert replace(left, arm_id=arm) == right


def test_training_examples_and_order_digest_are_identical_across_arms() -> None:
    examples_a, digest_a = deterministic_training_examples(root=1, per_family=2)
    examples_b, digest_b = deterministic_training_examples(root=1, per_family=2)

    assert digest_a == digest_b
    assert [world.content_id for world in examples_a] == [world.content_id for world in examples_b]
    assert [world.family for world in examples_a] == sorted(
        [world.family for world in examples_a]
    )


def test_checkpoint_digest_is_deterministic_and_changes_with_model_state() -> None:
    torch.manual_seed(7)
    model = TinyArm()
    first = checkpoint_state_digest(model)
    second = checkpoint_state_digest(model)
    assert first == second

    with torch.no_grad():
        model.embedding.weight[0, 0].add_(1.0)
    changed = checkpoint_state_digest(model)
    assert changed != first


def test_training_trial_updates_model_and_emits_immutable_receipt() -> None:
    torch.manual_seed(11)
    model = TinyArm()
    before = checkpoint_state_digest(model)
    plan = build_trial_plans(
        "C_NRS_CORE",
        root=0,
        max_steps=2,
        token_budget=4096,
        test_only=True,
    )[0]
    examples, order_digest = deterministic_training_examples(root=0, per_family=1)

    result = run_training_trial(
        model,
        plan=plan,
        training_examples=examples,
        training_order_digest=order_digest,
    )

    assert result.receipt.arm_id == "C_NRS_CORE"
    assert result.receipt.root == 0
    assert result.receipt.trial_index == 0
    assert result.receipt.learning_rate == EXP301_LR_CANDIDATES[0]
    assert result.receipt.completed_steps == 2
    assert result.receipt.training_order_digest == order_digest
    assert result.receipt.checkpoint_digest == checkpoint_state_digest(model)
    assert result.receipt.checkpoint_digest != before
    assert result.receipt.scientific_evidence_eligible is False
    assert result.mean_training_loss > 0.0
    assert result.receipt.effort_schedule == (1, 2)


def test_fixed_arm_training_uses_restart_argument_but_same_effort_schedule() -> None:
    torch.manual_seed(13)
    model = TinyArm()
    plan = build_trial_plans(
        "A_FIXED",
        root=0,
        max_steps=4,
        token_budget=4096,
        test_only=True,
    )[0]
    examples, digest = deterministic_training_examples(root=0, per_family=1)
    result = run_training_trial(
        model,
        plan=plan,
        training_examples=examples,
        training_order_digest=digest,
    )
    assert result.receipt.effort_schedule == (1, 2, 4, 8)


def test_selection_is_highest_development_success_then_lower_lr() -> None:
    low_lr, high_lr = EXP301_LR_CANDIDATES
    plans = build_trial_plans(
        "B_LOOP_SIMPLE",
        root=0,
        max_steps=1,
        token_budget=1024,
        test_only=True,
    )

    high_score_low_lr = Exp301TrialResult.synthetic_for_selection(
        plan=plans[0], development_verified_success=0.75
    )
    lower_score_high_lr = Exp301TrialResult.synthetic_for_selection(
        plan=plans[1], development_verified_success=0.70
    )
    assert select_best_trial((high_score_low_lr, lower_score_high_lr)).plan.learning_rate == low_lr

    tie_low = Exp301TrialResult.synthetic_for_selection(
        plan=plans[0], development_verified_success=0.75
    )
    tie_high = Exp301TrialResult.synthetic_for_selection(
        plan=plans[1], development_verified_success=0.75
    )
    selected = select_best_trial((tie_high, tie_low))
    assert selected.plan.learning_rate == low_lr
    assert selected.selection_rule == DECISION_TIE_BREAK_RULE
    assert high_lr > low_lr


def test_runtime_rejects_unknown_arm_root_or_scientific_test_only_confusion() -> None:
    with pytest.raises(ValueError, match="arm"):
        build_trial_plans("UNKNOWN", root=0, max_steps=1, token_budget=100, test_only=True)
    with pytest.raises(ValueError, match="root"):
        build_trial_plans("A_FIXED", root=9, max_steps=1, token_budget=100, test_only=True)
    with pytest.raises(ValueError, match="positive"):
        build_trial_plans("A_FIXED", root=0, max_steps=0, token_budget=100, test_only=True)
