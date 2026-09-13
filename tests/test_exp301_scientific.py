from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_scientific import (
    ScientificTrialResult,
    build_scientific_arm,
    forward_scientific_arm,
    frozen_trial_plan,
    greedy_generate,
    model_state_digest,
    select_development_trial,
)


def test_frozen_trial_plan_is_exactly_three_arms_times_two_lrs() -> None:
    plan = frozen_trial_plan(root=2)
    assert len(plan) == 6
    assert {item.arm_id for item in plan} == {"A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"}
    assert {item.learning_rate for item in plan} == {1e-4, 3e-4}
    assert all(item.root == 2 for item in plan)
    assert len({item.model_init_seed for item in plan}) == 6


def test_real_scientific_arms_are_exact_10m_and_accept_frozen_efforts() -> None:
    for arm_id in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"):
        compiled = build_scientific_arm(arm_id, device="meta")
        assert sum(p.numel() for p in compiled.model.parameters() if p.requires_grad) == 10_000_000
        tokens = torch.zeros((1, 3), dtype=torch.long, device="meta")
        logits = forward_scientific_arm(compiled, tokens, effort=4)
        assert logits.shape == (1, 3, 4_608)


def _trial(*, lr: float, score: float, per_family: tuple[tuple[str, float], ...]) -> ScientificTrialResult:
    return ScientificTrialResult(
        arm_id="C_NRS_CORE",
        root=0,
        learning_rate=lr,
        model_init_seed=123,
        training_steps=512,
        development_family_balanced_score=score,
        development_family_scores=per_family,
        checkpoint_digest="a" * 64,
        trial_receipt_digest=("b" if lr == 1e-4 else "c") * 64,
    )


def test_development_selection_is_family_balanced_and_ties_choose_lower_lr() -> None:
    families = (
        ("algorithmic-sequence-transform", 0.50),
        ("generator-heldout-abstract-transformation", 0.50),
        ("iterative-grid-and-maze", 0.50),
        ("language-sequence-control", 0.50),
    )
    low = _trial(lr=1e-4, score=0.50, per_family=families)
    high = _trial(lr=3e-4, score=0.50, per_family=families)
    selected = select_development_trial((high, low))
    assert selected.learning_rate == 1e-4

    better = replace(high, development_family_balanced_score=0.51)
    assert select_development_trial((low, better)).learning_rate == 3e-4


def test_greedy_generation_never_requires_canonical_answer() -> None:
    class StubCompiled:
        arm_id = type("Arm", (), {"value": "C_NRS_CORE"})()

        class Model(torch.nn.Module):
            def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
                logits = torch.full((*tokens.shape, 4_608), -1000.0, device=tokens.device)
                # Emit ASCII '7' and then EOS once it is in the context.
                next_id = 3 if int(tokens[0, -1]) == 4 + ord("7") else 4 + ord("7")
                logits[:, -1, next_id] = 1000.0
                return logits

        model = Model()

    result = greedy_generate(StubCompiled(), prompt="Q", effort=1, max_new_tokens=8)
    assert result.candidate_answer == "7"
    assert result.generated_token_count == 2


def test_model_state_digest_changes_when_parameter_changes() -> None:
    compiled = build_scientific_arm("C_NRS_CORE", device="cpu")
    before = model_state_digest(compiled.model)
    with torch.no_grad():
        parameter = next(compiled.model.parameters())
        parameter.view(-1)[0].add_(1.0)
    after = model_state_digest(compiled.model)
    assert before != after
