from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_scientific import (
    ScientificTrialResult,
    build_scientific_arm,
    family_balanced_development_score,
    forward_scientific_arm,
    frozen_trial_plan,
    greedy_generate,
    model_state_digest,
    run_scientific_trial,
    scientific_challenge_generate,
    scientific_train_step,
    select_development_trial,
)
from nolane_ai.experiments.exp301_worlds import generate_world_instance


def test_frozen_trial_plan_is_exactly_three_arms_times_two_lrs() -> None:
    plan = frozen_trial_plan(root=2)
    assert len(plan) == 6
    assert {item.arm_id for item in plan} == {"A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"}
    assert {item.learning_rate for item in plan} == {1e-4, 3e-4}
    assert all(item.root == 2 for item in plan)
    assert len({item.model_init_seed for item in plan}) == 6


def test_real_scientific_arms_are_exact_10m() -> None:
    for arm_id in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"):
        compiled = build_scientific_arm(arm_id, device="meta")
        assert sum(p.numel() for p in compiled.model.parameters() if p.requires_grad) == 10_000_000


def test_forward_dispatch_uses_restarts_only_for_fixed_arm_and_loops_for_recurrent() -> None:
    calls: list[tuple[str, int]] = []

    class FixedModel:
        def __call__(self, tokens: torch.Tensor, *, restarts: int) -> torch.Tensor:
            calls.append(("restarts", restarts))
            return torch.zeros((*tokens.shape, 4_608))

    class RecurrentModel:
        def __call__(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
            calls.append(("loops", loops))
            return torch.zeros((*tokens.shape, 4_608))

    def compiled(arm: str, model: object):
        return type("Compiled", (), {"arm_id": type("Arm", (), {"value": arm})(), "model": model})()

    tokens = torch.zeros((1, 3), dtype=torch.long)
    assert forward_scientific_arm(compiled("A_FIXED", FixedModel()), tokens, effort=4).shape == (1, 3, 4_608)
    assert forward_scientific_arm(compiled("B_LOOP_SIMPLE", RecurrentModel()), tokens, effort=4).shape == (1, 3, 4_608)
    assert forward_scientific_arm(compiled("C_NRS_CORE", RecurrentModel()), tokens, effort=4).shape == (1, 3, 4_608)
    assert calls == [("restarts", 4), ("loops", 4), ("loops", 4)]


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


def test_family_balanced_score_is_not_pooled_by_family_size() -> None:
    score, families = family_balanced_development_score(
        {
            "family-a": (1, 1),
            "family-b": (0, 9),
        }
    )
    assert families == (("family-a", 1.0), ("family-b", 0.0))
    assert score == pytest.approx(0.5)


def test_scientific_train_step_uses_frozen_effort_schedule_and_updates_model() -> None:
    class ToyCompiled:
        arm_id = type("Arm", (), {"value": "C_NRS_CORE"})()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(4_608))

            def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
                assert loops == 1
                return self.bias.view(1, 1, -1).expand(tokens.shape[0], tokens.shape[1], -1)

        model = Model()

    compiled = ToyCompiled()
    optimizer = torch.optim.AdamW(compiled.model.parameters(), lr=1e-4, weight_decay=0.01)
    world = generate_world_instance(
        family="iterative-grid-and-maze",
        root=0,
        split="train",
        index=0,
    )
    before = compiled.model.bias.detach().clone()
    loss = scientific_train_step(compiled, world, optimizer=optimizer, step=0)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, compiled.model.bias.detach())


def test_public_scientific_trial_has_no_tuning_override_surface() -> None:
    parameters = set(inspect.signature(run_scientific_trial).parameters)
    assert {"plan", "device", "checkpoint_path"} <= parameters
    for forbidden in ("max_steps", "sample_count", "learning_rate", "loops", "threshold", "task_weight"):
        assert forbidden not in parameters


def test_greedy_generation_never_requires_canonical_answer() -> None:
    class StubCompiled:
        arm_id = type("Arm", (), {"value": "C_NRS_CORE"})()

        class Model(torch.nn.Module):
            def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
                logits = torch.full((*tokens.shape, 4_608), -1000.0, device=tokens.device)
                next_id = 3 if int(tokens[0, -1]) == 4 + ord("7") else 4 + ord("7")
                logits[:, -1, next_id] = 1000.0
                return logits

        model = Model()

    result = greedy_generate(StubCompiled(), prompt="Q", effort=1, max_new_tokens=8)
    assert result.candidate_answer == "7"
    assert result.generated_token_count == 2


def test_scientific_challenge_generation_runs_full_96_decode_budget_after_early_eos() -> None:
    class StubCompiled:
        arm_id = type("Arm", (), {"value": "C_NRS_CORE"})()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.forward_calls = 0

            def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
                self.forward_calls += 1
                logits = torch.full((*tokens.shape, 4_608), -1000.0, device=tokens.device)
                if self.forward_calls == 1:
                    next_id = 4 + ord("7")
                elif self.forward_calls == 2:
                    next_id = 3
                else:
                    next_id = 4 + ord("x")
                logits[:, -1, next_id] = 1000.0 + self.anchor
                return logits

        model = Model()

    compiled = StubCompiled()
    result = scientific_challenge_generate(compiled, prompt="Q", effort=1)
    assert result.candidate_answer == "7"
    assert result.stopped_on_eos is True
    assert result.generated_token_count == 96
    assert compiled.model.forward_calls == 96


def test_model_state_digest_changes_when_parameter_changes() -> None:
    compiled = build_scientific_arm("C_NRS_CORE", device="cpu")
    before = model_state_digest(compiled.model)
    with torch.no_grad():
        parameter = next(compiled.model.parameters())
        parameter.view(-1)[0].add_(1.0)
    after = model_state_digest(compiled.model)
    assert before != after
