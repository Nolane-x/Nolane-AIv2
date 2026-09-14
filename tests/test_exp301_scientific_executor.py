from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_ceremony import (
    ChallengeMaterialization,
    build_arm_selection_receipt,
    build_root_selection_manifest,
)
from nolane_ai.experiments.exp301_scientific import GenerationResult, ScientificTrialResult, frozen_trial_plan
from nolane_ai.experiments.exp301_scientific_executor import (
    checkpoint_path_for_plan,
    commit_challenge_predictions,
    execute_root_training_selection,
    run_root_training_selection,
    score_challenge_commitments,
)
from nolane_ai.experiments.exp301_worlds import generate_world_instance


def _trial(arm: str, root: int, lr: float, score: float, marker: str) -> ScientificTrialResult:
    return ScientificTrialResult(
        arm_id=arm,
        root=root,
        learning_rate=lr,
        model_init_seed=123,
        training_steps=512,
        development_family_balanced_score=score,
        development_family_scores=(("f", score),),
        checkpoint_digest=marker * 64,
        trial_receipt_digest=("a" if marker != "a" else "b") * 64,
    )


def test_checkpoint_path_is_deterministic_and_contains_no_tuning_surface(tmp_path: Path) -> None:
    plan = frozen_trial_plan(root=2)[0]
    path = checkpoint_path_for_plan(tmp_path, plan)
    assert path.parent == tmp_path / "root-2" / plan.arm_id
    assert path.name.startswith("trial-0-lr-")
    assert path.suffix == ".pt"


def test_public_root_training_selection_exposes_no_scientific_tuning_flags() -> None:
    parameters = set(inspect.signature(run_root_training_selection).parameters)
    assert {"root", "device", "output_dir"} <= parameters
    for forbidden in ("learning_rate", "max_steps", "sample_count", "loops", "threshold", "task_weight"):
        assert forbidden not in parameters


def test_training_selection_executes_exactly_six_frozen_trials_and_selects_three_arms(tmp_path: Path) -> None:
    calls = []
    markers = iter("cdefab")

    def fake_trial_runner(plan, *, device, checkpoint_path):
        calls.append((plan.arm_id, plan.learning_rate, plan.root, Path(checkpoint_path)))
        marker = next(markers)
        return _trial(plan.arm_id, plan.root, plan.learning_rate, 0.7 if plan.learning_rate == 1e-4 else 0.6, marker)

    manifest = execute_root_training_selection(
        root=1,
        device="cpu",
        output_dir=tmp_path,
        trial_runner=fake_trial_runner,
    )
    assert len(calls) == 6
    assert manifest.root == 1
    assert tuple(item.arm_id for item in manifest.arm_selections) == (
        "A_FIXED",
        "B_LOOP_SIMPLE",
        "C_NRS_CORE",
    )
    assert all(item.selected_learning_rate == 1e-4 for item in manifest.arm_selections)


def test_commit_phase_covers_all_arms_and_efforts_without_verifier() -> None:
    world = generate_world_instance(
        family="iterative-grid-and-maze",
        root=0,
        split="challenge",
        index=0,
        challenge_nonce="x" * 64,
    )
    challenge = ChallengeMaterialization(
        schema="EXP301-ROOT-CHALLENGE-MATERIALIZATION-V1",
        root=0,
        challenge_nonce="x" * 64,
        worlds=(world,),
        materialization_digest="1" * 64,
    )
    selected_models = {arm: object() for arm in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")}
    calls = []

    def predictor(model, *, prompt: str, effort: int):
        calls.append((model, prompt, effort))
        return GenerationResult(candidate_answer="0", generated_token_count=2, stopped_on_eos=True)

    commitments = commit_challenge_predictions(
        challenge,
        selected_models=selected_models,
        predictor=predictor,
    )
    assert len(commitments) == 18
    assert len(calls) == 18
    assert {item.arm_id for item in commitments} == {"A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"}
    assert {item.effort_multiplier for item in commitments} == {1, 2, 4, 8, 12, 16}
    assert all(item.content_id == world.content_id for item in commitments)


def test_default_commit_phase_executes_fixed_96_decode_budget_for_every_arm_and_effort() -> None:
    world = generate_world_instance(
        family="algorithmic-sequence-transform",
        root=0,
        split="challenge",
        index=1,
        challenge_nonce="z" * 64,
    )
    challenge = ChallengeMaterialization(
        schema="EXP301-ROOT-CHALLENGE-MATERIALIZATION-V1",
        root=0,
        challenge_nonce="z" * 64,
        worlds=(world,),
        materialization_digest="3" * 64,
    )

    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.forward_calls = 0

        def forward(self, tokens: torch.Tensor, *, restarts: int) -> torch.Tensor:
            self.forward_calls += 1
            logits = torch.full((*tokens.shape, 4_608), -1000.0, device=tokens.device)
            logits[:, -1, 3] = 1000.0 + self.anchor
            return logits

    class RecurrentModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.forward_calls = 0

        def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
            self.forward_calls += 1
            logits = torch.full((*tokens.shape, 4_608), -1000.0, device=tokens.device)
            logits[:, -1, 3] = 1000.0 + self.anchor
            return logits

    def compiled(arm: str, model: torch.nn.Module):
        return type("Compiled", (), {"arm_id": type("Arm", (), {"value": arm})(), "model": model})()

    selected_models = {
        "A_FIXED": compiled("A_FIXED", FixedModel()),
        "B_LOOP_SIMPLE": compiled("B_LOOP_SIMPLE", RecurrentModel()),
        "C_NRS_CORE": compiled("C_NRS_CORE", RecurrentModel()),
    }
    commitments = commit_challenge_predictions(challenge, selected_models=selected_models)

    assert len(commitments) == 18
    assert {item.generation_token_count for item in commitments} == {96}
    assert all(item.compute_match_status == "VALID_COMPUTE_MATCH" for item in commitments)
    for compiled_model in selected_models.values():
        assert compiled_model.model.forward_calls == 6 * 96


def test_score_phase_requires_complete_commitment_grid() -> None:
    world = generate_world_instance(
        family="language-sequence-control",
        root=3,
        split="challenge",
        index=0,
        challenge_nonce="y" * 64,
    )
    challenge = ChallengeMaterialization(
        schema="EXP301-ROOT-CHALLENGE-MATERIALIZATION-V1",
        root=3,
        challenge_nonce="y" * 64,
        worlds=(world,),
        materialization_digest="2" * 64,
    )
    models = {arm: object() for arm in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")}

    def predictor(model, *, prompt: str, effort: int):
        return GenerationResult(candidate_answer="wrong", generated_token_count=1, stopped_on_eos=True)

    commitments = commit_challenge_predictions(challenge, selected_models=models, predictor=predictor)
    rows = score_challenge_commitments(challenge, commitments)
    assert len(rows) == 18
    assert not any(row.verified_success for row in rows)

    with pytest.raises(ValueError, match="complete commitment grid"):
        score_challenge_commitments(challenge, commitments[:-1])
