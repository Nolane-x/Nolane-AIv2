from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

from .exp301_ceremony import (
    ChallengeMaterialization,
    RootSelectionManifest,
    build_arm_selection_receipt,
    build_root_selection_manifest,
)
from .exp301_evaluation import (
    EXP301_ARMS,
    EXP301_EFFORTS,
    Exp301EvaluationRow,
    PredictionCommitment,
    commit_prediction,
    score_committed_prediction,
)
from .exp301_scientific import (
    GenerationResult,
    ScientificTrialPlan,
    ScientificTrialResult,
    frozen_trial_plan,
    greedy_generate,
    run_scientific_trial,
)


def checkpoint_path_for_plan(output_dir: str | Path, plan: ScientificTrialPlan) -> Path:
    root = Path(output_dir) / f"root-{plan.root}" / plan.arm_id
    lr_label = format(plan.learning_rate, ".0e").replace("+", "")
    return root / f"trial-{plan.trial_index}-lr-{lr_label}.pt"


def execute_root_training_selection(
    *,
    root: int,
    device: str,
    output_dir: str | Path,
    trial_runner: Callable[..., ScientificTrialResult],
) -> RootSelectionManifest:
    results_by_arm: dict[str, list[ScientificTrialResult]] = {arm: [] for arm in EXP301_ARMS}
    for plan in frozen_trial_plan(root=root):
        result = trial_runner(
            plan,
            device=device,
            checkpoint_path=checkpoint_path_for_plan(output_dir, plan),
        )
        results_by_arm[plan.arm_id].append(result)

    receipts = tuple(
        build_arm_selection_receipt(results_by_arm[arm])
        for arm in EXP301_ARMS
    )
    return build_root_selection_manifest(receipts)


def run_root_training_selection(
    *,
    root: int,
    device: str,
    output_dir: str | Path,
) -> RootSelectionManifest:
    return execute_root_training_selection(
        root=root,
        device=device,
        output_dir=output_dir,
        trial_runner=run_scientific_trial,
    )


def commit_challenge_predictions(
    challenge: ChallengeMaterialization,
    *,
    selected_models: Mapping[str, object],
    predictor: Callable[..., GenerationResult] = greedy_generate,
) -> tuple[PredictionCommitment, ...]:
    if set(selected_models) != set(EXP301_ARMS):
        raise ValueError(f"selected_models must contain exactly arms {EXP301_ARMS}")

    commitments: list[PredictionCommitment] = []
    for world in challenge.worlds:
        for arm_id in EXP301_ARMS:
            model = selected_models[arm_id]
            for effort in EXP301_EFFORTS:
                generation = predictor(model, prompt=world.model_input, effort=effort)
                commitments.append(
                    commit_prediction(
                        world,
                        arm_id=arm_id,
                        root=challenge.root,
                        effort_multiplier=effort,
                        candidate_answer=generation.candidate_answer,
                        generation_token_count=generation.generated_token_count,
                    )
                )
    return tuple(commitments)


def score_challenge_commitments(
    challenge: ChallengeMaterialization,
    commitments: Iterable[PredictionCommitment],
) -> tuple[Exp301EvaluationRow, ...]:
    materialized = tuple(commitments)
    expected_keys = {
        (world.content_id, arm_id, effort)
        for world in challenge.worlds
        for arm_id in EXP301_ARMS
        for effort in EXP301_EFFORTS
    }
    actual_keys = {
        (item.content_id, item.arm_id, item.effort_multiplier)
        for item in materialized
    }
    if len(materialized) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("complete commitment grid required before verifier scoring")

    worlds_by_id = {world.content_id: world for world in challenge.worlds}
    return tuple(
        score_committed_prediction(commitment, worlds_by_id[commitment.content_id])
        for commitment in materialized
    )
