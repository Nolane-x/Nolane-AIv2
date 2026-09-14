from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch

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
    build_scientific_arm,
    frozen_trial_plan,
    model_state_digest,
    run_scientific_trial,
    scientific_challenge_generate,
    trial_result_digest_payload,
    validate_trial_result,
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


def _load_checkpoint_payload(
    path: Path,
    *,
    arm_id: str,
    root: int,
    learning_rate: float,
    device: str,
    arm_builder: Callable[..., object],
) -> tuple[object, ScientificTrialResult]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "EXP301-SCIENTIFIC-CHECKPOINT-V1":
        raise ValueError("scientific checkpoint schema mismatch")
    raw_result = payload.get("trial_result")
    state_dict = payload.get("state_dict")
    if not isinstance(raw_result, dict) or not isinstance(state_dict, dict):
        raise ValueError("scientific checkpoint payload is incomplete")
    try:
        result = ScientificTrialResult(**raw_result)
    except TypeError as exc:
        raise ValueError("scientific checkpoint trial result schema mismatch") from exc
    validate_trial_result(result)
    if result.trial_receipt_digest != trial_result_digest_payload(result):
        raise ValueError("scientific checkpoint trial receipt digest mismatch")
    if (
        result.arm_id != arm_id
        or result.root != root
        or result.learning_rate != learning_rate
    ):
        raise ValueError("scientific checkpoint selection identity mismatch")

    compiled = arm_builder(arm_id, device=device)
    model = getattr(compiled, "model")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("scientific checkpoint state_dict mismatch") from exc
    actual_digest = model_state_digest(model)
    if actual_digest != result.checkpoint_digest:
        raise ValueError(
            "scientific checkpoint digest mismatch: "
            f"expected {result.checkpoint_digest}, got {actual_digest}"
        )
    return compiled, result


def load_selected_models(
    selection_manifest: RootSelectionManifest,
    *,
    output_dir: str | Path,
    device: str,
    arm_builder: Callable[..., object] = build_scientific_arm,
) -> dict[str, object]:
    rebuilt = build_root_selection_manifest(selection_manifest.arm_selections)
    if rebuilt != selection_manifest:
        raise ValueError("root selection manifest digest mismatch")

    output_dir = Path(output_dir)
    selected_by_arm = {item.arm_id: item for item in selection_manifest.arm_selections}
    loaded: dict[str, object] = {}
    for arm_id in EXP301_ARMS:
        selection = selected_by_arm[arm_id]
        plan = next(
            (
                item
                for item in frozen_trial_plan(root=selection_manifest.root)
                if item.arm_id == arm_id
                and item.learning_rate == selection.selected_learning_rate
            ),
            None,
        )
        if plan is None:
            raise ValueError("selected learning rate is outside frozen trial plan")
        compiled, result = _load_checkpoint_payload(
            checkpoint_path_for_plan(output_dir, plan),
            arm_id=arm_id,
            root=selection_manifest.root,
            learning_rate=selection.selected_learning_rate,
            device=device,
            arm_builder=arm_builder,
        )
        if result.model_init_seed != plan.model_init_seed or result.training_steps != 512:
            raise ValueError("selected checkpoint frozen training identity mismatch")
        if result.trial_receipt_digest != selection.selected_trial_receipt_digest:
            raise ValueError("selected checkpoint trial receipt mismatch")
        if result.checkpoint_digest != selection.selected_checkpoint_digest:
            raise ValueError("selected checkpoint digest does not match selection manifest")
        loaded[arm_id] = compiled
    return loaded


def commit_challenge_predictions(
    challenge: ChallengeMaterialization,
    *,
    selected_models: Mapping[str, object],
    predictor: Callable[..., GenerationResult] = scientific_challenge_generate,
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
