from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_ceremony import (
    build_arm_selection_receipt,
    build_root_selection_manifest,
)
from nolane_ai.experiments.exp301_scientific import (
    ScientificTrialResult,
    frozen_trial_plan,
    model_state_digest,
    trial_result_digest_payload,
)
from nolane_ai.experiments.exp301_scientific_executor import (
    checkpoint_path_for_plan,
    load_selected_models,
)


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(3))


class ToyCompiled:
    def __init__(self, arm_id: str) -> None:
        self.arm_id = type("Arm", (), {"value": arm_id})()
        self.model = ToyModel()


def _builder(arm_id: str, *, device: str):
    compiled = ToyCompiled(arm_id)
    compiled.model.to(device)
    return compiled


def _trial(
    arm: str,
    *,
    root: int,
    lr: float,
    score: float,
    checkpoint_digest: str,
) -> ScientificTrialResult:
    trial_index = (1e-4, 3e-4).index(lr)
    plan = next(
        item
        for item in frozen_trial_plan(root=root)
        if item.arm_id == arm and item.trial_index == trial_index
    )
    provisional = ScientificTrialResult(
        arm_id=arm,
        root=root,
        learning_rate=lr,
        model_init_seed=plan.model_init_seed,
        training_steps=512,
        development_family_balanced_score=score,
        development_family_scores=(("family", score),),
        checkpoint_digest=checkpoint_digest,
        trial_receipt_digest="",
    )
    return ScientificTrialResult(
        **{
            **asdict(provisional),
            "trial_receipt_digest": trial_result_digest_payload(provisional),
        }
    )


def _materialize_selected_checkpoints(tmp_path: Path, *, root: int = 1):
    receipts = []
    selected_results = {}
    for index, arm in enumerate(ARMS):
        source = ToyCompiled(arm)
        with torch.no_grad():
            source.model.anchor.fill_(float(index + 1))
        digest = model_state_digest(source.model)
        low = _trial(
            arm,
            root=root,
            lr=1e-4,
            score=0.8,
            checkpoint_digest=digest,
        )
        high = _trial(
            arm,
            root=root,
            lr=3e-4,
            score=0.7,
            checkpoint_digest=(chr(ord("4") + index) * 64),
        )
        receipts.append(build_arm_selection_receipt((low, high)))
        selected_results[arm] = low

        plan = next(
            item
            for item in frozen_trial_plan(root=root)
            if item.arm_id == arm and item.learning_rate == low.learning_rate
        )
        path = checkpoint_path_for_plan(tmp_path, plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "EXP301-SCIENTIFIC-CHECKPOINT-V1",
                "trial_result": asdict(low),
                "state_dict": source.model.state_dict(),
            },
            path,
        )

    return build_root_selection_manifest(receipts), selected_results


def test_selected_checkpoint_loader_binds_manifest_receipt_and_model_bytes(tmp_path: Path) -> None:
    manifest, selected = _materialize_selected_checkpoints(tmp_path)
    loaded = load_selected_models(
        manifest,
        output_dir=tmp_path,
        device="cpu",
        arm_builder=_builder,
    )

    assert tuple(loaded) == ARMS
    for arm in ARMS:
        assert model_state_digest(loaded[arm].model) == selected[arm].checkpoint_digest


def test_selected_checkpoint_loader_rejects_tampered_state_bytes(tmp_path: Path) -> None:
    manifest, selected = _materialize_selected_checkpoints(tmp_path)
    arm = "C_NRS_CORE"
    plan = next(
        item
        for item in frozen_trial_plan(root=manifest.root)
        if item.arm_id == arm and item.learning_rate == selected[arm].learning_rate
    )
    path = checkpoint_path_for_plan(tmp_path, plan)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state_dict"]["anchor"] = payload["state_dict"]["anchor"] + 1.0
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint digest"):
        load_selected_models(
            manifest,
            output_dir=tmp_path,
            device="cpu",
            arm_builder=_builder,
        )


def test_selected_checkpoint_loader_rejects_receipt_mismatch(tmp_path: Path) -> None:
    manifest, selected = _materialize_selected_checkpoints(tmp_path)
    arm = "A_FIXED"
    plan = next(
        item
        for item in frozen_trial_plan(root=manifest.root)
        if item.arm_id == arm and item.learning_rate == selected[arm].learning_rate
    )
    path = checkpoint_path_for_plan(tmp_path, plan)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["trial_result"]["trial_receipt_digest"] = "f" * 64
    torch.save(payload, path)

    with pytest.raises(ValueError, match="trial receipt"):
        load_selected_models(
            manifest,
            output_dir=tmp_path,
            device="cpu",
            arm_builder=_builder,
        )
