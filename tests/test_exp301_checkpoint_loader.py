from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_scientific import (
    ScientificTrialResult,
    build_scientific_arm,
    load_scientific_checkpoint,
    model_state_digest,
    trial_result_digest_payload,
)


def _result(*, arm: str, root: int, lr: float, checkpoint_digest: str) -> ScientificTrialResult:
    provisional = ScientificTrialResult(
        arm_id=arm,
        root=root,
        learning_rate=lr,
        model_init_seed=123,
        training_steps=512,
        development_family_balanced_score=0.5,
        development_family_scores=(("language-sequence-control", 0.5),),
        checkpoint_digest=checkpoint_digest,
        trial_receipt_digest="",
    )
    return ScientificTrialResult(
        **{
            **asdict(provisional),
            "trial_receipt_digest": trial_result_digest_payload(provisional),
        }
    )


def _write_checkpoint(path: Path, *, arm: str = "C_NRS_CORE", root: int = 0, lr: float = 1e-4):
    compiled = build_scientific_arm(arm, device="cpu")
    result = _result(
        arm=arm,
        root=root,
        lr=lr,
        checkpoint_digest=model_state_digest(compiled.model),
    )
    torch.save(
        {
            "schema": "EXP301-SCIENTIFIC-CHECKPOINT-V1",
            "trial_result": asdict(result),
            "state_dict": compiled.model.state_dict(),
        },
        path,
    )
    return result


def test_selected_checkpoint_loader_rebuilds_exact_arm_and_verifies_state_digest(tmp_path: Path) -> None:
    path = tmp_path / "selected.pt"
    result = _write_checkpoint(path)
    loaded, loaded_result = load_scientific_checkpoint(
        path,
        expected_arm_id="C_NRS_CORE",
        expected_root=0,
        expected_learning_rate=1e-4,
        device="cpu",
    )
    assert loaded_result == result
    assert model_state_digest(loaded.model) == result.checkpoint_digest
    assert sum(p.numel() for p in loaded.model.parameters() if p.requires_grad) == 10_000_000


def test_selected_checkpoint_loader_rejects_selection_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "selected.pt"
    _write_checkpoint(path)
    with pytest.raises(ValueError, match="selection identity"):
        load_scientific_checkpoint(
            path,
            expected_arm_id="C_NRS_CORE",
            expected_root=1,
            expected_learning_rate=1e-4,
            device="cpu",
        )


def test_selected_checkpoint_loader_rejects_state_or_receipt_tampering(tmp_path: Path) -> None:
    path = tmp_path / "selected.pt"
    _write_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["trial_result"]["development_family_balanced_score"] = 0.75
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trial receipt digest"):
        load_scientific_checkpoint(
            path,
            expected_arm_id="C_NRS_CORE",
            expected_root=0,
            expected_learning_rate=1e-4,
            device="cpu",
        )
