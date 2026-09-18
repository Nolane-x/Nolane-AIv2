from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp301_ceremony import (
    ChallengeMaterialization,
    build_arm_selection_receipt,
    build_root_selection_manifest,
)
from nolane_ai.experiments.exp301_identity import EXP301_PREREG_V2_DIGEST
from nolane_ai.experiments.exp301_scientific import ScientificTrialResult
from nolane_ai.experiments.exp301_scientific_executor import run_scientific_root


def _trial(arm: str, root: int, lr: float, marker: str) -> ScientificTrialResult:
    return ScientificTrialResult(
        arm_id=arm,
        root=root,
        learning_rate=lr,
        model_init_seed=123,
        training_steps=512,
        development_family_balanced_score=0.7 if lr == 1e-4 else 0.6,
        development_family_scores=(("f", 0.7 if lr == 1e-4 else 0.6),),
        checkpoint_digest=marker * 64,
        trial_receipt_digest=("a" if marker != "a" else "b") * 64,
    )


def _selection(root: int):
    receipts = []
    markers = iter("cdefab")
    for arm in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"):
        receipts.append(
            build_arm_selection_receipt(
                (
                    _trial(arm, root, 1e-4, next(markers)),
                    _trial(arm, root, 3e-4, next(markers)),
                )
            )
        )
    return build_root_selection_manifest(receipts)


def test_public_full_root_executor_has_no_scientific_tuning_surface() -> None:
    parameters = set(inspect.signature(run_scientific_root).parameters)
    assert {
        "root",
        "device",
        "output_dir",
        "frozen_implementation_identity",
        "challenge_beacon",
    } <= parameters
    for forbidden in (
        "learning_rate",
        "max_steps",
        "sample_count",
        "bootstrap_samples",
        "loops",
        "threshold",
        "task_weight",
        "generation_tokens",
    ):
        assert forbidden not in parameters


def test_full_root_executor_enforces_pre_challenge_order_and_writes_evidence(monkeypatch, tmp_path: Path) -> None:
    import nolane_ai.experiments.exp301_scientific_executor as executor

    calls: list[str] = []
    root = 2
    selection = _selection(root)
    challenge = ChallengeMaterialization(
        schema="EXP301-ROOT-CHALLENGE-MATERIALIZATION-V1",
        root=root,
        challenge_nonce="1" * 64,
        worlds=(),
        materialization_digest="2" * 64,
    )
    runtime = SimpleNamespace(run_identity="3" * 64)
    artifact = SimpleNamespace(root=root, artifact_digest="4" * 64)
    frozen = SimpleNamespace(
        prereg_semantic_digest=EXP301_PREREG_V2_DIGEST,
        scientific_execution_contract_digest="5" * 64,
        frozen_implementation_digest="6" * 64,
    )

    monkeypatch.setattr(executor, "scientific_execution_contract_digest", lambda: "5" * 64)
    monkeypatch.setattr(
        executor,
        "run_root_training_selection",
        lambda **_kwargs: calls.append("train-select") or selection,
    )
    monkeypatch.setattr(
        executor,
        "write_root_selection_manifest",
        lambda _path, _selection: calls.append("persist-selection"),
    )
    monkeypatch.setattr(
        executor,
        "load_selected_models",
        lambda *_args, **_kwargs: calls.append("load-selected") or {arm: object() for arm in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")},
    )
    monkeypatch.setattr(
        executor,
        "materialize_root_challenge",
        lambda **_kwargs: calls.append("materialize-challenge") or challenge,
    )
    monkeypatch.setattr(
        executor,
        "build_runtime_identity_for_root",
        lambda **_kwargs: calls.append("runtime-identity") or runtime,
    )
    monkeypatch.setattr(
        executor,
        "commit_challenge_predictions",
        lambda *_args, **_kwargs: calls.append("commit-all") or ("commitment",),
    )
    monkeypatch.setattr(
        executor,
        "score_challenge_commitments",
        lambda *_args, **_kwargs: calls.append("verify-all") or ("row",),
    )
    monkeypatch.setattr(
        executor,
        "build_root_evidence_artifact",
        lambda **_kwargs: calls.append("build-evidence") or artifact,
    )
    written: list[Path] = []
    monkeypatch.setattr(
        executor,
        "write_root_evidence_artifact",
        lambda path, _artifact: calls.append("write-evidence") or written.append(Path(path)),
    )

    result = run_scientific_root(
        root=root,
        device="cpu",
        output_dir=tmp_path,
        frozen_implementation_identity=frozen,
        challenge_beacon="workflow-run-123",
    )
    assert result is artifact
    assert calls == [
        "train-select",
        "persist-selection",
        "load-selected",
        "materialize-challenge",
        "runtime-identity",
        "commit-all",
        "verify-all",
        "build-evidence",
        "write-evidence",
    ]
    assert written == [tmp_path / "root-2" / "root-evidence.json"]


def test_full_root_executor_rejects_frozen_execution_contract_drift_before_training(monkeypatch, tmp_path: Path) -> None:
    import nolane_ai.experiments.exp301_scientific_executor as executor

    frozen = SimpleNamespace(
        prereg_semantic_digest=EXP301_PREREG_V2_DIGEST,
        scientific_execution_contract_digest="5" * 64,
        frozen_implementation_digest="6" * 64,
    )
    monkeypatch.setattr(executor, "scientific_execution_contract_digest", lambda: "7" * 64)
    monkeypatch.setattr(
        executor,
        "run_root_training_selection",
        lambda **_kwargs: pytest.fail("training must not start after frozen contract drift"),
    )
    with pytest.raises(ValueError, match="execution contract"):
        run_scientific_root(
            root=0,
            device="cpu",
            output_dir=tmp_path,
            frozen_implementation_identity=frozen,
            challenge_beacon="workflow-run-123",
        )


def test_full_root_executor_rejects_prereg_drift_before_training(monkeypatch, tmp_path: Path) -> None:
    import nolane_ai.experiments.exp301_scientific_executor as executor

    frozen = SimpleNamespace(
        prereg_semantic_digest="0" * 64,
        scientific_execution_contract_digest="5" * 64,
        frozen_implementation_digest="6" * 64,
    )
    monkeypatch.setattr(executor, "scientific_execution_contract_digest", lambda: "5" * 64)
    monkeypatch.setattr(
        executor,
        "run_root_training_selection",
        lambda **_kwargs: pytest.fail("training must not start after prereg drift"),
    )
    with pytest.raises(ValueError, match="prereg"):
        run_scientific_root(
            root=0,
            device="cpu",
            output_dir=tmp_path,
            frozen_implementation_identity=frozen,
            challenge_beacon="workflow-run-123",
        )
