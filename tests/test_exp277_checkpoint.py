from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


def _development(tmp_path):
    from nolane_ai.experiments.exp277_paired_runner import run_exp277_paired_development

    return run_exp277_paired_development(
        root_seed="exp277-checkpoint-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=3,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )


def test_exp277_checkpoint_replays_training_and_matches_final_functional_digests(tmp_path) -> None:
    from nolane_ai.experiments.exp277_checkpoint import (
        build_exp277_trained_checkpoint,
        load_exp277_trained_checkpoint,
        validate_exp277_checkpoint_receipt,
    )
    from nolane_ai.experiments.exp277_paired_runner import _functional_state_digest

    execution = _development(tmp_path)
    path = tmp_path / "exp277-checkpoint.pt"
    receipt = build_exp277_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
    assert receipt["schema"] == "NLM-EXP-277-TRAINED-CHECKPOINT-V1"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["state_policy"] == "functional-only"
    assert receipt["training_replay_verified"] is True
    assert receipt["arcs_branch_final_digest"] == execution["final_state"]["arcs_branch_digest"]
    assert receipt["oracle_cbrf_final_digest"] == execution["final_state"]["oracle_cbrf_digest"]
    assert receipt["scientific_identity_digest"]
    assert receipt["checkpoint_file_sha256"]
    assert validate_exp277_checkpoint_receipt(receipt) == []

    arcs, oracle = load_exp277_trained_checkpoint(checkpoint_path=path, receipt=receipt)
    assert _functional_state_digest(arcs) == receipt["arcs_branch_final_digest"]
    assert _functional_state_digest(oracle) == receipt["oracle_cbrf_final_digest"]


def test_exp277_checkpoint_payload_contains_functional_state_only(tmp_path) -> None:
    import torch

    from nolane_ai.experiments.exp277_checkpoint import build_exp277_trained_checkpoint

    execution = _development(tmp_path)
    path = tmp_path / "exp277-checkpoint.pt"
    build_exp277_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "NLM-EXP-277-TRAINED-TENSORS-V1"
    assert set(payload) == {
        "schema",
        "state_policy",
        "execution_contract",
        "execution_contract_digest",
        "arcs_branch_state",
        "oracle_cbrf_state",
    }
    assert "optimizer" not in payload
    assert "optimizer_state" not in payload


def test_exp277_checkpoint_rejects_development_final_state_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp277_checkpoint import build_exp277_trained_checkpoint

    execution = _development(tmp_path)
    execution["final_state"]["arcs_branch_digest"] = "x" * 64
    with pytest.raises(RuntimeError, match="final functional digest"):
        build_exp277_trained_checkpoint(
            execution_artifact=execution,
            checkpoint_path=tmp_path / "bad.pt",
        )


def test_exp277_checkpoint_loader_rejects_tensor_tamper_even_with_receipt_unchanged(tmp_path) -> None:
    import torch

    from nolane_ai.experiments.exp277_checkpoint import (
        build_exp277_trained_checkpoint,
        load_exp277_trained_checkpoint,
    )

    execution = _development(tmp_path)
    path = tmp_path / "exp277-checkpoint.pt"
    receipt = build_exp277_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(payload["arcs_branch_state"]))
    payload["arcs_branch_state"][key] = payload["arcs_branch_state"][key].clone()
    payload["arcs_branch_state"][key].view(-1)[0] += 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint file SHA256|functional digest"):
        load_exp277_trained_checkpoint(checkpoint_path=path, receipt=receipt)


def test_exp277_checkpoint_receipt_rejects_rehashed_contract_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp277_checkpoint import (
        _receipt_digest,
        build_exp277_trained_checkpoint,
        validate_exp277_checkpoint_receipt,
    )

    receipt = build_exp277_trained_checkpoint(
        execution_artifact=_development(tmp_path),
        checkpoint_path=tmp_path / "exp277-checkpoint.pt",
    )
    bad = deepcopy(receipt)
    bad["execution_contract"]["train_replicates"] = 999
    bad["receipt_digest"] = _receipt_digest(bad)
    errors = validate_exp277_checkpoint_receipt(bad)
    assert "EXP-277 checkpoint execution contract digest mismatch" in errors
    assert "EXP-277 checkpoint training lineage drift" in errors
