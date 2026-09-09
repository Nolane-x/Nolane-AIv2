from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


def _development() -> dict:
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development

    return run_exp279_paired_development(
        root_seed="exp279-checkpoint-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
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


def test_exp279_checkpoint_replays_training_and_matches_three_final_functional_digests(tmp_path) -> None:
    from nolane_ai.experiments.exp279_checkpoint import (
        build_exp279_trained_checkpoint,
        load_exp279_trained_checkpoint,
        validate_exp279_checkpoint_receipt,
    )
    from nolane_ai.experiments.exp279_paired_runner import _functional_state_digest

    execution = _development()
    path = tmp_path / "exp279-checkpoint.pt"
    receipt = build_exp279_trained_checkpoint(
        execution_artifact=execution,
        checkpoint_path=path,
    )

    assert receipt["schema"] == "NLM-EXP-279-TRAINED-CHECKPOINT-V1"
    assert receipt["experiment_id"] == "EXP-279"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["state_policy"] == "functional-only"
    assert receipt["training_replay_verified"] is True
    assert receipt["confirmatory_data_consumed"] is False
    assert receipt["challenge_materialized"] is False
    assert receipt["decision_rule_executed"] is False
    assert receipt["replay_contract_digest"] == execution["replay_contract_digest"]
    assert receipt["final_state_digest"] == execution["final_state_digest"]
    for arm_id in ("propagation_only", "branch_only", "hybrid"):
        assert receipt["final_state"][f"{arm_id}_digest"] == execution["final_state"][f"{arm_id}_digest"]
    assert receipt["checkpoint_file_sha256"]
    assert receipt["scientific_identity_digest"]
    assert validate_exp279_checkpoint_receipt(receipt) == []

    propagation, branch, hybrid = load_exp279_trained_checkpoint(
        checkpoint_path=path,
        receipt=receipt,
    )
    assert _functional_state_digest(propagation) == receipt["final_state"]["propagation_only_digest"]
    assert _functional_state_digest(branch) == receipt["final_state"]["branch_only_digest"]
    assert _functional_state_digest(hybrid) == receipt["final_state"]["hybrid_digest"]


def test_exp279_checkpoint_payload_is_functional_only_and_excludes_optimizer_and_evaluation(tmp_path) -> None:
    import torch

    from nolane_ai.experiments.exp279_checkpoint import build_exp279_trained_checkpoint

    execution = _development()
    path = tmp_path / "exp279-checkpoint.pt"
    build_exp279_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema"] == "NLM-EXP-279-TRAINED-TENSORS-V1"
    assert payload["state_policy"] == "functional-only"
    assert set(payload) == {
        "schema",
        "state_policy",
        "replay_contract",
        "replay_contract_digest",
        "propagation_only_state",
        "branch_only_state",
        "hybrid_state",
    }
    rendered = repr(payload).lower()
    assert "optimizer_state" not in rendered
    assert "evaluation" not in payload
    assert "evaluation" not in payload["replay_contract"]
    assert "per_replicate" not in payload["replay_contract"]
    assert "challenge" not in rendered


def test_exp279_checkpoint_rejects_development_final_state_tamper_even_after_top_level_rehash(tmp_path) -> None:
    from nolane_ai.experiments.exp279_checkpoint import build_exp279_trained_checkpoint
    from nolane_ai.experiments.exp279_paired_runner import _artifact_digest

    execution = _development()
    execution["final_state"]["hybrid_digest"] = "f" * 64
    execution["final_state_digest"] = "e" * 64
    execution["artifact_digest"] = _artifact_digest(execution)

    with pytest.raises((ValueError, RuntimeError), match="final state|functional digest|replay contract"):
        build_exp279_trained_checkpoint(
            execution_artifact=execution,
            checkpoint_path=tmp_path / "bad.pt",
        )


def test_exp279_checkpoint_loader_rejects_tensor_tamper_with_unchanged_receipt(tmp_path) -> None:
    import torch

    from nolane_ai.experiments.exp279_checkpoint import (
        build_exp279_trained_checkpoint,
        load_exp279_trained_checkpoint,
    )

    execution = _development()
    path = tmp_path / "exp279-checkpoint.pt"
    receipt = build_exp279_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(payload["hybrid_state"]))
    payload["hybrid_state"][key] = payload["hybrid_state"][key].clone()
    payload["hybrid_state"][key].view(-1)[0] += 1.0
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint file SHA256|functional digest"):
        load_exp279_trained_checkpoint(checkpoint_path=path, receipt=receipt)


def test_exp279_checkpoint_receipt_rejects_rehashed_replay_contract_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_checkpoint import (
        _receipt_digest,
        build_exp279_trained_checkpoint,
        validate_exp279_checkpoint_receipt,
    )

    receipt = build_exp279_trained_checkpoint(
        execution_artifact=_development(),
        checkpoint_path=tmp_path / "exp279-checkpoint.pt",
    )
    bad = deepcopy(receipt)
    bad["replay_contract"]["arm_geometry"]["route_threshold"] = 0.25
    bad["receipt_digest"] = _receipt_digest(bad)
    errors = validate_exp279_checkpoint_receipt(bad)
    assert any("replay contract" in error.lower() for error in errors)
    assert any("route threshold" in error.lower() or "scientific identity" in error.lower() for error in errors)


def test_exp279_checkpoint_creation_is_create_once(tmp_path) -> None:
    from nolane_ai.experiments.exp279_checkpoint import build_exp279_trained_checkpoint

    execution = _development()
    path = tmp_path / "exp279-checkpoint.pt"
    build_exp279_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
    with pytest.raises(FileExistsError, match="already exists"):
        build_exp279_trained_checkpoint(execution_artifact=execution, checkpoint_path=path)
