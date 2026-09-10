from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


def _execution():
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development

    return run_exp286_paired_development(
        root_seed="exp286-checkpoint-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=2,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=5,
        decoys=2,
        max_search_steps=8,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )


def _checkpoint_api():
    try:
        from nolane_ai.experiments.exp286_checkpoint import (
            build_exp286_trained_checkpoint,
            load_exp286_trained_checkpoint,
            validate_exp286_checkpoint_receipt,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 trained-state checkpoint authority is missing")
    return (
        build_exp286_trained_checkpoint,
        load_exp286_trained_checkpoint,
        validate_exp286_checkpoint_receipt,
    )


def test_exp286_development_freezes_optimizer_losses_and_final_functional_state() -> None:
    execution = _execution()
    training = execution["training"]
    assert training["optimizer"] == {
        "type": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    assert len(training["chronological_failure_losses"]) == 2
    assert len(training["oracle_conflict_core_losses"]) == 2
    assert training["mean_losses"]["chronological_failure"] == pytest.approx(
        sum(training["chronological_failure_losses"]) / 2
    )
    assert training["mean_losses"]["oracle_conflict_core"] == pytest.approx(
        sum(training["oracle_conflict_core_losses"]) / 2
    )

    final_state = execution["final_state"]
    assert set(final_state) == {
        "chronological_failure_digest",
        "oracle_conflict_core_digest",
    }
    for digest in final_state.values():
        assert isinstance(digest, str) and len(digest) == 64
        int(digest, 16)


def test_exp286_checkpoint_replays_and_loads_exact_frozen_functional_state(tmp_path) -> None:
    build, load, validate = _checkpoint_api()
    from nolane_ai.experiments.exp286_paired_runner import _functional_state_digest

    execution = _execution()
    checkpoint_path = tmp_path / "exp286-trained.pt"
    receipt = build(
        execution_artifact=execution,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.is_file()
    assert receipt["schema"] == "NLM-EXP-286-TRAINED-CHECKPOINT-V1"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["state_policy"] == "functional-only"
    assert receipt["training_replay_verified"] is True
    assert receipt["confirmatory_data_consumed"] is False
    assert receipt["challenge_materialized"] is False
    assert validate(receipt) == []

    chronological, oracle = load(
        checkpoint_path=checkpoint_path,
        receipt=receipt,
    )
    assert _functional_state_digest(chronological) == receipt["chronological_failure_final_digest"]
    assert _functional_state_digest(oracle) == receipt["oracle_conflict_core_final_digest"]
    assert receipt["chronological_failure_final_digest"] == execution["final_state"][
        "chronological_failure_digest"
    ]
    assert receipt["oracle_conflict_core_final_digest"] == execution["final_state"][
        "oracle_conflict_core_digest"
    ]


def test_exp286_checkpoint_receipt_rejects_rehashed_scientific_identity_tampering(tmp_path) -> None:
    build, _, validate = _checkpoint_api()
    from nolane_ai.protocol.evidence import canonical_sha256

    execution = _execution()
    checkpoint_path = tmp_path / "exp286-trained.pt"
    receipt = build(
        execution_artifact=execution,
        checkpoint_path=checkpoint_path,
    )
    changed = deepcopy(receipt)
    changed["scientific_identity_digest"] = "0" * 64
    clean = deepcopy(changed)
    clean.pop("receipt_digest", None)
    changed["receipt_digest"] = canonical_sha256(clean)

    errors = validate(changed)
    assert any("scientific identity" in item.lower() for item in errors)
