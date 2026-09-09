from __future__ import annotations

from copy import deepcopy
import math

import pytest

pytest.importorskip("torch")


RUN_KWARGS = dict(
    root_seed="exp279-replay-contract-test",
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


def _run() -> dict:
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development

    return run_exp279_paired_development(**RUN_KWARGS)


def test_exp279_development_artifact_carries_replayable_training_identity() -> None:
    from nolane_ai.experiments.exp279_paired_runner import validate_exp279_paired_development
    from nolane_ai.protocol.seeds import derive_stream_seed

    artifact = _run()
    assert artifact["root_seed"] == RUN_KWARGS["root_seed"]
    assert artifact["model_init_seed"] == derive_stream_seed(
        RUN_KWARGS["root_seed"], "EXP-279", 0, "model_init"
    )
    assert artifact["arm_geometry"] == {
        "d_model": 8,
        "hidden_size": 6,
        "target_parameters": 5_000,
        "route_threshold": 0.5,
    }
    assert artifact["world_geometry"] == {
        "batch_size": 2,
        "timesteps": 3,
        "variables": 4,
        "constraints": 2,
        "d_model": 8,
        "noise_std": 0.05,
    }

    training = artifact["training"]
    assert training["optimizer"] == {
        "type": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    losses = training["losses"]
    assert set(losses) == {"propagation_only", "branch_only", "hybrid"}
    for arm_id, values in losses.items():
        assert len(values) == 3
        assert all(math.isfinite(float(value)) for value in values)
        assert training["mean_losses"][arm_id] == pytest.approx(sum(values) / len(values))

    final_state = artifact["final_state"]
    assert set(final_state) == {
        "propagation_only_digest",
        "branch_only_digest",
        "hybrid_digest",
    }
    assert all(isinstance(value, str) and len(value) == 64 for value in final_state.values())
    assert artifact["final_state_digest"]
    assert artifact["replay_contract_digest"]
    assert validate_exp279_paired_development(artifact) == []


def test_exp279_validator_cross_checks_replay_identity_after_top_level_rehash() -> None:
    from nolane_ai.experiments.exp279_paired_runner import (
        _artifact_digest,
        validate_exp279_paired_development,
    )

    original = _run()

    bad = deepcopy(original)
    bad["root_seed"] = "different-root-seed"
    bad["artifact_digest"] = _artifact_digest(bad)
    assert any("model-init" in error or "root seed" in error for error in validate_exp279_paired_development(bad))

    bad = deepcopy(original)
    bad["arm_geometry"]["route_threshold"] = 0.25
    bad["artifact_digest"] = _artifact_digest(bad)
    assert any("route threshold" in error for error in validate_exp279_paired_development(bad))

    bad = deepcopy(original)
    bad["training"]["losses"]["hybrid"][0] += 1.0
    bad["artifact_digest"] = _artifact_digest(bad)
    assert any("training loss" in error for error in validate_exp279_paired_development(bad))

    bad = deepcopy(original)
    bad["final_state"]["hybrid_digest"] = "f" * 64
    bad["artifact_digest"] = _artifact_digest(bad)
    assert any("final state" in error for error in validate_exp279_paired_development(bad))
