from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


def _execution():
    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development

    return run_exp289_paired_development(
        root_seed="exp289-checkpoint-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=2,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        restarts=3,
        variables=6,
        decoys=2,
        max_search_steps=12,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )


def _checkpoint_api():
    try:
        from nolane_ai.experiments.exp289_checkpoint import (
            build_exp289_trained_checkpoint,
            load_exp289_trained_checkpoint,
            validate_exp289_checkpoint_receipt,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-289 trained-state checkpoint authority is missing")
    return build_exp289_trained_checkpoint, load_exp289_trained_checkpoint, validate_exp289_checkpoint_receipt


def test_exp289_checkpoint_replays_and_loads_exact_frozen_functional_state(tmp_path) -> None:
    build, load, validate = _checkpoint_api()
    from nolane_ai.experiments.exp289_paired_runner import _functional_state_digest

    execution = _execution()
    checkpoint_path = tmp_path / "exp289-trained.pt"
    receipt = build(execution_artifact=execution, checkpoint_path=checkpoint_path)

    assert checkpoint_path.is_file()
    assert receipt["schema"] == "NLM-EXP-289-TRAINED-CHECKPOINT-V1"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["state_policy"] == "functional-only"
    assert receipt["training_replay_verified"] is True
    assert receipt["confirmatory_data_consumed"] is False
    assert receipt["challenge_materialized"] is False
    assert receipt["challenge_seed_materialized"] is False
    assert validate(receipt) == []

    no_nogood, local_nogood = load(checkpoint_path=checkpoint_path, receipt=receipt)
    assert _functional_state_digest(no_nogood) == receipt["no_nogood_final_digest"]
    assert _functional_state_digest(local_nogood) == receipt["local_nogood_final_digest"]
    assert receipt["no_nogood_final_digest"] == execution["training"]["post_training_no_nogood_digest"]
    assert receipt["local_nogood_final_digest"] == execution["training"]["post_training_local_nogood_digest"]


def test_exp289_checkpoint_binds_exact_training_lineage(tmp_path) -> None:
    build, _, validate = _checkpoint_api()
    from nolane_ai.protocol.evidence import canonical_sha256

    execution = _execution()
    receipt = build(
        execution_artifact=execution,
        checkpoint_path=tmp_path / "exp289-trained.pt",
    )
    contract = receipt["execution_contract"]
    assert contract["training_lineage"]["paired_batch_digests"] == execution["training"]["paired_batch_digests"]
    assert contract["training_lineage"]["per_replicate"] == execution["training"]["per_replicate"]
    assert contract["training_lineage"]["optimizer_family"] == "AdamW"

    changed = deepcopy(receipt)
    changed["scientific_identity_digest"] = "0" * 64
    clean = deepcopy(changed)
    clean.pop("receipt_digest", None)
    changed["receipt_digest"] = canonical_sha256(clean)
    errors = validate(changed)
    assert any("scientific identity" in item.lower() for item in errors)


def test_exp289_checkpoint_rejects_confirmatory_or_challenge_contamination(tmp_path) -> None:
    build, _, _ = _checkpoint_api()
    execution = _execution()
    execution["confirmatory_data_consumed"] = True
    with pytest.raises(ValueError, match="confirmatory|challenge"):
        build(
            execution_artifact=execution,
            checkpoint_path=tmp_path / "forbidden.pt",
        )
