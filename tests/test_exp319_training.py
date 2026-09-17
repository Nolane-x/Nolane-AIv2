from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn


def _training_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_training")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 resumable training chains are not implemented yet: {exc}")


def _digest(char: str) -> str:
    return char * 64


def _build_receipt(training, **overrides):
    values = {
        "stage": "A_SANITY",
        "run_identity": "github-run-123-exp319-v1",
        "source_commit_sha": "a" * 40,
        "arm_id": "A_FIXED",
        "root": 0,
        "learning_rate": 1e-4,
        "model_init_seed": training.model_init_seed(
            stage="A_SANITY", arm_id="A_FIXED", root=0
        ),
        "chunk_start_step": 0,
        "chunk_end_step": 256,
        "model_state_digest": _digest("1"),
        "optimizer_state_digest": _digest("2"),
        "rng_state_digest": _digest("3"),
        "parent_artifact_digest": None,
    }
    values.update(overrides)
    return training.build_training_receipt(**values)


def test_receipt_binds_full_chain_identity_and_roundtrips() -> None:
    training = _training_module()
    receipt = _build_receipt(training)

    assert receipt.schema == "EXP319-TRAINING-CHAIN-RECEIPT-V1"
    assert receipt.stage == "A_SANITY"
    assert receipt.run_identity == "github-run-123-exp319-v1"
    assert receipt.source_commit_sha == "a" * 40
    assert receipt.arm_id == "A_FIXED"
    assert receipt.root == 0
    assert receipt.learning_rate == pytest.approx(1e-4)
    assert receipt.cumulative_step == 256
    assert receipt.chunk_start_step == 0
    assert receipt.chunk_end_step == 256
    assert len(receipt.data_order_digest) == 64
    assert len(receipt.data_cursor_digest) == 64
    assert receipt.model_state_digest == _digest("1")
    assert receipt.optimizer_state_digest == _digest("2")
    assert receipt.rng_state_digest == _digest("3")
    assert receipt.parent_artifact_digest is None
    assert len(receipt.training_contract_digest) == 64
    assert len(receipt.artifact_digest) == 64
    assert receipt.snapshot_steps == (128,)

    payload = receipt.to_json_dict()
    restored = training.Exp319TrainingReceipt.from_json_dict(payload)
    assert restored == receipt
    training.validate_training_receipt(restored)


def test_continuation_accepts_exact_parent_and_rejects_splicing() -> None:
    training = _training_module()
    parent = _build_receipt(training)
    child = _build_receipt(
        training,
        chunk_start_step=256,
        chunk_end_step=512,
        model_state_digest=_digest("4"),
        optimizer_state_digest=_digest("5"),
        rng_state_digest=_digest("6"),
        parent_artifact_digest=parent.artifact_digest,
    )

    training.validate_continuation(parent, child)
    assert child.snapshot_steps == (512,)

    mutations = (
        replace(child, run_identity="github-run-999-exp319-v1"),
        replace(child, source_commit_sha="b" * 40),
        replace(child, arm_id="C_NRS_CORE"),
        replace(child, learning_rate=3e-4),
        replace(child, model_init_seed=child.model_init_seed + 1),
        replace(child, chunk_start_step=255),
        replace(child, parent_artifact_digest=_digest("f")),
        replace(child, data_order_digest=_digest("e")),
        replace(child, data_cursor_digest=_digest("d")),
    )
    for mutated in mutations:
        with pytest.raises(ValueError):
            training.validate_continuation(parent, mutated)


def test_chunk_limits_and_stage_b_cannot_consume_stage_a_weights() -> None:
    training = _training_module()

    with pytest.raises(ValueError, match="256"):
        _build_receipt(training, chunk_end_step=257)

    with pytest.raises(ValueError, match="512"):
        _build_receipt(
            training,
            stage="B_TRAIN",
            arm_id="A_FIXED",
            root=1,
            model_init_seed=training.model_init_seed(
                stage="B_TRAIN", arm_id="A_FIXED", root=1
            ),
            chunk_end_step=513,
        )

    stage_a = _build_receipt(training)
    stage_b = _build_receipt(
        training,
        stage="B_TRAIN",
        arm_id="A_FIXED",
        root=1,
        model_init_seed=training.model_init_seed(
            stage="B_TRAIN", arm_id="A_FIXED", root=1
        ),
        chunk_start_step=256,
        chunk_end_step=512,
        parent_artifact_digest=stage_a.artifact_digest,
    )
    with pytest.raises(ValueError, match="stage"):
        training.validate_continuation(stage_a, stage_b)


def test_data_order_and_cursor_digests_are_deterministic_and_prefix_bound() -> None:
    training = _training_module()

    order_a = training.data_order_digest(stage="A_SANITY", root=0)
    assert order_a == training.data_order_digest(stage="A_SANITY", root=0)
    assert order_a != training.data_order_digest(stage="B_TRAIN", root=1)

    cursor_32 = training.data_cursor_digest(
        stage="A_SANITY", root=0, cumulative_step=32
    )
    cursor_64 = training.data_cursor_digest(
        stage="A_SANITY", root=0, cumulative_step=64
    )
    assert cursor_32 != cursor_64
    assert cursor_64 == training.data_cursor_digest(
        stage="A_SANITY", root=0, cumulative_step=64
    )


def test_snapshot_schedule_is_frozen_to_preregistered_steps() -> None:
    training = _training_module()

    assert training.expected_snapshot_steps("A_SANITY", 0, 256) == (128,)
    assert training.expected_snapshot_steps("A_SANITY", 256, 512) == (512,)
    assert training.expected_snapshot_steps("A_SANITY", 768, 1024) == (1024,)
    assert training.expected_snapshot_steps("B_TRAIN", 0, 512) == (512,)
    assert training.expected_snapshot_steps("B_TRAIN", 512, 1024) == (1024,)
    assert training.expected_snapshot_steps("B_TRAIN", 1536, 2048) == (2048,)


def test_training_contract_freezes_optimizer_gradient_clip_and_effort_cycle() -> None:
    training = _training_module()

    contract_a = training.training_contract_payload("A_SANITY")
    contract_b = training.training_contract_payload("B_TRAIN")
    assert contract_a["optimizer"] == "AdamW"
    assert contract_a["weight_decay"] == pytest.approx(0.01)
    assert contract_a["gradient_clip"] == pytest.approx(1.0)
    assert contract_a["effort_cycle"] == [1, 2, 4, 8]
    assert contract_a["max_chunk_steps"] == 256
    assert contract_b["max_chunk_steps"] == 512
    assert training.training_contract_digest("A_SANITY") != training.training_contract_digest(
        "B_TRAIN"
    )


def test_checkpoint_bundle_is_write_once_and_detects_state_tampering(tmp_path: Path) -> None:
    training = _training_module()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    receipt = _build_receipt(
        training,
        model_state_digest=training.model_state_digest(model),
        optimizer_state_digest=training.optimizer_state_digest(optimizer),
        rng_state_digest=training.rng_state_digest(),
    )
    checkpoint_path = tmp_path / "chunk.pt"
    receipt_path = tmp_path / "chunk.json"

    training.write_checkpoint_bundle(
        checkpoint_path,
        receipt_path,
        model=model,
        optimizer=optimizer,
        receipt=receipt,
    )
    loaded = training.load_checkpoint_bundle(checkpoint_path, receipt_path)
    assert loaded.receipt == receipt

    with pytest.raises(FileExistsError):
        training.write_checkpoint_bundle(
            checkpoint_path,
            receipt_path,
            model=model,
            optimizer=optimizer,
            receipt=receipt,
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    first_key = next(iter(raw["model_state_dict"]))
    raw["model_state_dict"][first_key] = raw["model_state_dict"][first_key].clone()
    raw["model_state_dict"][first_key].view(-1)[0] += 1.0
    torch.save(raw, checkpoint_path)

    with pytest.raises(ValueError, match="model state digest"):
        training.load_checkpoint_bundle(checkpoint_path, receipt_path)


def test_malformed_receipt_json_is_rejected() -> None:
    training = _training_module()
    receipt = _build_receipt(training)
    payload = receipt.to_json_dict()
    payload["snapshot_steps"] = [128, 128]
    with pytest.raises(ValueError):
        training.Exp319TrainingReceipt.from_json_dict(payload)
