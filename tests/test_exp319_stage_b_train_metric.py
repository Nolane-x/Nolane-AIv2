from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments import exp319_training as training
from nolane_ai.experiments.exp319_worlds import materialize_stage_b


def test_snapshot_schema_carries_stage_b_training_exact_match() -> None:
    snapshot = training.SnapshotMetrics(
        step=512,
        answer_only_loss=1.0,
        answer_token_accuracy=0.5,
        exact_match=0.25,
        family_balanced_exact_match=0.25,
        family_exact=(("iterative-grid-and-maze", 0.25),),
        eos_correctness=0.5,
        invalid_output_rate=0.0,
        nonzero_exact_families=1,
        gradient_norm_preclip=1.0,
        parameter_update_norm_ratio=0.01,
        nonfinite_events=0,
        train_exact_match=0.75,
    )
    assert snapshot.train_exact_match == pytest.approx(0.75)
    assert snapshot.to_json_dict()["train_exact_match"] == pytest.approx(0.75)


def test_stage_b_training_exact_match_is_measured_on_training_worlds(monkeypatch) -> None:
    worlds = materialize_stage_b(root=1, split="train")[:4]
    answers = {world.model_input: world.canonical_answer for world in worlds}

    monkeypatch.setattr(
        training,
        "training_worlds",
        lambda *, stage, root: worlds if stage == "B_TRAIN" and root == 1 else (),
    )
    monkeypatch.setattr(
        training,
        "_diagnostic_generate",
        lambda compiled, *, prompt: (answers[prompt], (), True),
    )

    measured = training.measure_training_exact_match(object(), stage="B_TRAIN", root=1)
    assert measured == pytest.approx(1.0)
