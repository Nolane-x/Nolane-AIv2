from __future__ import annotations

from pathlib import Path


def test_exp321_measurement_surface_is_gradient_free() -> None:
    source = Path("src/nolane_ai/experiments/exp321_measure.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        ".backward(",
        "optimizer.step(",
        "torch.optim.",
        "run_training_chunk(",
        "scientific_train_step(",
    )
    for token in forbidden:
        assert token not in source


def test_exp321_runner_writes_once_and_does_not_train() -> None:
    source = Path("scripts/exp321_localize_checkpoint.py").read_text(
        encoding="utf-8"
    )
    assert 'open("xb")' in source
    assert "run_training_chunk" not in source
    assert "optimizer" not in source
