from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn


def _metrics_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_metrics")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 metrics are not implemented yet: {exc}")


def _perfect_logits(targets: torch.Tensor, *, vocab_size: int = 16) -> torch.Tensor:
    logits = torch.full((*targets.shape, vocab_size), -8.0)
    for batch in range(targets.shape[0]):
        for time in range(targets.shape[1]):
            logits[batch, time, int(targets[batch, time])] = 8.0
    return logits


def test_answer_only_metrics_ignore_prompt_and_include_eos() -> None:
    metrics = _metrics_module()
    targets = torch.tensor([[7, 8, 9, 3]], dtype=torch.long)
    answer_start = 3
    logits = _perfect_logits(targets)

    baseline_loss = metrics.answer_only_cross_entropy(
        logits, targets=targets, answer_start=answer_start
    )
    assert baseline_loss.item() < 1e-4
    assert metrics.answer_token_accuracy(
        logits, targets=targets, answer_start=answer_start
    ) == pytest.approx(1.0)

    prompt_corrupted = logits.clone()
    prompt_corrupted[:, : answer_start - 1, :] = 0.0
    assert metrics.answer_only_cross_entropy(
        prompt_corrupted, targets=targets, answer_start=answer_start
    ).item() == pytest.approx(baseline_loss.item())

    eos_corrupted = logits.clone()
    eos_corrupted[0, -1, :] = -8.0
    eos_corrupted[0, -1, 4] = 8.0
    assert metrics.answer_token_accuracy(
        eos_corrupted, targets=targets, answer_start=answer_start
    ) == pytest.approx(0.5)
    assert metrics.answer_only_cross_entropy(
        eos_corrupted, targets=targets, answer_start=answer_start
    ).item() > baseline_loss.item()


def test_generation_quality_tracks_exact_eos_invalid_and_family_balance() -> None:
    metrics = _metrics_module()
    records = (
        metrics.GenerationMetricRecord(
            family="family-a",
            exact=True,
            stopped_on_eos=True,
            invalid_output=False,
        ),
        metrics.GenerationMetricRecord(
            family="family-a",
            exact=False,
            stopped_on_eos=True,
            invalid_output=True,
        ),
        metrics.GenerationMetricRecord(
            family="family-b",
            exact=True,
            stopped_on_eos=False,
            invalid_output=False,
        ),
    )
    summary = metrics.summarize_generation_records(records)

    assert summary.exact_match == pytest.approx(2 / 3)
    assert summary.family_exact == (("family-a", 0.5), ("family-b", 1.0))
    assert summary.family_balanced_exact_match == pytest.approx(0.75)
    assert summary.eos_correctness == pytest.approx(2 / 3)
    assert summary.invalid_output_rate == pytest.approx(1 / 3)
    assert summary.nonzero_exact_families == 2


def test_invalid_output_rate_detects_non_byte_control_before_eos() -> None:
    metrics = _metrics_module()
    sequences = (
        (4 + ord("a"), 4 + ord("b"), 3),
        (2, 3),
        (4 + ord("z"), 3),
        (999, 3),
    )
    assert metrics.invalid_output_rate(sequences) == pytest.approx(0.5)


def test_global_gradient_norm_is_preclip_l2_norm() -> None:
    metrics = _metrics_module()
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    assert metrics.global_gradient_norm((first, second)) == pytest.approx(13.0)


def test_parameter_update_norm_ratio_matches_concatenated_definition() -> None:
    metrics = _metrics_module()
    before = (torch.tensor([3.0, 4.0]), torch.tensor([0.0]))
    after = (torch.tensor([6.0, 8.0]), torch.tensor([0.0]))

    assert metrics.parameter_update_norm_ratio(before, after) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="geometry"):
        metrics.parameter_update_norm_ratio(before, (torch.tensor([1.0]),))


def test_nan_inf_detection_counts_parameters_gradients_and_loss() -> None:
    metrics = _metrics_module()
    parameter = nn.Parameter(torch.tensor([1.0, float("nan")]))
    parameter.grad = torch.tensor([float("inf"), 2.0])

    report = metrics.nonfinite_report((parameter,), loss=torch.tensor(float("inf")))
    assert report.parameter_nonfinite == 1
    assert report.gradient_nonfinite == 1
    assert report.loss_nonfinite == 1
    assert report.total == 3


def test_gating_generation_is_frozen_to_effort_4_and_96_token_cap() -> None:
    metrics = _metrics_module()

    class EosModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.seen_restarts: list[int] = []

        def forward(self, tokens: torch.Tensor, *, restarts: int) -> torch.Tensor:
            self.seen_restarts.append(restarts)
            logits = torch.full(
                (tokens.shape[0], tokens.shape[1], 4608),
                -100.0,
                device=tokens.device,
            )
            logits[:, -1, 3] = 100.0
            return logits

    model = EosModel()
    compiled = SimpleNamespace(arm_id="A_FIXED", model=model)
    result = metrics.gating_generate(compiled, prompt="tiny")

    assert model.seen_restarts == [4]
    assert result.generated_token_count == 1
    assert result.stopped_on_eos is True
    assert result.candidate_answer == ""
    assert metrics.GATING_EFFORT == 4
    assert metrics.GATING_MAX_NEW_TOKENS == 96
