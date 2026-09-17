from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import nn

from .exp301_scientific import GenerationResult, greedy_generate
from .exp301_training import compute_answer_only_loss
from .exp319_contract import COMMON_GATING_EFFORT, MAX_GENERATION_TOKENS


GATING_EFFORT = COMMON_GATING_EFFORT
GATING_MAX_NEW_TOKENS = MAX_GENERATION_TOKENS


@dataclass(frozen=True, slots=True)
class GenerationMetricRecord:
    family: str
    exact: bool
    stopped_on_eos: bool
    invalid_output: bool


@dataclass(frozen=True, slots=True)
class GenerationMetricSummary:
    exact_match: float
    family_exact: tuple[tuple[str, float], ...]
    family_balanced_exact_match: float
    eos_correctness: float
    invalid_output_rate: float
    nonzero_exact_families: int


@dataclass(frozen=True, slots=True)
class NonFiniteReport:
    parameter_nonfinite: int
    gradient_nonfinite: int
    loss_nonfinite: int

    @property
    def total(self) -> int:
        return self.parameter_nonfinite + self.gradient_nonfinite + self.loss_nonfinite


def _answer_target_start(answer_start: int, target_time: int) -> int:
    if answer_start <= 0:
        raise ValueError("answer_start must be positive")
    first_target_index = answer_start - 1
    if first_target_index >= target_time:
        raise ValueError("answer_start leaves no answer targets")
    return first_target_index


def answer_only_cross_entropy(
    logits: torch.Tensor,
    *,
    targets: torch.Tensor,
    answer_start: int,
) -> torch.Tensor:
    """Frozen answer-only loss, including EOS and excluding prompt targets."""

    return compute_answer_only_loss(
        logits,
        targets=targets,
        answer_start=answer_start,
    )


def answer_token_accuracy(
    logits: torch.Tensor,
    *,
    targets: torch.Tensor,
    answer_start: int,
) -> float:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be [batch,time,vocab] and targets [batch,time]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits/targets time geometry mismatch")
    first_target_index = _answer_target_start(answer_start, targets.shape[1])
    predicted = torch.argmax(logits[:, first_target_index:, :], dim=-1)
    expected = targets[:, first_target_index:]
    if expected.numel() == 0:
        raise ValueError("answer target span cannot be empty")
    return float((predicted == expected).to(torch.float64).mean().item())


def summarize_generation_records(
    records: Iterable[GenerationMetricRecord],
) -> GenerationMetricSummary:
    materialized = tuple(records)
    if not materialized:
        raise ValueError("generation records cannot be empty")
    families: dict[str, list[int]] = {}
    exact_total = 0
    eos_total = 0
    invalid_total = 0
    for record in materialized:
        if not record.family:
            raise ValueError("generation record family cannot be empty")
        bucket = families.setdefault(record.family, [0, 0])
        bucket[0] += int(record.exact)
        bucket[1] += 1
        exact_total += int(record.exact)
        eos_total += int(record.stopped_on_eos)
        invalid_total += int(record.invalid_output)
    family_exact = tuple(
        (family, successes / total)
        for family, (successes, total) in sorted(families.items())
    )
    return GenerationMetricSummary(
        exact_match=exact_total / len(materialized),
        family_exact=family_exact,
        family_balanced_exact_match=sum(score for _, score in family_exact) / len(family_exact),
        eos_correctness=eos_total / len(materialized),
        invalid_output_rate=invalid_total / len(materialized),
        nonzero_exact_families=sum(score > 0.0 for _, score in family_exact),
    )


def invalid_output_rate(
    generated_token_sequences: Iterable[Sequence[int]],
    *,
    eos_id: int = 3,
    byte_offset: int = 4,
    byte_count: int = 256,
) -> float:
    sequences = tuple(tuple(int(token) for token in sequence) for sequence in generated_token_sequences)
    if not sequences:
        raise ValueError("generated token sequences cannot be empty")
    byte_limit = byte_offset + byte_count
    invalid = 0
    for sequence in sequences:
        sequence_invalid = False
        for token in sequence:
            if token == eos_id:
                break
            if not byte_offset <= token < byte_limit:
                sequence_invalid = True
                break
        invalid += int(sequence_invalid)
    return invalid / len(sequences)


def global_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Global L2 gradient norm measured before clipping."""

    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(dtype=torch.float64)
        squared += float(torch.sum(gradient * gradient).item())
    return math.sqrt(squared)


def snapshot_trainable_parameters(parameters: Iterable[nn.Parameter]) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().cpu().clone()
        for parameter in parameters
        if parameter.requires_grad
    )


def parameter_update_norm_ratio(
    before: Sequence[torch.Tensor],
    after: Sequence[torch.Tensor],
) -> float:
    if len(before) != len(after):
        raise ValueError("parameter geometry mismatch")
    delta_squared = 0.0
    before_squared = 0.0
    for before_tensor, after_tensor in zip(before, after):
        if before_tensor.shape != after_tensor.shape:
            raise ValueError("parameter geometry mismatch")
        before64 = before_tensor.detach().to(dtype=torch.float64, device="cpu")
        after64 = after_tensor.detach().to(dtype=torch.float64, device="cpu")
        delta = after64 - before64
        delta_squared += float(torch.sum(delta * delta).item())
        before_squared += float(torch.sum(before64 * before64).item())
    denominator = max(math.sqrt(before_squared), 1e-12)
    return math.sqrt(delta_squared) / denominator


def _count_nonfinite(tensor: torch.Tensor) -> int:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return 0
    return int((~torch.isfinite(tensor.detach())).sum().item())


def nonfinite_report(
    parameters: Iterable[nn.Parameter],
    *,
    loss: torch.Tensor | float | None = None,
) -> NonFiniteReport:
    parameter_nonfinite = 0
    gradient_nonfinite = 0
    for parameter in parameters:
        parameter_nonfinite += _count_nonfinite(parameter)
        if parameter.grad is not None:
            gradient_nonfinite += _count_nonfinite(parameter.grad)
    if loss is None:
        loss_nonfinite = 0
    elif isinstance(loss, torch.Tensor):
        loss_nonfinite = _count_nonfinite(loss)
    else:
        loss_nonfinite = int(not math.isfinite(float(loss)))
    return NonFiniteReport(
        parameter_nonfinite=parameter_nonfinite,
        gradient_nonfinite=gradient_nonfinite,
        loss_nonfinite=loss_nonfinite,
    )


def gating_generate(compiled: object, *, prompt: str) -> GenerationResult:
    """Run the one preregistered EXP-319 gating decode contract."""

    return greedy_generate(
        compiled,
        prompt=prompt,
        effort=GATING_EFFORT,
        max_new_tokens=GATING_MAX_NEW_TOKENS,
    )
