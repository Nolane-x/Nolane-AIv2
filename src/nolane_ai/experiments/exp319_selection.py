from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .exp319_contract import CONTRACT, DIAGNOSTIC_ARMS, PRIMARY_ARMS, TASK_FAMILIES


@dataclass(frozen=True, slots=True)
class StageASelectionRecord:
    arm_id: str
    root: int
    learning_rate: float
    step: int
    initial_answer_only_loss: float
    answer_only_loss: float
    answer_token_accuracy: float
    exact_match: float
    eos_correctness: float
    invalid_output_rate: float
    nonfinite_events: int


@dataclass(frozen=True, slots=True)
class StageBSelectionRecord:
    arm_id: str
    root: int
    step: int
    train_exact_match: float
    iid_answer_only_loss: float
    iid_answer_token_accuracy: float
    iid_family_balanced_exact_match: float
    iid_family_exact: tuple[tuple[str, float], ...]
    eos_correctness: float
    invalid_output_rate: float


@dataclass(frozen=True, slots=True)
class StageALearningRateSelection:
    selected: StageASelectionRecord
    passes_floor: bool


@dataclass(frozen=True, slots=True)
class StageBRootSelection:
    selected: StageBSelectionRecord
    passes_floor: bool
    used_passing_checkpoint_rule: bool


def _finite_unit_interval(value: float, *, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")


def _finite_nonnegative(value: float, *, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


def validate_stage_a_record(record: StageASelectionRecord) -> None:
    if record.arm_id not in DIAGNOSTIC_ARMS:
        raise ValueError("Stage A arm mismatch")
    if record.root != CONTRACT.stage_a.root:
        raise ValueError("Stage A is frozen to root=0")
    if record.learning_rate not in CONTRACT.stage_a.learning_rates:
        raise ValueError("Stage A learning-rate candidate mismatch")
    if record.step != max(CONTRACT.stage_a.checkpoints):
        raise ValueError("Stage A LR selection requires the step-1024 checkpoint")
    _finite_nonnegative(record.initial_answer_only_loss, label="initial_answer_only_loss")
    if record.initial_answer_only_loss <= 0.0:
        raise ValueError("initial_answer_only_loss must be positive")
    _finite_nonnegative(record.answer_only_loss, label="answer_only_loss")
    _finite_unit_interval(record.answer_token_accuracy, label="answer_token_accuracy")
    _finite_unit_interval(record.exact_match, label="exact_match")
    _finite_unit_interval(record.eos_correctness, label="eos_correctness")
    _finite_unit_interval(record.invalid_output_rate, label="invalid_output_rate")
    if isinstance(record.nonfinite_events, bool) or record.nonfinite_events < 0:
        raise ValueError("nonfinite_events must be a non-negative integer")


def stage_a_passes(record: StageASelectionRecord) -> bool:
    validate_stage_a_record(record)
    floor = CONTRACT.stage_a.floor
    return (
        record.exact_match >= floor.exact_match
        and record.answer_token_accuracy >= floor.answer_token_accuracy
        and record.eos_correctness >= floor.eos_correctness
        and record.answer_only_loss
        <= floor.final_loss_fraction_of_initial * record.initial_answer_only_loss
        and record.invalid_output_rate <= floor.invalid_output_rate_max
        and record.nonfinite_events <= floor.nan_inf_events_max
    )


def select_stage_a_learning_rate(
    records: Iterable[StageASelectionRecord],
) -> StageASelectionRecord:
    materialized = tuple(records)
    if len(materialized) != 2:
        raise ValueError("Stage A LR selection requires exactly two frozen candidates")
    for record in materialized:
        validate_stage_a_record(record)
    if len({record.arm_id for record in materialized}) != 1:
        raise ValueError("Stage A LR selection records must share one arm")
    if len({record.root for record in materialized}) != 1:
        raise ValueError("Stage A LR selection records must share one root")
    if {record.learning_rate for record in materialized} != set(CONTRACT.stage_a.learning_rates):
        raise ValueError("Stage A learning-rate candidates must match the frozen pair")
    return max(
        materialized,
        key=lambda record: (
            record.exact_match,
            record.answer_token_accuracy,
            -record.answer_only_loss,
            -record.learning_rate,
        ),
    )


def select_stage_a_learning_rate_with_disposition(
    records: Iterable[StageASelectionRecord],
) -> StageALearningRateSelection:
    selected = select_stage_a_learning_rate(records)
    return StageALearningRateSelection(selected=selected, passes_floor=stage_a_passes(selected))


def validate_stage_b_record(record: StageBSelectionRecord) -> None:
    if record.arm_id not in PRIMARY_ARMS:
        raise ValueError("Stage B arm mismatch")
    if record.root not in CONTRACT.stage_b.roots:
        raise ValueError("Stage B root mismatch")
    if record.step not in CONTRACT.stage_b.checkpoints:
        raise ValueError("Stage B checkpoint mismatch")
    _finite_unit_interval(record.train_exact_match, label="train_exact_match")
    _finite_nonnegative(record.iid_answer_only_loss, label="iid_answer_only_loss")
    _finite_unit_interval(record.iid_answer_token_accuracy, label="iid_answer_token_accuracy")
    _finite_unit_interval(
        record.iid_family_balanced_exact_match,
        label="iid_family_balanced_exact_match",
    )
    _finite_unit_interval(record.eos_correctness, label="eos_correctness")
    _finite_unit_interval(record.invalid_output_rate, label="invalid_output_rate")
    if len(record.iid_family_exact) != len(TASK_FAMILIES):
        raise ValueError("Stage B family metrics must contain exactly four families")
    names = tuple(name for name, _ in record.iid_family_exact)
    if len(set(names)) != len(names) or set(names) != set(TASK_FAMILIES):
        raise ValueError("Stage B family metrics must match the frozen task families")
    for family, score in record.iid_family_exact:
        _finite_unit_interval(score, label=f"iid_family_exact[{family}]")
    recomputed = sum(score for _, score in record.iid_family_exact) / len(record.iid_family_exact)
    if not math.isclose(
        recomputed,
        record.iid_family_balanced_exact_match,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Stage B family-balanced exact match does not match family metrics")


def stage_b_checkpoint_passes(record: StageBSelectionRecord) -> bool:
    validate_stage_b_record(record)
    floor = CONTRACT.stage_b.floor
    qualifying_families = sum(
        score >= floor.per_family_exact_min for _, score in record.iid_family_exact
    )
    return (
        record.train_exact_match >= floor.train_exact_match
        and record.iid_answer_token_accuracy >= floor.iid_answer_token_accuracy
        and record.iid_family_balanced_exact_match >= floor.iid_family_balanced_exact_match
        and qualifying_families >= floor.min_families_exact_at_least
        and record.eos_correctness >= floor.eos_correctness
        and record.invalid_output_rate <= floor.invalid_output_rate_max
    )


def select_stage_b_checkpoint(
    records: Iterable[StageBSelectionRecord],
) -> StageBSelectionRecord:
    materialized = tuple(records)
    expected_steps = set(CONTRACT.stage_b.checkpoints)
    if len(materialized) != len(expected_steps):
        raise ValueError("Stage B selection requires all three frozen checkpoints")
    for record in materialized:
        validate_stage_b_record(record)
    if len({record.arm_id for record in materialized}) != 1:
        raise ValueError("Stage B selection records must share one arm")
    if len({record.root for record in materialized}) != 1:
        raise ValueError("Stage B selection records must share one root")
    if {record.step for record in materialized} != expected_steps:
        raise ValueError("Stage B selection checkpoints must be exactly 512, 1024, and 2048")

    passing = tuple(record for record in materialized if stage_b_checkpoint_passes(record))
    if passing:
        return min(passing, key=lambda record: record.step)
    return max(
        materialized,
        key=lambda record: (
            record.iid_family_balanced_exact_match,
            record.iid_answer_token_accuracy,
            -record.iid_answer_only_loss,
            -record.step,
        ),
    )


def select_stage_b_checkpoint_with_disposition(
    records: Iterable[StageBSelectionRecord],
) -> StageBRootSelection:
    materialized = tuple(records)
    selected = select_stage_b_checkpoint(materialized)
    passes = stage_b_checkpoint_passes(selected)
    return StageBRootSelection(
        selected=selected,
        passes_floor=passes,
        used_passing_checkpoint_rule=passes,
    )


def select_stage_b_checkpoints_by_root(
    records: Iterable[StageBSelectionRecord],
    *,
    arm_id: str,
) -> tuple[StageBSelectionRecord, ...]:
    if arm_id not in PRIMARY_ARMS:
        raise ValueError("Stage B aggregate selection arm mismatch")
    relevant = tuple(record for record in records if record.arm_id == arm_id)
    roots = {record.root for record in relevant}
    if roots != set(CONTRACT.stage_b.roots):
        raise ValueError("Stage B aggregate selection requires all four frozen roots")
    selected: list[StageBSelectionRecord] = []
    for root in CONTRACT.stage_b.roots:
        root_records = tuple(record for record in relevant if record.root == root)
        selected.append(select_stage_b_checkpoint(root_records))
    return tuple(selected)


def stage_b_cross_root_passes(records: Iterable[StageBSelectionRecord]) -> bool:
    materialized = tuple(records)
    if len(materialized) != len(CONTRACT.stage_b.roots):
        raise ValueError("Stage B cross-root gate requires exactly four selected roots")
    for record in materialized:
        validate_stage_b_record(record)
    if {record.root for record in materialized} != set(CONTRACT.stage_b.roots):
        raise ValueError("Stage B cross-root gate requires roots 1, 2, 3, and 4")
    if len({record.arm_id for record in materialized}) != 1:
        raise ValueError("Stage B cross-root records must share one arm")
    return sum(stage_b_checkpoint_passes(record) for record in materialized) >= CONTRACT.stage_b.cross_root_min_passes
