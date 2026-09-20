from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp337_contract import (
    ARMS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FINAL_CUMULATIVE_STEP,
    GOLD_LOSS_WEIGHT,
    REDUCER_ORDER,
    SELF_ROLLIN_LOSS_WEIGHT,
    STARTING_CUMULATIVE_STEP,
    UPDATES_PER_CHUNK,
    WORLD_IDS,
    BoundaryResult,
    arm_pass,
    chunk_schedule,
    data_order_digest,
    reduce_full32,
    sham_equivalent,
    validate_boundary,
)


D = "0" * 64


def boundary(
    arm: str,
    *,
    passed: bool,
    chunk_index: int = 7,
    model_digest: str = D,
    projection_update_count: int = 100,
    projected_target_count: int = 500,
) -> BoundaryResult:
    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    updates = exposure * len(WORLD_IDS)
    token = [1.0] * len(WORLD_IDS)
    exact = [1.0] * len(WORLD_IDS)
    greedy = 0.95
    aggregate_token = 0.995
    if not passed:
        token[0] = 0.5
        exact[0] = 0.0
        greedy = 0.50
        aggregate_token = 0.90

    measure = 0 if arm == "CONTROL_PROJECT_GOLD_PREFIX" else updates
    active = updates if arm == "SELF_ROLLIN_RECOVERY_PROJECT" else 0
    divergence = min(measure, 7)
    return BoundaryResult(
        arm=arm,
        chunk_index=chunk_index,
        cumulative_exposure_per_world=exposure,
        cumulative_source_updates=updates,
        cumulative_training_step=STARTING_CUMULATIVE_STEP + updates,
        world_token_accuracies=tuple(token),
        world_full_answer_exact=tuple(exact),
        aggregate_answer_only_loss=0.1,
        aggregate_answer_token_accuracy=aggregate_token,
        aggregate_greedy_exact_match=greedy,
        aggregate_eos_correctness=1.0,
        aggregate_invalid_output_rate=0.0,
        aggregate_loss_fraction_of_original_initial=0.001,
        model_state_digest=model_digest,
        optimizer_state_digest=D,
        rng_state_digest=D,
        nonfinite_events=0,
        negative_target_count=1000,
        projection_update_count=projection_update_count,
        projected_target_count=projected_target_count,
        self_rollin_measurement_count=measure,
        self_rollin_active_update_count=active,
        self_rollin_divergent_update_count=divergence,
    )


def matched_control_sham(*, passed: bool) -> tuple[BoundaryResult, BoundaryResult]:
    control = boundary("CONTROL_PROJECT_GOLD_PREFIX", passed=passed)
    sham = boundary("SHAM_SELF_ROLLIN_MEASURE_PROJECT", passed=passed)
    return control, sham


def test_geometry_is_exactly_8_chunks_and_1024_updates() -> None:
    assert CHUNK_COUNT == 8
    assert FINAL_CUMULATIVE_STEP == 3072
    for chunk_index in range(CHUNK_COUNT):
        rows = chunk_schedule(chunk_index)
        assert len(rows) == UPDATES_PER_CHUNK == 128
        assert rows[0][2] == chunk_index * EXPOSURES_PER_CHUNK
        assert rows[-1][2] == (chunk_index + 1) * EXPOSURES_PER_CHUNK - 1
    assert len(data_order_digest(32)) == 64


def test_loss_weights_are_frozen_half_half() -> None:
    assert GOLD_LOSS_WEIGHT == pytest.approx(0.5)
    assert SELF_ROLLIN_LOSS_WEIGHT == pytest.approx(0.5)
    assert GOLD_LOSS_WEIGHT + SELF_ROLLIN_LOSS_WEIGHT == pytest.approx(1.0)


def test_boundary_rollin_counters_are_arm_specific() -> None:
    for arm in ARMS:
        validate_boundary(boundary(arm, passed=True))
    bad = replace(
        boundary("CONTROL_PROJECT_GOLD_PREFIX", passed=True),
        self_rollin_measurement_count=1,
    )
    with pytest.raises(ValueError, match="CONTROL roll-in"):
        validate_boundary(bad)


def test_sham_equivalence_ignores_measurement_only_rollin_counters() -> None:
    control, sham = matched_control_sham(passed=False)
    assert sham_equivalent(control, sham)
    broken = replace(sham, model_state_digest="1" * 64)
    assert not sham_equivalent(control, broken)


def test_reducer_parent_and_sham_fail_closed() -> None:
    control, sham = matched_control_sham(passed=False)
    recovery = boundary("SELF_ROLLIN_RECOVERY_PROJECT", passed=False)
    decision, _ = reduce_full32(control, sham, recovery, parent_authority_valid=False)
    assert decision == REDUCER_ORDER[1]

    broken_sham = replace(sham, optimizer_state_digest="2" * 64)
    decision, _ = reduce_full32(control, broken_sham, recovery, parent_authority_valid=True)
    assert decision == REDUCER_ORDER[2]


def test_reducer_both_pass_selects_simpler_control_successor_state() -> None:
    control, sham = matched_control_sham(passed=True)
    recovery = boundary("SELF_ROLLIN_RECOVERY_PROJECT", passed=True)
    decision, vectors = reduce_full32(control, sham, recovery, parent_authority_valid=True)
    assert decision == REDUCER_ORDER[3]
    assert vectors["control_aggregate_pass"] is True
    assert vectors["recovery_aggregate_pass"] is True


def test_reducer_control_only_rejects_recovery() -> None:
    control, sham = matched_control_sham(passed=True)
    recovery = boundary("SELF_ROLLIN_RECOVERY_PROJECT", passed=False)
    decision, _ = reduce_full32(control, sham, recovery, parent_authority_valid=True)
    assert decision == REDUCER_ORDER[4]


def test_reducer_causal_rescue_requires_recovery_pass_and_no_regression() -> None:
    control, sham = matched_control_sham(passed=False)
    recovery = boundary("SELF_ROLLIN_RECOVERY_PROJECT", passed=True)
    decision, vectors = reduce_full32(control, sham, recovery, parent_authority_valid=True)
    assert decision == REDUCER_ORDER[5]
    assert WORLD_IDS[0] in vectors["rescued_control_failures"]
    assert vectors["recovery_regressions"] == []


def test_partial_improvement_is_still_negative() -> None:
    control, sham = matched_control_sham(passed=False)
    recovery = boundary("SELF_ROLLIN_RECOVERY_PROJECT", passed=False)
    # Make recovery locally better without satisfying the full gate.
    token = list(recovery.world_token_accuracies)
    exact = list(recovery.world_full_answer_exact)
    token[0] = 0.98
    exact[0] = 0.0
    recovery = replace(recovery, world_token_accuracies=tuple(token), world_full_answer_exact=tuple(exact))
    decision, _ = reduce_full32(control, sham, recovery, parent_authority_valid=True)
    assert decision == REDUCER_ORDER[6]
    assert not arm_pass(recovery)
