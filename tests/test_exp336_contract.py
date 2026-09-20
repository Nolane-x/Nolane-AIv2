from nolane_ai.experiments.exp336_contract import (
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FINAL_CUMULATIVE_STEP,
    REDUCER_ORDER,
    STARTING_CUMULATIVE_STEP,
    UPDATES_PER_CHUNK,
    WORLD_IDS,
    BoundaryResult,
    aggregate_pass,
    arm_pass,
    chunk_schedule,
    reduce_full32,
)


def _boundary(
    arm: str,
    *,
    world_fail=(),
    projections=0,
    state="a",
    aggregate_ok=True,
):
    fail = set(world_fail)
    token = tuple(0.5 if world in fail else 1.0 for world in WORLD_IDS)
    exact = tuple(0.0 if world in fail else 1.0 for world in WORLD_IDS)
    return BoundaryResult(
        arm=arm,
        chunk_index=7,
        cumulative_exposure_per_world=32,
        cumulative_source_updates=1024,
        cumulative_training_step=2048,
        world_token_accuracies=token,
        world_full_answer_exact=exact,
        aggregate_answer_only_loss=1.0,
        aggregate_answer_token_accuracy=1.0 if aggregate_ok else 0.98,
        aggregate_greedy_exact_match=1.0 if aggregate_ok else 0.5,
        aggregate_eos_correctness=1.0,
        aggregate_invalid_output_rate=0.0,
        aggregate_loss_fraction_of_original_initial=0.01,
        model_state_digest=state * 64,
        optimizer_state_digest="b" * 64,
        rng_state_digest="c" * 64,
        nonfinite_events=0,
        negative_target_count=projections,
        projection_update_count=projections if arm == "SUBSPACE_PROJECT_CNRS_FULL32" else 0,
        projected_target_count=projections if arm == "SUBSPACE_PROJECT_CNRS_FULL32" else 0,
    )


def test_frozen_cnrs_full32_geometry():
    assert ARMS == (
        "CONTROL_CNRS_FULL32",
        "SHAM_MEASURE_CNRS_FULL32",
        "SUBSPACE_PROJECT_CNRS_FULL32",
    )
    assert len(WORLD_IDS) == 32
    assert CHUNK_COUNT == 8
    assert EXPOSURES_PER_CHUNK == 4
    assert UPDATES_PER_CHUNK == 128
    assert STARTING_CUMULATIVE_STEP == 1024
    assert FINAL_CUMULATIVE_STEP == 2048
    full = []
    for chunk in range(CHUNK_COUNT):
        schedule = chunk_schedule(chunk)
        assert len(schedule) == 128
        full.extend(schedule)
    assert len(full) == 1024
    assert [row[3] for row in full[:128:32]] == [1, 2, 4, 8]


def test_inherited_projector_rescue_reducer_requires_real_world_rescue():
    failed = (WORLD_IDS[2], WORLD_IDS[6])
    control = _boundary("CONTROL_CNRS_FULL32", world_fail=failed)
    sham = _boundary("SHAM_MEASURE_CNRS_FULL32", world_fail=failed)
    project = _boundary("SUBSPACE_PROJECT_CNRS_FULL32", projections=12)
    decision, vectors = reduce_full32(
        control, sham, project, parent_authority_valid=True
    )
    assert decision == "CNRS_FULL32_STAGE_A_RESCUED_BY_INHERITED_PROJECTOR_NO_REGRESSION"
    assert vectors["control_failed_worlds"] == list(failed)
    assert vectors["rescued_control_failures"] == list(failed)
    assert vectors["project_regressions"] == []
    assert vectors["control_aggregate_pass"] is True
    assert vectors["project_aggregate_pass"] is True


def test_aggregate_gate_blocks_teacher_forced_only_promotion():
    control = _boundary("CONTROL_CNRS_FULL32", aggregate_ok=False)
    sham = _boundary("SHAM_MEASURE_CNRS_FULL32", aggregate_ok=False)
    project = _boundary("SUBSPACE_PROJECT_CNRS_FULL32", aggregate_ok=False, projections=10)
    assert all(control.world_token_accuracies)
    assert aggregate_pass(control) is False
    assert arm_pass(control) is False
    assert reduce_full32(control, sham, project, parent_authority_valid=True)[0] == (
        "CNRS_FULL32_STAGE_A_NOT_ESTABLISHED"
    )


def test_control_pass_project_regression_and_sham_fail_closed():
    control = _boundary("CONTROL_CNRS_FULL32")
    sham = _boundary("SHAM_MEASURE_CNRS_FULL32")
    project = _boundary(
        "SUBSPACE_PROJECT_CNRS_FULL32",
        world_fail=(WORLD_IDS[3],),
        projections=2,
    )
    assert reduce_full32(control, sham, project, parent_authority_valid=True)[0] == (
        "CNRS_FULL32_STAGE_A_ESTABLISHED_CONTROL_ONLY_PROJECT_REGRESSION"
    )
    drifted = BoundaryResult(
        arm=sham.arm,
        chunk_index=sham.chunk_index,
        cumulative_exposure_per_world=sham.cumulative_exposure_per_world,
        cumulative_source_updates=sham.cumulative_source_updates,
        cumulative_training_step=sham.cumulative_training_step,
        world_token_accuracies=sham.world_token_accuracies,
        world_full_answer_exact=sham.world_full_answer_exact,
        aggregate_answer_only_loss=sham.aggregate_answer_only_loss,
        aggregate_answer_token_accuracy=sham.aggregate_answer_token_accuracy,
        aggregate_greedy_exact_match=sham.aggregate_greedy_exact_match,
        aggregate_eos_correctness=sham.aggregate_eos_correctness,
        aggregate_invalid_output_rate=sham.aggregate_invalid_output_rate,
        aggregate_loss_fraction_of_original_initial=sham.aggregate_loss_fraction_of_original_initial,
        model_state_digest="d" * 64,
        optimizer_state_digest=sham.optimizer_state_digest,
        rng_state_digest=sham.rng_state_digest,
        nonfinite_events=sham.nonfinite_events,
        negative_target_count=sham.negative_target_count,
        projection_update_count=sham.projection_update_count,
        projected_target_count=sham.projected_target_count,
    )
    assert reduce_full32(control, drifted, project, parent_authority_valid=True)[0] == (
        "SHAM_CNRS_FULL32_MISMATCH"
    )
    assert REDUCER_ORDER[0] == "INVALID_CNRS_FULL32_STAGE_A"


def test_forbidden_downstream_authorizations_remain_false():
    assert AUTHORIZATION_FLAGS == {
        "exp302_implementation_authorized": False,
        "exp320_implementation_authorized": False,
        "stage_b_implementation_authorized": False,
        "stage_c_implementation_authorized": False,
        "scale_authorized": False,
        "authorized_30m": False,
        "authorized_100m": False,
    }
