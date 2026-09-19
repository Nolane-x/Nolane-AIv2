from nolane_ai.experiments.exp335_contract import (
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FAMILIES,
    REDUCER_ORDER,
    UPDATES_PER_CHUNK,
    WORLD_IDS,
    BoundaryResult,
    chunk_schedule,
    reduce_full32,
)


def _boundary(arm: str, *, control_fail=(), project_fail=(), projections=0, state="a"):
    fail = set(control_fail if arm != "SUBSPACE_PROJECT_FULL32" else project_fail)
    token = tuple(0.5 if world in fail else 1.0 for world in WORLD_IDS)
    exact = tuple(0.0 if world in fail else 1.0 for world in WORLD_IDS)
    return BoundaryResult(
        arm=arm,
        chunk_index=7,
        cumulative_exposure_per_world=32,
        cumulative_source_updates=1024,
        world_token_accuracies=token,
        world_full_answer_exact=exact,
        model_state_digest=state * 64,
        optimizer_state_digest="b" * 64,
        rng_state_digest="c" * 64,
        nonfinite_events=0,
        negative_target_count=projections,
        projection_update_count=projections if arm == "SUBSPACE_PROJECT_FULL32" else 0,
        projected_target_count=projections if arm == "SUBSPACE_PROJECT_FULL32" else 0,
    )


def test_frozen_full32_chunk_geometry():
    assert ARMS == ("CONTROL_FULL32", "SHAM_MEASURE_FULL32", "SUBSPACE_PROJECT_FULL32")
    assert len(FAMILIES) == 4
    assert len(WORLD_IDS) == 32
    assert CHUNK_COUNT == 8
    assert EXPOSURES_PER_CHUNK == 4
    assert UPDATES_PER_CHUNK == 128
    full = []
    for chunk in range(CHUNK_COUNT):
        schedule = chunk_schedule(chunk)
        assert len(schedule) == 128
        full.extend(schedule)
    assert len(full) == 1024
    assert [row[3] for row in full[:128:32]] == [1, 2, 4, 8]


def test_strongest_project_rescue_reducer():
    failed = (WORLD_IDS[0], WORLD_IDS[7])
    control = _boundary("CONTROL_FULL32", control_fail=failed)
    sham = _boundary("SHAM_MEASURE_FULL32", control_fail=failed)
    project = _boundary("SUBSPACE_PROJECT_FULL32", project_fail=(), projections=12)
    decision, vectors = reduce_full32(
        control, sham, project, parent_authority_valid=True
    )
    assert decision == "AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION"
    assert vectors["control_failed_worlds"] == list(failed)
    assert vectors["rescued_control_failures"] == list(failed)
    assert vectors["project_regressions"] == []


def test_reducer_is_control_first_and_fail_closed():
    control = _boundary("CONTROL_FULL32")
    sham = _boundary("SHAM_MEASURE_FULL32")
    project = _boundary("SUBSPACE_PROJECT_FULL32", project_fail=(WORLD_IDS[3],), projections=2)
    assert reduce_full32(control, sham, project, parent_authority_valid=True)[0] == (
        "AFIXED_FULL32_FOUNDATION_REENTERED_CONTROL_ONLY_PROJECT_REGRESSION"
    )
    drifted = BoundaryResult(
        **{**sham.__dict__, "model_state_digest": "d" * 64}
    ) if hasattr(sham, "__dict__") else BoundaryResult(
        sham.arm, sham.chunk_index, sham.cumulative_exposure_per_world,
        sham.cumulative_source_updates, sham.world_token_accuracies,
        sham.world_full_answer_exact, "d"*64, sham.optimizer_state_digest,
        sham.rng_state_digest, sham.nonfinite_events, sham.negative_target_count,
        sham.projection_update_count, sham.projected_target_count
    )
    assert reduce_full32(control, drifted, project, parent_authority_valid=True)[0] == "SHAM_FULL32_MISMATCH"
    assert reduce_full32(control, sham, project, parent_authority_valid=False)[0] == "PARENT_AUTHORITY_MISMATCH"
    assert REDUCER_ORDER[0] == "INVALID_FULL32_FOUNDATION_REENTRY"


def test_all_forbidden_authorizations_remain_false():
    assert AUTHORIZATION_FLAGS == {
        "exp302_implementation_authorized": False,
        "exp320_implementation_authorized": False,
        "scale_authorized": False,
        "authorized_30m": False,
        "authorized_100m": False,
    }
