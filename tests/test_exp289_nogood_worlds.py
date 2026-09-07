from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make(
    *,
    replicate: int = 7,
    rng_stream: str = "evaluation",
    variables: int = 6,
    decoys: int = 2,
    restarts: int = 3,
):
    from nolane_ai.experiments.exp289_nogood_worlds import Exp289NogoodGenerator

    return Exp289NogoodGenerator(root_seed="exp289-nogood-world-test").make_batch(
        replicate=replicate,
        batch_size=3,
        timesteps=4,
        restarts=restarts,
        variables=variables,
        decoys=decoys,
        d_model=8,
        noise_std=0.05,
        rng_stream=rng_stream,
    )


def test_exp289_restart_world_regenerates_byte_identically() -> None:
    first = _make()
    second = _make()

    assert first.digest == second.digest
    assert first.metadata == second.metadata
    assert torch.equal(first.surface_events, second.surface_events)
    assert torch.equal(first.variable_states, second.variable_states)
    assert torch.equal(first.solution_targets, second.solution_targets)
    assert torch.equal(first.restart_orders, second.restart_orders)
    assert torch.equal(first.restart_value_orders, second.restart_value_orders)


def test_exp289_restart_world_digest_binds_lineage_and_geometry() -> None:
    baseline = _make().digest
    assert _make(replicate=8).digest != baseline
    assert _make(rng_stream="augmentation").digest != baseline
    assert _make(variables=7).digest != baseline
    assert _make(decoys=3).digest != baseline
    assert _make(restarts=4).digest != baseline


def test_exp289_worlds_are_globally_solvable_with_arm_independent_repeat_opportunities() -> None:
    from nolane_ai.experiments.exp289_nogood_worlds import problem_from_exp289_episode

    batch = _make(decoys=2, restarts=3)
    assert batch.surface_events.shape == (3, 4, 8)
    assert batch.variable_states.shape == (3, 6, 8)
    assert batch.solution_targets.shape == (3, 6)
    assert batch.restart_orders.shape == (3, 3, 6)
    assert batch.restart_value_orders.shape == (3, 3, 6, 2)

    for index, episode in enumerate(batch.metadata["episodes"]):
        assert episode["globally_solvable"] is True
        assert episode["productive_solution_exists"] is True
        assert episode["repeat_opportunity_semantics"] == "generator_frozen_before_arm_execution"
        assert episode["predeclared_repeat_opportunities"] >= 1
        assert episode["rder_denominator_eligible"] is True
        assert episode["evaluator_metadata_delivered_to_arm"] is False
        assert episode["arm_specific_world_mutation"] is False
        opportunities = episode["repeat_opportunities"]
        assert len(opportunities) == episode["predeclared_repeat_opportunities"]
        for opportunity in opportunities:
            assert opportunity["canonical_dead_end_key"]
            assert opportunity["first_restart_index"] == 0
            assert opportunity["later_restart_indices"]
            assert all(value > 0 for value in opportunity["later_restart_indices"])

        problem = problem_from_exp289_episode(episode)
        valid = [
            assignment
            for assignment in problem.enumerate_assignments(limit=1024)
            if problem.is_solution(assignment)
        ]
        assert valid
        assert episode["valid_solutions"] == [
            [[name, int(value)] for name, value in sorted(assignment.items())]
            for assignment in valid
        ]
        target = {
            f"v{variable}": int(batch.solution_targets[index, variable].item())
            for variable in range(batch.solution_targets.shape[1])
        }
        assert problem.is_solution(target)


def test_exp289_repeat_opportunity_metadata_is_not_an_arm_input_tensor() -> None:
    batch = _make()
    tensor_field_names = {
        "surface_events",
        "variable_states",
        "solution_targets",
        "restart_orders",
        "restart_value_orders",
    }
    assert "repeat_opportunities" not in tensor_field_names
    assert "valid_solutions" not in tensor_field_names
    assert "repeat_opportunities" in batch.metadata["episodes"][0]
    assert batch.metadata["episodes"][0]["evaluator_metadata_delivered_to_arm"] is False


def test_exp289_zero_opportunity_episode_is_retained_but_denominator_ineligible() -> None:
    batch = _make(decoys=0, restarts=3)
    assert len(batch.metadata["episodes"]) == 3
    for episode in batch.metadata["episodes"]:
        assert episode["predeclared_repeat_opportunities"] == 0
        assert episode["repeat_opportunities"] == []
        assert episode["rder_denominator_eligible"] is False
        assert episode["zero_opportunity_policy"] == "retain_raw_episode_exclude_from_rder_denominator"


def test_exp289_generator_rejects_nondevelopment_rng_or_invalid_geometry() -> None:
    from nolane_ai.experiments.exp289_nogood_worlds import Exp289NogoodGenerator

    generator = Exp289NogoodGenerator(root_seed="exp289-nogood-world-test")
    common = dict(
        replicate=0,
        batch_size=2,
        timesteps=3,
        restarts=3,
        variables=5,
        decoys=2,
        d_model=8,
        noise_std=0.05,
    )
    for stream in ("challenge", "model_init", "environment", "controller_noise"):
        with pytest.raises(ValueError, match="rng_stream"):
            generator.make_batch(**common, rng_stream=stream)
    with pytest.raises(ValueError, match="decoys"):
        generator.make_batch(
            replicate=0,
            batch_size=2,
            timesteps=3,
            restarts=3,
            variables=4,
            decoys=4,
            d_model=8,
            noise_std=0.05,
            rng_stream="evaluation",
        )
    with pytest.raises(ValueError, match="restarts"):
        generator.make_batch(
            replicate=0,
            batch_size=2,
            timesteps=3,
            restarts=0,
            variables=5,
            decoys=2,
            d_model=8,
            noise_std=0.05,
            rng_stream="evaluation",
        )
