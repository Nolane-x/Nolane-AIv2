from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make(
    *,
    replicate: int = 7,
    rng_stream: str = "evaluation",
    variables: int = 6,
    decoys: int = 2,
):
    from nolane_ai.experiments.exp286_conflict_worlds import Exp286ConflictGenerator

    return Exp286ConflictGenerator(root_seed="exp286-conflict-world-test").make_batch(
        replicate=replicate,
        batch_size=3,
        timesteps=4,
        variables=variables,
        decoys=decoys,
        d_model=8,
        noise_std=0.05,
        rng_stream=rng_stream,
    )


def test_exp286_conflict_world_regenerates_byte_identically() -> None:
    first = _make()
    second = _make()

    assert first.digest == second.digest
    assert first.metadata == second.metadata
    assert torch.equal(first.surface_events, second.surface_events)
    assert torch.equal(first.variable_states, second.variable_states)
    assert torch.equal(first.solution_targets, second.solution_targets)
    assert torch.equal(first.bad_branch_values, second.bad_branch_values)
    assert torch.equal(first.core_masks, second.core_masks)
    assert torch.equal(first.decoy_order, second.decoy_order)


def test_exp286_conflict_world_digest_binds_lineage_and_geometry() -> None:
    baseline = _make().digest
    assert _make(replicate=8).digest != baseline
    assert _make(rng_stream="augmentation").digest != baseline
    assert _make(variables=7).digest != baseline
    assert _make(decoys=3).digest != baseline


def test_exp286_conflict_world_is_solvable_with_exact_local_conflict_core() -> None:
    batch = _make()

    assert batch.surface_events.shape == (3, 4, 8)
    assert batch.variable_states.shape == (3, 6, 8)
    assert batch.solution_targets.shape == (3, 6)
    assert batch.bad_branch_values.shape == (3, 6)
    assert batch.core_masks.shape == (3, 6)
    assert batch.decoy_order.shape == (3, 6)
    assert set(batch.solution_targets.unique().tolist()) <= {0, 1}
    assert torch.equal(batch.bad_branch_values, 1 - batch.solution_targets)
    assert torch.all(batch.core_masks.sum(dim=1) >= 2)

    episodes = batch.metadata["episodes"]
    assert len(episodes) == 3
    for index, episode in enumerate(episodes):
        assert episode["globally_solvable"] is True
        assert episode["productive_solution_exists"] is True
        assert episode["local_contradiction_present"] is True
        assert episode["core_minimality"] == "generator_exact_by_construction"
        assert len(episode["core_variables"]) >= 2
        assert len(episode["decoy_variables"]) == 2
        assert episode["bad_branch_value"] != episode["solution_value_at_conflict_variable"]
        for variable in episode["core_variables"]:
            assert batch.core_masks[index, variable].item() == 1.0


def test_exp286_decoys_are_outside_the_ground_truth_conflict_core() -> None:
    batch = _make(decoys=2)
    for index, episode in enumerate(batch.metadata["episodes"]):
        for variable in episode["decoy_variables"]:
            assert batch.core_masks[index, variable].item() == 0.0
        assert batch.decoy_order[index, :2].tolist() == episode["decoy_variables"]


def test_exp286_generator_rejects_nondevelopment_rng_or_invalid_geometry() -> None:
    from nolane_ai.experiments.exp286_conflict_worlds import Exp286ConflictGenerator

    generator = Exp286ConflictGenerator(root_seed="exp286-conflict-world-test")
    common = dict(
        replicate=0,
        batch_size=2,
        timesteps=3,
        variables=5,
        decoys=2,
        d_model=8,
        noise_std=0.05,
    )
    with pytest.raises(ValueError, match="rng_stream"):
        generator.make_batch(**common, rng_stream="challenge")
    with pytest.raises(ValueError, match="decoys"):
        generator.make_batch(
            replicate=0,
            batch_size=2,
            timesteps=3,
            variables=4,
            decoys=3,
            d_model=8,
            noise_std=0.05,
            rng_stream="evaluation",
        )
