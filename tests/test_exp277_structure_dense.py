from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make(*, replicate: int = 7, rng_stream: str = "evaluation", variables: int = 5):
    from nolane_ai.experiments.exp277_structure_dense import Exp277StructureDenseGenerator

    return Exp277StructureDenseGenerator(root_seed="exp277-structure-test").make_batch(
        replicate=replicate,
        batch_size=2,
        timesteps=4,
        variables=variables,
        constraints=2,
        d_model=8,
        noise_std=0.05,
        rng_stream=rng_stream,
    )


def test_structure_dense_batch_regenerates_exactly_and_has_closed_shapes() -> None:
    first = _make()
    second = _make()
    assert first.digest == second.digest
    assert first.replicate == 7
    assert first.rng_stream == "evaluation"
    assert torch.equal(first.surface_events, second.surface_events)
    assert torch.equal(first.variable_states, second.variable_states)
    assert torch.equal(first.oracle_incidence, second.oracle_incidence)
    assert torch.equal(first.targets, second.targets)
    assert first.surface_events.shape == (2, 4, 8)
    assert first.variable_states.shape == (2, 5, 8)
    assert first.oracle_incidence.shape == (2, 2, 5)
    assert first.targets.shape == (2, 5)
    assert set(first.targets.flatten().tolist()) <= {0, 1}
    assert torch.all(first.oracle_incidence.sum(dim=1) == 1)


def test_structure_dense_digest_changes_with_lineage_stream_or_geometry() -> None:
    base = _make()
    assert _make(replicate=8).digest != base.digest
    assert _make(rng_stream="augmentation").digest != base.digest
    assert _make(variables=6).digest != base.digest


def test_structure_dense_rejects_invalid_geometry_and_stream() -> None:
    from nolane_ai.experiments.exp277_structure_dense import Exp277StructureDenseGenerator

    generator = Exp277StructureDenseGenerator(root_seed="exp277-structure-test")
    kwargs = dict(
        replicate=0,
        batch_size=2,
        timesteps=2,
        variables=4,
        constraints=2,
        d_model=8,
        noise_std=0.0,
        rng_stream="evaluation",
    )
    with pytest.raises(ValueError, match="constraints cannot exceed variables"):
        generator.make_batch(**(kwargs | {"constraints": 5}))
    with pytest.raises(ValueError, match="timesteps"):
        generator.make_batch(**(kwargs | {"timesteps": 1}))
    with pytest.raises(ValueError, match="rng_stream"):
        generator.make_batch(**(kwargs | {"rng_stream": "environment"}))
