from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


STRATA = ("PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL")


def _make(*, replicate: int = 7, rng_stream: str = "evaluation", stratum: str = "PROPAGATION_FIT", variables: int = 4):
    from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator

    return Exp279RoutingGenerator(root_seed="exp279-routing-world-test").make_batch(
        replicate=replicate,
        batch_size=2,
        timesteps=3,
        variables=variables,
        constraints=2,
        d_model=8,
        noise_std=0.05,
        rng_stream=rng_stream,
        stratum=stratum,
    )


def test_exp279_routing_world_regenerates_byte_identically() -> None:
    first = _make()
    second = _make()
    assert first.digest == second.digest
    assert first.metadata == second.metadata
    assert torch.equal(first.surface_events, second.surface_events)
    assert torch.equal(first.variable_states, second.variable_states)
    assert torch.equal(first.incidence, second.incidence)
    assert torch.equal(first.targets, second.targets)


def test_exp279_routing_world_digest_binds_replicate_stream_stratum_and_geometry() -> None:
    baseline = _make().digest
    assert _make(replicate=8).digest != baseline
    assert _make(rng_stream="augmentation").digest != baseline
    assert _make(stratum="BRANCH_FIT").digest != baseline
    assert _make(variables=5).digest != baseline


def test_exp279_generator_supports_all_predeclared_structure_fit_strata() -> None:
    batches = [_make(stratum=stratum) for stratum in STRATA]
    assert [batch.stratum for batch in batches] == list(STRATA)
    assert len({batch.digest for batch in batches}) == len(STRATA)
    for batch in batches:
        assert batch.incidence.shape == (2, 2, 4)
        assert batch.surface_events.shape == (2, 3, 8)
        assert batch.variable_states.shape == (2, 4, 8)
        assert batch.targets.shape == (2, 4)
        assert set(batch.targets.unique().tolist()) <= {0, 1}
        assert batch.metadata["stratum"] == batch.stratum
        assert batch.metadata["experiment_id"] == "EXP-279"
        assert batch.metadata["stratum_semantics"]


def test_exp279_generator_rejects_nonpredeclared_stratum_or_rng_stream() -> None:
    from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator

    generator = Exp279RoutingGenerator(root_seed="exp279-routing-world-test")
    common = dict(
        replicate=0,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        d_model=8,
        noise_std=0.05,
    )
    with pytest.raises(ValueError, match="stratum"):
        generator.make_batch(**common, rng_stream="evaluation", stratum="TUNED_AFTER_RESULT")
    with pytest.raises(ValueError, match="rng_stream"):
        generator.make_batch(**common, rng_stream="challenge", stratum="PROPAGATION_FIT")
