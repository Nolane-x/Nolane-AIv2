from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _build(*, challenge_seed: int = 123456789, stratum: str = "PROPAGATION_FIT"):
    from nolane_ai.experiments.exp279_challenge_worlds import build_exp279_challenge_batch

    return build_exp279_challenge_batch(
        challenge_seed=challenge_seed,
        replicate=400,
        stratum=stratum,
        batch_size=3,
        timesteps=4,
        variables=5,
        constraints=3,
        d_model=8,
        noise_std=0.05,
        device="cpu",
    )


def test_exp279_challenge_world_uses_preregistered_seed_directly_without_development_rehash() -> None:
    batch = _build(challenge_seed=0x1234ABCD, stratum="MIXED_RESIDUAL")
    assert batch.replicate == 400
    assert batch.stratum == "MIXED_RESIDUAL"
    assert batch.rng_stream == "challenge"
    assert batch.metadata["scope"] == "post-freeze-exp279-confirmatory-challenge"
    assert batch.metadata["seed"] == 0x1234ABCD
    assert batch.metadata["challenge_seed"] == 0x1234ABCD
    assert batch.metadata["rng_stream"] == "challenge"
    assert batch.metadata["stratum"] == "MIXED_RESIDUAL"
    assert "root_seed" not in batch.metadata


def test_exp279_challenge_world_is_deterministic_and_binds_seed_and_stratum() -> None:
    first = _build(challenge_seed=777, stratum="PROPAGATION_FIT")
    same = _build(challenge_seed=777, stratum="PROPAGATION_FIT")
    other_seed = _build(challenge_seed=778, stratum="PROPAGATION_FIT")
    other_stratum = _build(challenge_seed=777, stratum="BRANCH_FIT")

    assert first.digest == same.digest
    assert first.digest != other_seed.digest
    assert first.digest != other_stratum.digest
    assert first.metadata == same.metadata


def test_exp279_challenge_world_preserves_frozen_geometry_and_information_surface() -> None:
    batch = _build()
    assert tuple(batch.surface_events.shape) == (3, 4, 8)
    assert tuple(batch.variable_states.shape) == (3, 5, 8)
    assert tuple(batch.incidence.shape) == (3, 3, 5)
    assert tuple(batch.targets.shape) == (3, 5)
    assert batch.metadata["predeclared_structure_fit_strata"] == [
        "PROPAGATION_FIT",
        "BRANCH_FIT",
        "MIXED_RESIDUAL",
    ]
    assert batch.metadata["batch_size"] == 3
    assert batch.metadata["timesteps"] == 4
    assert batch.metadata["variables"] == 5
    assert batch.metadata["constraints"] == 3
    assert batch.metadata["d_model"] == 8
    assert batch.metadata["noise_std"] == pytest.approx(0.05)


def test_exp279_challenge_world_rejects_invalid_seed_stratum_or_geometry() -> None:
    from nolane_ai.experiments.exp279_challenge_worlds import build_exp279_challenge_batch

    common = dict(
        replicate=400,
        batch_size=3,
        timesteps=4,
        variables=5,
        constraints=3,
        d_model=8,
        noise_std=0.05,
        device="cpu",
    )
    with pytest.raises(ValueError, match="challenge_seed"):
        build_exp279_challenge_batch(challenge_seed=-1, stratum="PROPAGATION_FIT", **common)
    with pytest.raises(ValueError, match="stratum"):
        build_exp279_challenge_batch(challenge_seed=1, stratum="POSTHOC", **common)
    with pytest.raises(ValueError, match="geometry|constraints"):
        build_exp279_challenge_batch(
            challenge_seed=1,
            stratum="PROPAGATION_FIT",
            **{**common, "constraints": 6},
        )
