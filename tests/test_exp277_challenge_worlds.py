from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")


GEOMETRY = dict(
    batch_size=3,
    timesteps=4,
    variables=6,
    constraints=3,
    d_model=8,
    noise_std=0.05,
)


def _make(*, challenge_seed: int = 123456789, replicate: int = 50000):
    from nolane_ai.experiments.exp277_challenge_worlds import build_exp277_challenge_batch

    return build_exp277_challenge_batch(
        challenge_seed=challenge_seed,
        replicate=replicate,
        **GEOMETRY,
    )


def test_exp277_challenge_regenerates_exactly_from_seed_and_replicate() -> None:
    import torch

    first = _make()
    second = _make()
    assert first.schema == "NLM-EXP-277-POST-FREEZE-CHALLENGE-BATCH-V1"
    assert first.scope == "POST_FREEZE_CHALLENGE"
    assert first.experiment_id == "EXP-277"
    assert first.replicate == 50000
    assert first.digest == second.digest
    assert torch.equal(first.surface_events, second.surface_events)
    assert torch.equal(first.variable_states, second.variable_states)
    assert torch.equal(first.oracle_incidence, second.oracle_incidence)
    assert torch.equal(first.targets, second.targets)
    assert first.arm_visible_metadata == second.arm_visible_metadata


def test_exp277_challenge_digest_changes_with_seed_replicate_or_geometry() -> None:
    assert _make(challenge_seed=123456790).digest != _make().digest
    assert _make(replicate=50001).digest != _make().digest
    changed = dict(GEOMETRY)
    changed["variables"] = 7
    from nolane_ai.experiments.exp277_challenge_worlds import build_exp277_challenge_batch

    assert build_exp277_challenge_batch(challenge_seed=123456789, replicate=50000, **changed).digest != _make().digest


def test_exp277_challenge_has_fixed_closed_shapes_and_structure_semantics() -> None:
    import torch

    batch = _make()
    assert batch.surface_events.shape == (3, 4, 8)
    assert batch.variable_states.shape == (3, 6, 8)
    assert batch.oracle_incidence.shape == (3, 3, 6)
    assert batch.targets.shape == (3, 6)
    assert set(batch.targets.flatten().tolist()) <= {0, 1}
    assert torch.all(batch.oracle_incidence.sum(dim=1) == 1)
    assert batch.metadata["surface_semantics"] == "uncompiled component/anchor events"
    assert batch.metadata["oracle_semantics"] == "ground-truth component-variable incidence"
    assert batch.metadata["scope"] == "POST_FREEZE_CHALLENGE"


def test_exp277_challenge_arm_visible_view_excludes_truth_and_oracle_structure() -> None:
    batch = _make()
    visible = batch.arm_visible()
    assert set(visible) == {"surface_events", "variable_states", "metadata"}
    assert "targets" not in visible
    assert "oracle_incidence" not in visible
    assert "challenge_seed" not in visible["metadata"]
    assert "seed" not in visible["metadata"]
    serialized = repr(visible["metadata"]).lower()
    assert "target" not in serialized
    assert "oracle_incidence" not in serialized
    assert "anchor" not in serialized


def test_exp277_challenge_builder_does_not_expose_development_rng_api() -> None:
    from nolane_ai.experiments.exp277_challenge_worlds import build_exp277_challenge_batch

    signature = inspect.signature(build_exp277_challenge_batch)
    assert "root_seed" not in signature.parameters
    assert "rng_stream" not in signature.parameters
    assert "scope" not in signature.parameters
    assert set(signature.parameters) == {
        "challenge_seed",
        "replicate",
        "batch_size",
        "timesteps",
        "variables",
        "constraints",
        "d_model",
        "noise_std",
        "device",
    }


def test_exp277_challenge_rejects_invalid_seed_replicate_and_geometry() -> None:
    from nolane_ai.experiments.exp277_challenge_worlds import build_exp277_challenge_batch

    with pytest.raises(ValueError, match="challenge_seed"):
        build_exp277_challenge_batch(challenge_seed=-1, replicate=50000, **GEOMETRY)
    with pytest.raises(ValueError, match="replicate"):
        build_exp277_challenge_batch(challenge_seed=1, replicate=-1, **GEOMETRY)
    with pytest.raises(ValueError, match="constraints cannot exceed variables"):
        build_exp277_challenge_batch(
            challenge_seed=1,
            replicate=50000,
            **(GEOMETRY | {"constraints": 7}),
        )
    with pytest.raises(ValueError, match="timesteps"):
        build_exp277_challenge_batch(
            challenge_seed=1,
            replicate=50000,
            **(GEOMETRY | {"timesteps": 2}),
        )


def test_exp277_challenge_metadata_binds_seed_without_exposing_raw_seed_to_arms() -> None:
    batch = _make(challenge_seed=42)
    assert batch.metadata["challenge_seed_digest"]
    assert len(batch.metadata["challenge_seed_digest"]) == 64
    assert batch.metadata["challenge_seed_digest"] != f"{42:064x}"
    assert "challenge_seed" not in batch.arm_visible_metadata
    assert batch.arm_visible_metadata["challenge_seed_digest"] == batch.metadata["challenge_seed_digest"]
