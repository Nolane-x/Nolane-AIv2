import pytest

torch = pytest.importorskip("torch")


def test_exp282_partial_observability_worlds_are_deterministic_and_stream_separated():
    from nolane_ai.experiments.exp282_partial_observability import Exp282PartialObservabilityGenerator

    generator = Exp282PartialObservabilityGenerator(root_seed="exp282-worlds")
    train_a = generator.make_batch(replicate=7, batch_size=3, timesteps=5, variables=4, d_model=16, visibility_rate=0.4, noise_std=0.5, rng_stream="augmentation")
    train_b = generator.make_batch(replicate=7, batch_size=3, timesteps=5, variables=4, d_model=16, visibility_rate=0.4, noise_std=0.5, rng_stream="augmentation")
    heldout = generator.make_batch(replicate=7, batch_size=3, timesteps=5, variables=4, d_model=16, visibility_rate=0.4, noise_std=0.5, rng_stream="evaluation")
    assert train_a.digest == train_b.digest
    assert torch.equal(train_a.targets, train_b.targets)
    assert torch.equal(train_a.observations, train_b.observations)
    assert torch.equal(train_a.targets, heldout.targets)
    assert train_a.digest != heldout.digest
    assert not torch.equal(train_a.observations, heldout.observations)
    assert train_a.metadata["rng_stream"] == "augmentation"
    assert heldout.metadata["rng_stream"] == "evaluation"


def test_exp282_partial_observability_batch_exposes_visibility_and_exact_shapes():
    from nolane_ai.experiments.exp282_partial_observability import Exp282PartialObservabilityGenerator

    batch = Exp282PartialObservabilityGenerator(root_seed="shape").make_batch(replicate=1, batch_size=2, timesteps=6, variables=3, d_model=8, visibility_rate=0.5, noise_std=0.25, rng_stream="evaluation")
    assert batch.observations.shape == (2, 6, 3, 8)
    assert batch.targets.shape == (2, 3)
    assert batch.visibility_mask.shape == (2, 6, 3)
    assert batch.visibility_mask.dtype == torch.bool
    assert 0 < int(batch.visibility_mask.sum()) < batch.visibility_mask.numel()
    assert batch.metadata["scope"] == "synthetic-exp282-partial-observability-development"
    assert batch.metadata["latent_seed"] != batch.metadata["observation_seed"]


def test_exp282_partial_observability_rejects_invalid_visibility_or_stream():
    from nolane_ai.experiments.exp282_partial_observability import Exp282PartialObservabilityGenerator

    generator = Exp282PartialObservabilityGenerator(root_seed="invalid")
    kwargs = dict(replicate=0, batch_size=2, timesteps=3, variables=2, d_model=4, noise_std=0.5)
    with pytest.raises(ValueError, match="visibility_rate"):
        generator.make_batch(**kwargs, visibility_rate=0.0, rng_stream="evaluation")
    with pytest.raises(ValueError, match="rng_stream"):
        generator.make_batch(**kwargs, visibility_rate=0.5, rng_stream="controller_noise")
