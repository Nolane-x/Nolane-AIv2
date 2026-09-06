from importlib import import_module

import pytest


torch = pytest.importorskip("torch")


def _model_modules():
    config = import_module("nolane_ai.model.config")
    model = import_module("nolane_ai.model.nlm")
    assert hasattr(config, "NLMConfig")
    assert hasattr(model, "NolaneLivingModel")
    return config, model


def test_authoritative_model_has_exactly_100m_parameters_on_meta_device():
    config, model = _model_modules()
    candidate = model.NolaneLivingModel(config.NLMConfig.authoritative_100m(), device="meta")
    assert sum(p.numel() for p in candidate.parameters()) == 100_000_000
    assert sum(p.numel() for p in candidate.parameters() if p.requires_grad) == 90_000_000


def test_tiny_model_forward_shape():
    config, model = _model_modules()
    candidate = model.NolaneLivingModel(config.NLMConfig.tiny_for_tests(), device="cpu")
    tokens = torch.randint(0, candidate.config.vocab_size, (2, 5))
    logits = candidate(tokens)
    assert logits.shape == (2, 5, candidate.config.vocab_size)
