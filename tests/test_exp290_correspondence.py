from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_exp290_model_closes_exact_parameter_budget() -> None:
    from nolane_ai.experiments.exp290_correspondence import Exp290CorrespondenceModel

    model = Exp290CorrespondenceModel(
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
        device="cpu",
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 500000
    functional = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "capacity_reserve" not in name
    )
    assert 0 < functional < 500000


def test_exp290_embeddings_are_normalized_and_shapes_match() -> None:
    from nolane_ai.experiments.exp290_correspondence import Exp290CorrespondenceModel

    model = Exp290CorrespondenceModel(64, 48, 500000, device="cpu")
    source_events = torch.randn(2, 4, 64)
    target_events = torch.randn(2, 4, 64)
    source_variables = torch.randn(2, 8, 64)
    target_variables = torch.randn(2, 8, 64)
    output = model(source_events, target_events, source_variables, target_variables)

    assert output.source_embeddings.shape == (2, 8, 48)
    assert output.target_embeddings.shape == (2, 8, 48)
    assert output.source_to_target_similarity.shape == (2, 8, 8)
    assert output.target_to_source_similarity.shape == (2, 8, 8)
    assert torch.allclose(
        output.source_embeddings.norm(dim=-1),
        torch.ones(2, 8),
        atol=1e-5,
    )
    assert torch.allclose(
        output.target_embeddings.norm(dim=-1),
        torch.ones(2, 8),
        atol=1e-5,
    )


def test_exp290_top1_tie_break_is_lowest_target_index() -> None:
    from nolane_ai.experiments.exp290_correspondence import learned_row_top1_mapping

    similarity = torch.tensor(
        [[[1.0, 1.0, 0.0], [0.0, 2.0, 2.0]]],
        dtype=torch.float32,
    )
    mapping = learned_row_top1_mapping(similarity)
    assert mapping.tolist() == [[0, 1]]


def test_exp290_bidirectional_loss_has_fixed_equal_weights() -> None:
    from nolane_ai.experiments.exp290_correspondence import exp290_correspondence_loss

    source_to_target = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0]]],
        requires_grad=True,
    )
    target_to_source = source_to_target.transpose(1, 2).contiguous()
    labels = torch.tensor([[0, 1]], dtype=torch.long)
    inverse = torch.tensor([[0, 1]], dtype=torch.long)
    loss, audit = exp290_correspondence_loss(
        source_to_target,
        target_to_source,
        source_to_target_labels=labels,
        target_to_source_labels=inverse,
    )

    assert torch.isfinite(loss)
    assert audit["loss_weights"] == {
        "source_to_target_ce": 0.5,
        "target_to_source_ce": 0.5,
    }
    assert audit["calibration_used"] is False
    assert audit["early_stopping"] is False
    assert audit["hard_negative_mining"] is False


def test_exp290_model_state_digest_is_deterministic_and_scalar_safe() -> None:
    from nolane_ai.experiments.exp290_correspondence import (
        Exp290CorrespondenceModel,
        exp290_model_state_digest,
    )

    torch.manual_seed(1234)
    model = Exp290CorrespondenceModel(64, 48, 500000, device="cpu")
    first = exp290_model_state_digest(model)
    second = exp290_model_state_digest(model)
    assert first == second
    assert len(first) == 64
    int(first, 16)


def test_exp290_training_receipt_is_augmentation_only() -> None:
    from nolane_ai.experiments.exp290_correspondence import training_contract_receipt

    receipt = training_contract_receipt()
    assert receipt["rng_stream"] == "augmentation"
    assert receipt["evaluation_lineage_consumed"] is False
    assert receipt["evaluation_correspondence_used_for_training"] is False
    assert receipt["loss_weights"] == {
        "source_to_target_ce": 0.5,
        "target_to_source_ce": 0.5,
    }
