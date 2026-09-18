from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.v017_recursive import (
    ActiveCapacityExchange,
    LoopConditioner,
    V017CoreBackbone,
    V017Geometry,
    build_nrs_core_10m,
    build_simple_recurrent_10m,
    count_trainable_parameters,
)


def test_v017_default_geometry_reconstructs_frozen_base_count() -> None:
    geometry = V017Geometry()
    backbone = V017CoreBackbone(geometry=geometry, device="meta")

    assert geometry.vocab_size == 4_608
    assert geometry.d_model == 448
    assert geometry.n_heads == 7
    assert geometry.head_dim == 64
    assert geometry.shared_layers == 3
    assert geometry.d_ff == 1_152
    assert geometry.max_loops == 16
    assert count_trainable_parameters(backbone) == 9_120_832


def test_capacity_exchange_is_exact_active_capacity_not_dead_padding() -> None:
    exchange = ActiveCapacityExchange(
        d_model=448,
        bottleneck=981,
        tail_parameters=192,
    )
    assert count_trainable_parameters(exchange) == 879_168
    assert not any(
        forbidden in name.lower()
        for name, _ in exchange.named_parameters()
        for forbidden in ("padding", "filler", "reserve")
    )

    x = torch.randn(2, 3, 448, requires_grad=True)
    output = exchange(x)
    assert output.shape == x.shape
    assert not torch.equal(output, x)
    output.square().mean().backward()

    for name, parameter in exchange.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_loop_conditioner_is_bounded_and_all_rows_are_reachable() -> None:
    conditioner = LoopConditioner(d_model=448, max_loops=16)
    assert count_trainable_parameters(conditioner) == 16 * 448

    x = torch.randn(16, 1, 448, requires_grad=True)
    loop_indices = torch.arange(16)
    conditioned = conditioner(x, loop_indices=loop_indices)
    conditioned.square().mean().backward()

    assert conditioner.embedding.weight.grad is not None
    assert torch.count_nonzero(conditioner.embedding.weight.grad).item() == 16 * 448

    with pytest.raises(ValueError, match="loop"):
        conditioner(torch.randn(1, 1, 448), loop_indices=torch.tensor([16]))


def test_simple_and_nrs_arms_are_exactly_ten_million_with_capacity_exchange() -> None:
    simple = build_simple_recurrent_10m(device="meta")
    nrs = build_nrs_core_10m(device="meta")

    assert count_trainable_parameters(simple) == 10_000_000
    assert count_trainable_parameters(nrs) == 10_000_000

    assert simple.loop_conditioner is None
    assert simple.capacity_exchange.bottleneck == 981
    assert simple.capacity_exchange.tail_parameters == 192

    assert nrs.loop_conditioner is not None
    assert nrs.capacity_exchange.bottleneck == 973
    assert nrs.capacity_exchange.tail_parameters == 192
    assert count_trainable_parameters(nrs.loop_conditioner) == 7_168


def test_recurrent_forward_uses_tied_output_weight_and_rejects_invalid_loop_count() -> None:
    model = build_nrs_core_10m()
    assert model.output_weight is model.backbone.token_embedding.weight

    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits_one = model(tokens, loops=1)
    logits_two = model(tokens, loops=2)

    assert logits_one.shape == (1, 3, 4_608)
    assert logits_two.shape == logits_one.shape
    assert not torch.equal(logits_one, logits_two)

    with pytest.raises(ValueError, match="loops"):
        model(tokens, loops=0)
    with pytest.raises(ValueError, match="loops"):
        model(tokens, loops=17)
