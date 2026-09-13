from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_arms import (
    Exp301ArmId,
    build_nrs_core_arm,
    build_simple_recurrent_arm,
    compile_fixed_frontier_point,
)
from nolane_ai.model.v017_recursive import count_trainable_parameters


EXPECTED_FIXED_GEOMETRY = {
    1: (3, 448, 1_344, 117, 192),
    2: (6, 320, 1_024, 258, 320),
    4: (12, 192, 768, 5_293, 64),
    8: (24, 192, 384, 673, 64),
    12: (36, 128, 448, 3_314, 0),
    16: (48, 128, 320, 1_382, 0),
}


def test_exp301_arm_ids_are_frozen() -> None:
    assert tuple(arm.value for arm in Exp301ArmId) == (
        "A_FIXED",
        "B_LOOP_SIMPLE",
        "C_NRS_CORE",
    )


def test_fixed_frontier_geometry_is_deterministic_and_exactly_ten_million() -> None:
    for loop_budget, expected in EXPECTED_FIXED_GEOMETRY.items():
        compiled = compile_fixed_frontier_point(loop_budget=loop_budget, device="meta")
        depth, d_model, d_ff, bottleneck, tail = expected

        assert compiled.arm_id is Exp301ArmId.A_FIXED
        assert compiled.receipt.loop_budget == loop_budget
        assert compiled.receipt.depth == depth == 3 * loop_budget
        assert compiled.receipt.d_model == d_model
        assert compiled.receipt.head_dim == 64
        assert compiled.receipt.n_heads == d_model // 64
        assert compiled.receipt.d_ff == d_ff
        assert compiled.receipt.capacity_bottleneck == bottleneck
        assert compiled.receipt.capacity_tail_parameters == tail
        assert compiled.receipt.weight_tied_across_depth is False
        assert compiled.receipt.trainable_parameters == 10_000_000
        assert count_trainable_parameters(compiled.model) == 10_000_000
        assert len({id(layer) for layer in compiled.model.backbone.layers}) == depth


def test_fixed_frontier_rejects_unregistered_loop_budget() -> None:
    with pytest.raises(ValueError, match="loop_budget"):
        compile_fixed_frontier_point(loop_budget=3, device="meta")


def test_recurrent_rivals_are_matched_and_receipts_make_weight_tying_explicit() -> None:
    simple = build_simple_recurrent_arm(device="meta")
    nrs = build_nrs_core_arm(device="meta")

    assert simple.arm_id is Exp301ArmId.B_LOOP_SIMPLE
    assert nrs.arm_id is Exp301ArmId.C_NRS_CORE
    assert simple.receipt.trainable_parameters == 10_000_000
    assert nrs.receipt.trainable_parameters == 10_000_000
    assert count_trainable_parameters(simple.model) == 10_000_000
    assert count_trainable_parameters(nrs.model) == 10_000_000

    assert simple.receipt.weight_tied_across_depth is True
    assert nrs.receipt.weight_tied_across_depth is True
    assert simple.receipt.loop_conditioning is False
    assert nrs.receipt.loop_conditioning is True
    assert simple.receipt.d_model == nrs.receipt.d_model == 448
    assert simple.receipt.d_ff == nrs.receipt.d_ff == 1_152


def test_fixed_frontier_has_no_parameter_object_shared_between_depth_layers() -> None:
    compiled = compile_fixed_frontier_point(loop_budget=4)
    parameters_by_layer = [set(map(id, layer.parameters())) for layer in compiled.model.backbone.layers]

    for index, left in enumerate(parameters_by_layer):
        for right in parameters_by_layer[index + 1 :]:
            assert left.isdisjoint(right)
