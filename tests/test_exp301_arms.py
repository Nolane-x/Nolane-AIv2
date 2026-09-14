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


FROZEN_EFFORT_MULTIPLIERS = (1, 2, 4, 8, 12, 16)


def test_exp301_arm_ids_are_frozen() -> None:
    assert tuple(arm.value for arm in Exp301ArmId) == (
        "A_FIXED",
        "B_LOOP_SIMPLE",
        "C_NRS_CORE",
    )


def test_fixed_restart_baseline_is_exact_10m_and_geometry_stays_fixed() -> None:
    for effort_multiplier in FROZEN_EFFORT_MULTIPLIERS:
        compiled = compile_fixed_frontier_point(
            loop_budget=effort_multiplier,
            device="meta",
        )

        assert compiled.arm_id is Exp301ArmId.A_FIXED
        assert compiled.receipt.loop_budget == effort_multiplier
        assert compiled.receipt.depth == 3
        assert compiled.receipt.d_model == 448
        assert compiled.receipt.head_dim == 64
        assert compiled.receipt.n_heads == 7
        assert compiled.receipt.d_ff == 1_152
        assert compiled.receipt.capacity_bottleneck == 981
        assert compiled.receipt.capacity_tail_parameters == 192
        assert compiled.receipt.weight_tied_across_depth is False
        assert compiled.receipt.loop_conditioning is False
        assert compiled.receipt.stateless_restarts is True
        assert compiled.receipt.latent_state_carry is False
        assert compiled.receipt.trainable_parameters == 10_000_000
        assert count_trainable_parameters(compiled.model) == 10_000_000
        assert len(compiled.model.backbone.layers) == 3
        assert len({id(layer) for layer in compiled.model.backbone.layers}) == 3


def test_fixed_restart_baseline_rejects_unregistered_effort_multiplier() -> None:
    with pytest.raises(ValueError, match="loop_budget"):
        compile_fixed_frontier_point(loop_budget=3, device="meta")


def test_recurrent_rivals_are_matched_and_state_carry_is_explicit() -> None:
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
    assert simple.receipt.stateless_restarts is False
    assert nrs.receipt.stateless_restarts is False
    assert simple.receipt.latent_state_carry is True
    assert nrs.receipt.latent_state_carry is True
    assert simple.receipt.d_model == nrs.receipt.d_model == 448
    assert simple.receipt.d_ff == nrs.receipt.d_ff == 1_152


def test_fixed_restart_runtime_accepts_only_frozen_effort_counts() -> None:
    compiled = compile_fixed_frontier_point(loop_budget=2)
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    logits = compiled.model(tokens, restarts=2)
    assert logits.shape == (1, 3, 4_608)

    with pytest.raises(ValueError, match="restarts"):
        compiled.model(tokens, restarts=0)


def test_fixed_depth_layers_do_not_share_parameter_objects() -> None:
    compiled = compile_fixed_frontier_point(loop_budget=4)
    parameters_by_layer = [set(map(id, layer.parameters())) for layer in compiled.model.backbone.layers]

    for index, left in enumerate(parameters_by_layer):
        for right in parameters_by_layer[index + 1 :]:
            assert left.isdisjoint(right)
