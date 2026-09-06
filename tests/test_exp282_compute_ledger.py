import pytest

torch = pytest.importorskip("torch")


def test_exp282_pair_has_identical_full_primitive_compute_signature():
    from nolane_ai.experiments.matched_belief_arms import (
        account_matched_belief_arm_pair,
        build_matched_belief_arm_pair,
    )

    recurrent, explicit = build_matched_belief_arm_pair(d_model=16, hidden_size=12, target_parameters=10_000)
    audit = account_matched_belief_arm_pair(recurrent, explicit, timesteps=5, variables=4)
    assert audit["schema"] == "NLM-EXP-282-COMPUTE-LEDGER-V1"
    assert audit["recurrent_hidden"]["primitive_counts"] == audit["explicit_belief"]["primitive_counts"]
    assert audit["primitive_operation_match"] is True
    assert audit["accounted_flops_match"] is True
    assert audit["relative_accounted_flop_difference"] == pytest.approx(0.0)
    assert audit["recurrent_hidden"]["accounted_flops_per_episode"] > 0


def test_exp282_compute_ledger_scales_with_exact_sequence_geometry():
    from nolane_ai.experiments.matched_belief_arms import account_matched_belief_arm_pair, build_matched_belief_arm_pair

    recurrent, explicit = build_matched_belief_arm_pair(d_model=8, hidden_size=6, target_parameters=5_000)
    one = account_matched_belief_arm_pair(recurrent, explicit, timesteps=1, variables=2)
    five = account_matched_belief_arm_pair(recurrent, explicit, timesteps=5, variables=2)
    assert one["geometry"] == {"timesteps": 1, "variables": 2}
    assert five["geometry"] == {"timesteps": 5, "variables": 2}
    assert five["recurrent_hidden"]["primitive_counts"]["linear_multiply"] == 5 * one["recurrent_hidden"]["primitive_counts"]["linear_multiply"]
    assert five["recurrent_hidden"]["primitive_counts"]["decision_head_calls"] == 5 * one["recurrent_hidden"]["primitive_counts"]["decision_head_calls"]


def test_exp282_compute_ledger_rejects_invalid_geometry():
    from nolane_ai.experiments.matched_belief_arms import account_matched_belief_arm_pair, build_matched_belief_arm_pair

    recurrent, explicit = build_matched_belief_arm_pair(d_model=8, hidden_size=6, target_parameters=5_000)
    with pytest.raises(ValueError, match="timesteps and variables"):
        account_matched_belief_arm_pair(recurrent, explicit, timesteps=0, variables=2)
