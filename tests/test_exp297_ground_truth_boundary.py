import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_fidelity_worlds import generate_fidelity_world
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development


def test_exp297_arm_views_never_expose_evaluator_truth_or_stratum():
    batch = generate_fidelity_world(297777)
    for case in batch.candidates:
        arm_view = case.arm_view()
        assert "is_faithful" not in arm_view
        assert "stratum" not in arm_view
        assert arm_view["candidate_digest"] == case.candidate_digest


def test_exp297_execution_receipts_attest_no_truth_or_trap_leakage_to_arms():
    artifact = run_exp297_paired_development(
        root_seed="exp297-ground-truth-boundary",
        eval_replicates=1,
        eval_start_replicate=700,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    for row in artifact["evaluation"]["raw_candidates"]:
        receipt = row["arm_input_receipt"]
        assert receipt["candidate_set_frozen_before_arms"] is True
        assert receipt["byte_identical_candidate_order"] is True
        assert receipt["compile_only_candidate_digest"] == row["candidate_digest"]
        assert receipt["fidelity_court_candidate_digest"] == row["candidate_digest"]
        assert receipt["evaluator_truth_in_causal_path"] is False
        assert receipt["trap_family_in_causal_path"] is False
