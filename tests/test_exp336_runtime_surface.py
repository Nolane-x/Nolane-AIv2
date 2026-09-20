from pathlib import Path


def test_runtime_uses_selected_cnrs_parent_and_all_31_targets():
    text = Path("src/nolane_ai/experiments/exp336_runtime.py").read_text()
    for phrase in (
        'build_scientific_arm("C_NRS_CORE"',
        "load_checkpoint_bundle",
        "PARENT_CNRS_CHECKPOINT_SHA256",
        "PARENT_CNRS_MODEL_STATE_DIGEST",
        "PARENT_CNRS_OPTIMIZER_STATE_DIGEST",
        "PARENT_CNRS_RNG_STATE_DIGEST",
        'len(target_worlds) != 31',
        'if target_id != source_id',
    ):
        assert phrase in text
    assert 'build_scientific_arm("A_FIXED"' not in text


def test_runtime_preserves_exact_inherited_projection_and_rng_geometry():
    text = Path("src/nolane_ai/experiments/exp336_runtime.py").read_text()
    for phrase in (
        "torch.linalg.pinv",
        "PINV_RTOL",
        "TARGET_NORM_SQUARED_FLOOR",
        "rng_after_source",
        "torch.set_rng_state(rng_after_source)",
        "_backward_grads(compiled, target_world, effort)",
    ):
        assert phrase in text


def test_runtime_has_aggregate_stage_a_gate_and_fail_closed_chain():
    text = Path("src/nolane_ai/experiments/exp336_runtime.py").read_text()
    for phrase in (
        "evaluate_snapshot(",
        'stage="A_SANITY"',
        "ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS",
        '"aggregate_metrics"',
        '"parent_artifact_digest"',
        '"receipt_digest"',
        "duplicate/skipped chain position",
        "SHAM_CNRS_FULL32_MISMATCH",
    ):
        assert phrase in text
