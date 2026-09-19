from pathlib import Path


def test_runtime_enforces_all_31_targets_and_projection_geometry():
    text = Path("src/nolane_ai/experiments/exp335_runtime.py").read_text()
    assert 'len(target_worlds) != 31' in text
    assert 'if target_id != source_id' in text
    assert "torch.linalg.pinv" in text
    assert "PINV_RTOL" in text
    assert "TARGET_NORM_SQUARED_FLOOR" in text
    assert "rng_after_source" in text
    assert "torch.set_rng_state(rng_after_source)" in text


def test_runtime_binds_continuation_receipts_fail_closed():
    text = Path("src/nolane_ai/experiments/exp335_runtime.py").read_text()
    for phrase in (
        '"parent_artifact_digest"',
        '"source_tree_digest"',
        '"preregistration_digest"',
        '"data_order_digest"',
        '"checkpoint_sha256"',
        '"receipt_digest"',
        "duplicate/skipped chain position",
        "SHAM_FULL32_MISMATCH",
    ):
        assert phrase in text


def test_chunk_script_runs_all_arms_in_one_process():
    text = Path("src/nolane_ai/experiments/exp335_runtime.py").read_text()
    assert "for arm in ARMS:" in text
    assert "sham_equivalent(" in text
    assert 'boundaries["CONTROL_FULL32"]' in text
    assert 'boundaries["SHAM_MEASURE_FULL32"]' in text


def test_sham_elides_unused_projection_solve_but_keeps_all_raw_dots():
    text = Path("src/nolane_ai/experiments/exp335_runtime.py").read_text()
    assert "if project:\n        projected, selected, raw_dots, post_dots = _project_source(source_grads, targets)" in text
    assert "else:\n        raw_dots = {world_id: _dot(source_grads, grad) for world_id, grad in targets}" in text
    assert "selected = []" in text
    assert "post_dots = {}" in text
    assert "applied = source_grads" in text


def test_sham_streams_target_dots_without_retaining_all_gradients():
    text = Path("src/nolane_ai/experiments/exp335_runtime.py").read_text()
    assert "if project:\n            targets.append((target_id, target_grads))" in text
    assert "else:\n            raw_dots[target_id] = _dot(source_grads, target_grads)" in text
    assert "raw_dots = {world_id: _dot(source_grads, grad) for world_id, grad in targets}" not in text
