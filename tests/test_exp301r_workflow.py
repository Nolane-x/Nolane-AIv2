from pathlib import Path


WORKFLOW = Path(".github/workflows/exp301r-standard-runner-sharded-recovery.yml")


def test_exp301r_workflow_is_manual_cpu_standard_runner_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "EXP301_DEVICE: cpu" in text
    assert 'EXP301R_SHARD_COUNT: "16"' in text
    assert "run_attempt" not in text
    for forbidden in ("--lr", "--loops", "--threshold", "--sample-count", "--task-weight", "--bootstrap-seed"):
        assert forbidden not in text


def test_exp301r_workflow_requires_both_original_and_recovery_freezes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python scripts/verify_exp301_freeze.py" in text
    assert "python scripts/verify_exp301r_recovery.py" in text
    assert "python scripts/verify_exp301r_freeze.py" in text
    assert "tests/test_exp301r_identity.py" in text
    assert "tests/test_exp301r_freeze.py" in text


def test_exp301r_workflow_has_exact_sharded_topology_and_frozen_reducer() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "root: [0, 1, 2, 3]" in text
    assert "arm: [A_FIXED, B_LOOP_SIMPLE, C_NRS_CORE]" in text
    assert "trial: [0, 1]" in text
    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]" in text
    assert "python scripts/exp301r_train_trial.py" in text
    assert "python scripts/exp301r_select_root.py" in text
    assert "python scripts/exp301r_materialize_challenge.py" in text
    assert "python scripts/exp301r_predict_shard.py" in text
    assert "python scripts/exp301r_seal_root.py" in text
    assert "python scripts/reduce_exp301.py" in text
    assert "python scripts/exp301r_finalize.py" in text


def test_exp301r_workflow_reuses_one_run_id_beacon_across_retries() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "EXP301R_RUN_ID: ${{ github.run_id }}" in text
    assert "EXP301R_WORKFLOW_DIGEST" in text
    assert "sha256sum .github/workflows/exp301r-standard-runner-sharded-recovery.yml" in text
