from scripts.verify_exp301r_recovery import semantic_paths_for_validation


MARKER_PATHS = (
    "protocols/v017/exp301r_execution_identity_v1.json",
    "protocols/v017/exp301r_execution_identity_v1.sha256",
)


def test_semantic_isolation_excludes_only_exp301r_marker_files() -> None:
    changed = (
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        *MARKER_PATHS,
        "scripts/exp301r_train_trial.py",
    )
    assert semantic_paths_for_validation(changed) == (
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "scripts/exp301r_train_trial.py",
    )


def test_semantic_isolation_keeps_unrelated_paths_for_rejection() -> None:
    assert semantic_paths_for_validation((*MARKER_PATHS, "README.md")) == ("README.md",)
