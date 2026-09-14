from scripts.verify_exp301r_recovery import recovery_source_paths_for_validation


MARKER_PATHS = (
    "protocols/v017/exp301r_execution_identity_v1.json",
    "protocols/v017/exp301r_execution_identity_v1.sha256",
)


def test_recovery_preflight_excludes_only_marker_and_freeze_verifier_paths() -> None:
    changed = (
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "scripts/verify_exp301r_freeze.py",
        *MARKER_PATHS,
        "README.md",
    )
    assert recovery_source_paths_for_validation(changed) == (
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "README.md",
    )
