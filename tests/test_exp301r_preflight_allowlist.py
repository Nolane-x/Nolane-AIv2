from nolane_ai.experiments.exp301r_recovery import validate_recovery_changed_paths


def test_recovery_allowlist_includes_recovery_freeze_verifier() -> None:
    path = "scripts/verify_exp301r_freeze.py"
    assert validate_recovery_changed_paths((path,)) == (path,)
