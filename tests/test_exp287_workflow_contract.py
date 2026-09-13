from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "exp287-learned-conflict-localization-ci.yml"
DEVELOPMENT = ROOT / ".github" / "workflows" / "exp287-learned-conflict-localization-development.yml"
FREEZE_GUARD = ROOT / ".github" / "workflows" / "exp287-learned-conflict-localization-freeze-guard.yml"
MARKER = ROOT / "protocols" / "exp287-learned-conflict-localization-release.lock"
MARKER_REL = "protocols/exp287-learned-conflict-localization-release.lock"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dedicated_contract_ci_covers_all_exp287_surfaces_without_write_authority() -> None:
    text = _text(CI)
    assert "name: exp287-learned-conflict-localization-ci" in text
    assert "pull_request:" in text
    assert "src/nolane_ai/experiments/exp287_*.py" in text
    assert "src/nolane_ai/experiments/matched_conflict_localizer_arms.py" in text
    assert "scripts/*exp287*.py" in text
    assert "tests/test_exp287_*.py" in text
    assert "protocols/exp287_development_v1.json" in text
    assert "permissions:" in text
    assert "contents: read" in text
    assert "workflow_dispatch" not in text


def test_development_workflow_is_marker_only_and_has_exact_head_preflight() -> None:
    text = _text(DEVELOPMENT)
    assert "name: exp287-learned-conflict-localization-development" in text
    assert "workflow_dispatch" not in text
    assert "pull_request:" in text
    assert MARKER_REL in text
    assert "github.event.pull_request.head.sha" in text
    assert "LAST_TOUCH" in text
    assert 'test "$LAST_TOUCH" = "${{ github.event.pull_request.head.sha }}"' in text
    assert "EXP287_LEARNED_CONFLICT_LOCALIZATION_RELEASE_V1" in text
    assert "canonical_indices=0,1,2,3" in text
    assert "train_replicates=64" in text
    assert "eval_replicates=32" in text
    assert "eval_start_replicate=10000" in text
    assert "top_k=2" in text
    assert "capture_threshold=0.5" in text
    assert "precision_threshold=0.5" in text
    assert "root_prefix=20260913-exp287-learned-conflict-localization-v1-dev" in text
    assert "protocol_digest=c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440" in text
    assert "python scripts/verify_protocol.py" in text


def test_development_preflight_allows_only_the_first_authoritative_run_before_data_jobs() -> None:
    text = _text(DEVELOPMENT)
    assert "actions: read" in text
    assert "EXP287_DEVELOPMENT_WORKFLOW_NAME" in text
    assert 'current_run_id = int(os.environ["GITHUB_RUN_ID"])' in text
    assert "candidates.sort" in text
    assert 'authoritative = int(candidates[0]["id"])' in text
    assert "current_run_id != authoritative" in text
    assert "duplicate DEVELOPMENT run blocked before root execution" in text
    assert text.index("duplicate DEVELOPMENT run blocked before root execution") < text.index("root:")


def test_development_blocks_same_run_reruns_and_serializes_duplicate_run_ids() -> None:
    text = _text(DEVELOPMENT)
    assert "concurrency:" in text
    assert "exp287-development-pr-${{ github.event.pull_request.number }}" in text
    assert "cancel-in-progress: false" in text
    assert 'current_run_attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])' in text
    assert "current_run_attempt != 1" in text
    assert "rerun DEVELOPMENT attempt blocked before root execution" in text
    assert text.index("rerun DEVELOPMENT attempt blocked before root execution") < text.index("root:")


def test_development_workflow_has_exact_four_root_jobs_then_one_cross_reducer() -> None:
    text = _text(DEVELOPMENT)
    assert "root:" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "scripts/run_exp287_learned_conflict_localization_dev.py" in text
    assert "cross:" in text
    assert "needs: root" in text
    assert "scripts/reduce_exp287_learned_conflict_localization_dev.py" in text
    assert "pattern: exp287-root-*-${{ github.sha }}" in text
    assert "receipt.json.sha256" in text
    assert "retention-days:" in text
    assert "if-no-files-found: error" in text


def test_development_workflow_has_no_scientific_or_secret_escape_hatch() -> None:
    text = _text(DEVELOPMENT)
    forbidden = (
        "secrets.",
        "workflow_dispatch",
        "challenge_seed",
        "derive_challenge_seed",
        "beacon_pulse",
        "scientific_evidence_eligible = true",
        'promotion_claimed"] = True',
        "promotion_claimed: true",
        "actions: write",
    )
    for token in forbidden:
        assert token not in text
    assert "contents: read" in text


def test_freeze_guard_is_read_only_and_fails_closed_on_duplicate_development_runs() -> None:
    text = _text(FREEZE_GUARD)
    assert "name: exp287-learned-conflict-localization-freeze-guard" in text
    assert "pull_request:" in text
    assert "actions: read" in text
    assert "actions: write" not in text
    assert "exp287-learned-conflict-localization-development" in text
    assert "candidates.sort" in text
    assert 'authoritative = int(candidates[0]["id"]) if candidates else None' in text
    assert "authoritative_development_run_id" in text
    assert "duplicate_development_run_ids" in text
    assert "observed_development_run_ids" in text
    assert "duplicate DEVELOPMENT runs detected" in text
    assert "/cancel" not in text


def test_release_marker_is_absent_before_predata_green() -> None:
    assert not MARKER.exists(), "EXP-287 release marker must not exist before all pre-data gates are GREEN"
