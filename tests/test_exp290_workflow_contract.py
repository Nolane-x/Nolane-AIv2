from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "exp290-structural-clause-transfer-ci.yml"
DEVELOPMENT = ROOT / ".github" / "workflows" / "exp290-structural-clause-transfer-development.yml"
FREEZE_GUARD = ROOT / ".github" / "workflows" / "exp290-structural-clause-transfer-freeze-guard.yml"
MARKER = ROOT / "protocols" / "exp290-structural-clause-transfer-release.lock"
MARKER_REL = "protocols/exp290-structural-clause-transfer-release.lock"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dedicated_contract_ci_covers_all_exp290_surfaces_without_write_authority() -> None:
    text = _text(CI)
    assert "name: exp290-structural-clause-transfer-ci" in text
    assert "pull_request:" in text
    assert "protocols/exp290_structural_clause_transfer_v1.json" in text
    assert "src/nolane_ai/experiments/exp290_*.py" in text
    assert "src/nolane_ai/experiments/matched_clause_transfer_arms.py" in text
    assert "scripts/*exp290*.py" in text
    assert "tests/test_exp290_*.py" in text
    assert "permissions:" in text
    assert "contents: read" in text
    assert "workflow_dispatch" not in text


def test_development_workflow_is_marker_only_and_exact_head_bound() -> None:
    text = _text(DEVELOPMENT)
    assert "name: exp290-structural-clause-transfer-development" in text
    assert "workflow_dispatch" not in text
    assert "pull_request:" in text
    assert MARKER_REL in text
    assert "github.event.pull_request.head.sha" in text
    assert "LAST_TOUCH" in text
    assert 'test "$LAST_TOUCH" = "${{ github.event.pull_request.head.sha }}"' in text
    assert "EXP290_STRUCTURAL_CLAUSE_TRANSFER_RELEASE_V1" in text
    assert "canonical_indices=0,1,2,3" in text
    assert "train_replicates=64" in text
    assert "eval_replicates=32" in text
    assert "eval_start_replicate=40000" in text
    assert "batch_size=8" in text
    assert "d_model=64" in text
    assert "hidden_size=48" in text
    assert "target_parameters=500000" in text
    assert "timesteps=4" in text
    assert "variables=8" in text
    assert "decoys=3" in text
    assert "restarts=4" in text
    assert "max_search_steps=24" in text
    assert "noise_std=0.05" in text
    assert "lr=0.002" in text
    assert "weight_decay=0.0" in text
    assert "capture_threshold=0.5" in text
    assert "valid_state_overprune_rate_ceiling=0.005" in text
    assert "verified_solution_rate_floor_delta=-0.01" in text
    assert "root_prefix=20260913-exp290-structural-clause-transfer-v1-dev" in text
    assert "protocol_digest=c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440" in text
    assert "geometry_digest=bffb1f01d54a6867f3af6acd0c2339a83add69f6df4bc6fd8d5504aa9eb4d67a" in text
    assert "authority_scope=DEVELOPMENT_EV_E2_ONLY" in text
    assert "python scripts/verify_protocol.py" in text


def test_development_preflight_allows_only_first_run_and_blocks_reruns_before_data() -> None:
    text = _text(DEVELOPMENT)
    assert "actions: read" in text
    assert "EXP290_DEVELOPMENT_WORKFLOW_NAME" in text
    assert 'current_run_id = int(os.environ["GITHUB_RUN_ID"])' in text
    assert 'current_run_attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])' in text
    assert "current_run_attempt != 1" in text
    assert "rerun DEVELOPMENT attempt blocked before root execution" in text
    assert "candidates.sort" in text
    assert 'authoritative = int(candidates[0]["id"])' in text
    assert "current_run_id != authoritative" in text
    assert "duplicate DEVELOPMENT run blocked before root execution" in text
    assert text.index("rerun DEVELOPMENT attempt blocked before root execution") < text.index("root:")
    assert text.index("duplicate DEVELOPMENT run blocked before root execution") < text.index("root:")


def test_development_serializes_duplicate_runs_without_cancellation_write_authority() -> None:
    text = _text(DEVELOPMENT)
    assert "concurrency:" in text
    assert "exp290-development-pr-${{ github.event.pull_request.number }}" in text
    assert "cancel-in-progress: false" in text
    assert "actions: write" not in text
    assert "/cancel" not in text


def test_development_has_exact_four_roots_then_cross_and_verifies_code_identity() -> None:
    text = _text(DEVELOPMENT)
    assert "root:" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "scripts/run_exp290_structural_clause_transfer_dev.py" in text
    assert "cross:" in text
    assert "needs: root" in text
    assert "scripts/reduce_exp290_structural_clause_transfer_dev.py" in text
    assert "pattern: exp290-root-*-${{ github.sha }}" in text
    assert "receipt.json.sha256" in text
    assert "retention-days:" in text
    assert "if-no-files-found: error" in text
    assert 'data["code_digest"] == source_tree_digest(Path.cwd())' in text
    assert 'data["repository_head"] == os.environ["PR_HEAD"]' in text


def test_development_workflow_has_no_scientific_secret_or_confirmatory_escape_hatch() -> None:
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
        "confirmatory",
    )
    for token in forbidden:
        assert token not in text.lower() if token == "confirmatory" else token not in text
    assert "contents: read" in text


def test_freeze_guard_is_read_only_and_fails_closed_on_duplicates() -> None:
    text = _text(FREEZE_GUARD)
    assert "name: exp290-structural-clause-transfer-freeze-guard" in text
    assert "pull_request:" in text
    assert "actions: read" in text
    assert "actions: write" not in text
    assert "exp290-structural-clause-transfer-development" in text
    assert "candidates.sort" in text
    assert 'authoritative = int(candidates[0]["id"]) if candidates else None' in text
    assert "authoritative_development_run_id" in text
    assert "duplicate_development_run_ids" in text
    assert "observed_development_run_ids" in text
    assert "duplicate DEVELOPMENT runs detected" in text
    assert "/cancel" not in text


def test_release_marker_is_absent_before_predata_green() -> None:
    assert not MARKER.exists(), "EXP-290 release marker must not exist before all pre-data gates are GREEN"
