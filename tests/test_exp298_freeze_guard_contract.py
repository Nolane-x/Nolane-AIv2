from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE_GUARD = ROOT / ".github" / "workflows" / "exp298-cross-domain-fidelity-freeze-guard.yml"
DEVELOPMENT = ROOT / ".github" / "workflows" / "exp298-cross-domain-fidelity-development.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_freeze_guard_is_read_only_and_fails_closed_on_duplicate_development_runs() -> None:
    text = _text(FREEZE_GUARD)
    assert "name: exp298-cross-domain-fidelity-freeze-guard" in text
    assert "pull_request:" in text
    assert "actions: read" in text
    assert "actions: write" not in text
    assert "exp298-cross-domain-fidelity-development" in text
    assert "candidates.sort" in text
    assert 'authoritative = int(candidates[0]["id"]) if candidates else None' in text
    assert "authoritative_development_run_id" in text
    assert "duplicate_development_run_ids" in text
    assert "observed_development_run_ids" in text
    assert "duplicate DEVELOPMENT runs detected" in text
    assert "/cancel" not in text
    assert "workflow_dispatch" not in text


def test_development_serializes_duplicate_runs_without_cancellation_write_authority() -> None:
    text = _text(DEVELOPMENT)
    assert "concurrency:" in text
    assert "exp298-development-pr-${{ github.event.pull_request.number }}" in text
    assert "cancel-in-progress: false" in text
    assert "actions: write" not in text
    assert "/cancel" not in text
