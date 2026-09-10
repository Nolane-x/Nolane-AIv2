from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp289-confirmatory-ci.yml"


def test_exp289_confirmatory_ci_normalizes_freeze_timestamp_to_utc() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "git show -s --format=%cI HEAD > /tmp/nlm-exp289-freeze-time.txt" not in text
    assert "datetime.fromisoformat" in text
    assert "astimezone(timezone.utc)" in text
    assert 'replace("+00:00", "Z")' in text
    assert "/tmp/nlm-exp289-freeze-time.txt" in text
