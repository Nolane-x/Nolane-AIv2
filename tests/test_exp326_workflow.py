from __future__ import annotations
from pathlib import Path

W=Path(".github/workflows/exp326-balanced-multiworld-breakpoint.yml")

def text(): return W.read_text(encoding="utf-8")

def test_workflow_is_manual_and_binds_parent_artifacts() -> None:
    s=text()
    assert "workflow_dispatch:" in s
    assert "35406123945" in s and "10572536792" in s
    assert "35345351869" in s and "10547681681" in s
    assert "bbf4aca8cae39ff424ad2fa4cea0c122a60b405162198413fb921dd891dcaf3c" in s

def test_workflow_has_no_user_selected_scientific_controls() -> None:
    s=text().lower()
    for token in ("--learning-rate","--model-size","--group-size","--exposures","30000000","100000000"):
        assert token not in s
