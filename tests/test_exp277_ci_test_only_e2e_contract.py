from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
EXP277_E2E_TEST = "tests/test_exp277_confirmatory_gate_a_cli.py"


def test_normal_model_smoke_runs_exp277_test_only_gate_a_to_gate_b_e2e() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert EXP277_E2E_TEST in workflow, (
        "normal model-smoke must execute the EXP-277 Gate A -> TEST-ONLY Gate B "
        "E2E contract so the non-promotion firewall is continuously exercised"
    )
