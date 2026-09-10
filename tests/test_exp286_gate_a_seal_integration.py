from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "seal_exp286_confirmatory_gate_a.py"
WORKFLOW = ROOT / ".github" / "workflows" / "exp286-confirmatory-ci.yml"


def test_exp286_gate_a_seal_cli_entrypoint_exists_without_beacon_override() -> None:
    assert SCRIPT.exists(), "EXP-286 Gate-A seal CLI entrypoint is missing"
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--freeze-commit-sha" in text
    assert "--freeze-commit-timestamp-utc" in text
    assert "--execution" in text
    assert "--prep" in text
    assert "--authorization" in text
    assert "--output" in text
    assert "--beacon" not in text
    assert "--seed" not in text


def test_exp286_focused_ci_rehearses_immutable_gate_a_seal_without_entropy() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Seal EXP-286 Gate A with synthetic CI freeze metadata" in text
    assert "python scripts/seal_exp286_confirmatory_gate_a.py" in text
    assert "/tmp/nlm-exp286-ci-gatea-seal.json" in text
    assert "Assert EXP-286 Gate-A seal remains pre-beacon" in text
    assert 'CONFIRMATORY_GATE_A_SEALED' in text
    assert 'FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY' in text
    assert 'seed_materialization_status' in text
    assert 'NOT_EXECUTED' in text
    assert 'challenge_materialized' in text
    assert 'confirmatory_data_consumed' in text
    assert 'EV-E2' in text
    assert 'UNVERIFIED' in text
