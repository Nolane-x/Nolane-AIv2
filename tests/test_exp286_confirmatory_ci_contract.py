from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp286-confirmatory-ci.yml"


def test_exp286_focused_ci_exercises_real_gate_a_prep_without_challenge_entropy() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Build EXP-286 Gate A DEVELOPMENT pilot" in text
    assert "python scripts/run_exp286_paired_dev.py" in text
    assert "--eval-replicates 32" in text
    assert "--eval-start-replicate 20000" in text
    assert "Prepare EXP-286 Gate A" in text
    assert "python scripts/prepare_exp286_confirmatory_gate_a.py" in text
    assert "Assert EXP-286 Gate A pre-beacon boundary" in text
    assert 'challenge_materialized\"] is False' in text
    assert 'confirmatory_data_consumed\"] is False' in text
    assert "--beacon" not in text


def test_exp286_focused_ci_runs_authoritative_execution_court_before_any_beacon() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Authorize authoritative EXP-286 confirmatory execution" in text
    assert "python scripts/authorize_exp286_confirmatory_execution.py" in text
    assert "/tmp/nlm-exp286-authoritative-auth.json" in text
    assert "Assert EXP-286 execution authorization remains pre-beacon" in text
    assert 'AUTHORIZED_NOT_EXECUTED' in text
    assert 'seed_materialization_status' in text
    assert 'NOT_EXECUTED' in text
    assert "--beacon" not in text
