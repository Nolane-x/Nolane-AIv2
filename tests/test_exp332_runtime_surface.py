from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/nolane_ai/experiments/exp332_runtime.py"
RUNNER = ROOT / "scripts/exp332_run.py"

def test_runtime_reuses_exact_exp331_arm_machinery():
    text = RUNTIME.read_text()
    assert "from .exp331_runtime import _result, _run_arm, validate_exp330_parent" in text
    assert "_run_arm(checkpoint_path, receipt_path, selection_lock_path, pair_id=pair, mode=mode)" in text

def test_historical_gate_is_behavioral_but_within_run_sham_is_exact():
    text = RUNTIME.read_text()
    for token in (
        "metric_vector_equal(record.get(\"world_token_accuracies\")",
        "record.get(\"rng_state_digest\")",
        "sham_equivalent(results[f\"{pair}:CONTROL\"], results[f\"{pair}:SHAM\"])",
        "historical_state_divergence_pairs",
    ):
        assert token in text
    assert 'record.get("model_state_digest") ==' not in text.split("def _control_matches_behavior",1)[1].split("def run_court",1)[0]

def test_witnesses_must_share_behavior_and_diverge_in_historical_state():
    text = RUNTIME.read_text()
    assert "left[\"final_state\"][\"rng_state_digest\"] == right[\"final_state\"][\"rng_state_digest\"]" in text
    assert "left[\"final_state\"][\"model_state_digest\"] != right[\"final_state\"][\"model_state_digest\"]" in text
    assert "divergence_pairs == PAIR_IDS" in text

def test_runner_is_write_once_and_validates():
    text = RUNNER.read_text()
    assert 'open("x"' in text
    assert "validate_final_evidence(payload)" in text
