from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/nolane_ai/experiments/exp333_runtime.py"
RUNNER = ROOT / "scripts/exp333_run.py"

def test_runtime_reuses_exact_projection_measurement_primitive():
    text = RUNTIME.read_text()
    assert "from .exp331_runtime import _measured_step" in text
    assert 'project=mode == "PROJECT"' in text
    assert "for pair in PAIR_IDS" in text and "for mode in MODES" in text

def test_parent_anchor_reproduction_is_behavioral_not_cross_run_state_bitwise():
    text = RUNTIME.read_text()
    block = text.split("def _control_matches_parent",1)[1].split("def run_court",1)[0]
    assert 'record.get("rng_state_digest") == anchor["rng_state_digest"]' in block
    assert 'record.get("model_state_digest")' not in block
    assert 'record.get("optimizer_state_digest")' not in block

def test_final_validator_recomputes_all_derived_pair_vectors():
    text = RUNTIME.read_text()
    for token in (
        "expected_control_pass",
        "expected_new_failures",
        "expected_project_pass",
        "baseline_failures",
        "rescued_baseline_failures",
        "unresolved_baseline_failures",
        "project_regressions",
        "baseline_failures_without_projection_trigger",
        "projection_event_counts",
    ):
        assert token in text

def test_runner_is_write_once_and_validates_final_evidence():
    text = RUNNER.read_text()
    assert 'open("x"' in text
    assert "validate_final_evidence(payload)" in text
