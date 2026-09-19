from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/"src/nolane_ai/experiments/exp328_runtime.py"
RUNNER=ROOT/"scripts/exp328_run.py"

def test_runtime_uses_independent_locked_reconstruction_per_arm():
    text=RUNTIME.read_text()
    assert "load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)" in text
    assert "for a in ARM_IDS" in text
    assert "architecture" not in text.lower()
    assert "scale_authorized" in text

def test_runner_is_write_once_and_validates_final_evidence():
    text=RUNNER.read_text()
    assert 'open("x"' in text
    assert "validate_final_evidence(payload)" in text
