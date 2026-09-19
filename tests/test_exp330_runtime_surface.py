from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/"src/nolane_ai/experiments/exp330_runtime.py"
RUNNER=ROOT/"scripts/exp330_run.py"

def test_measured_step_preserves_source_rng_and_restores_source_gradient():
    text=RUNTIME.read_text()
    for token in (
        "rng_start=torch.get_rng_state().clone()",
        "rng_after_source=torch.get_rng_state().clone()",
        "torch.set_rng_state(rng_after_source)",
        "parameter.grad=applied.clone()",
        "source_grad-target_grad*coefficient",
    ):
        assert token in text

def test_control_uses_exact_inherited_training_primitive():
    text=RUNTIME.read_text()
    assert "_train_one_step_with_effort(compiled,worlds[world_index],optimizer=optimizer,effort=effort)" in text
    assert "training_schedule()" in text

def test_sham_and_projected_arms_are_independent_reconstructions():
    text=RUNTIME.read_text()
    assert text.count("load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)")>=2
    assert "arm_id=SHAM_ARM,project=False" in text
    assert "arm_id=PROJECT_ARM,project=True" in text
    assert "sham_equivalent" in text

def test_runner_is_write_once_and_validates():
    text=RUNNER.read_text()
    assert 'open("x"' in text
    assert "validate_final_evidence(payload)" in text
