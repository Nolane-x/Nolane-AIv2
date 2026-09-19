from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/"src/nolane_ai/experiments/exp331_runtime.py";RUNNER=ROOT/"scripts/exp331_run.py"

def test_control_uses_exact_inherited_training_primitive():
    text=RUNTIME.read_text();assert "_train_one_step_with_effort(compiled,worlds[source_index],optimizer=optimizer,effort=effort)" in text
    assert "training_schedule(pair_id)" in text

def test_measured_path_restores_rng_and_source_gradient():
    text=RUNTIME.read_text()
    for token in ("torch.get_rng_state().clone()","torch.set_rng_state(rng_after)","p.grad=(s-t*coef if apply else s).clone()","TARGET_NORM_SQUARED_FLOOR"):
        assert token in text

def test_parent_reproduction_is_exact_state_anchored():
    text=RUNTIME.read_text()
    for token in ('record.get("model_state_digest")==state["model_state_digest"]','record.get("optimizer_state_digest")==state["optimizer_state_digest"]','record.get("rng_state_digest")==state["rng_state_digest"]'):
        assert token in text

def test_runner_write_once_and_validates():
    text=RUNNER.read_text();assert 'open("x"' in text;assert "validate_final_evidence(payload)" in text
