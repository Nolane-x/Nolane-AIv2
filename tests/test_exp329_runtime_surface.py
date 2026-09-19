from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/"src/nolane_ai/experiments/exp329_runtime.py"
RUNNER=ROOT/"scripts/exp329_run.py"

def test_probes_are_cloned_and_rng_restored():
    text=RUNTIME.read_text()
    assert "copy.deepcopy((compiled,optimizer))" in text
    assert "torch.get_rng_state().clone()" in text
    assert "torch.set_rng_state(rng)" in text
    assert "PROBE_ROUNDS" in text

def test_main_replay_is_frozen_alt_0_2():
    text=RUNTIME.read_text()
    assert "for world_index in WORLD_INDICES" in text
    assert "effort_for_round(round_index)" in text
    assert "_train_one_step_with_effort" in text

def test_runner_is_write_once_and_validates():
    text=RUNNER.read_text()
    assert 'open("x"' in text
    assert "validate_final_evidence(payload)" in text
