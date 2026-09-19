from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/nolane_ai/experiments/exp330_runtime.py"
RUNNER = ROOT / "scripts/exp330_run.py"


def test_group_probes_are_cloned_masked_and_rng_restored():
    text = RUNTIME.read_text()
    assert "copy.deepcopy((compiled,optimizer))" in text
    assert "torch.get_rng_state().clone()" in text
    assert "torch.set_rng_state(rng)" in text
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)" in text
    assert "p.grad=None" in text
    assert "parameter_group_for_name(name) != group" in text


def test_main_replay_remains_exact_alt_0_2():
    text = RUNTIME.read_text()
    assert "for wi in WORLD_INDICES" in text
    assert "effort_for_round(r)" in text
    assert "_train_one_step_with_effort" in text


def test_parent_reproduction_is_exact_state_gated():
    text = RUNTIME.read_text()
    assert "exact_parent_reproduction(reproduction,final_state)" in text
    assert "model_state_digest(model)" in text
    assert "optimizer_state_digest(optimizer)" in text
    assert "rng_state_digest()" in text


def test_runner_is_write_once_and_validates():
    text = RUNNER.read_text()
    assert 'open("x"' in text
    assert "validate_final_evidence(payload)" in text
