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


def test_parent_anchor_json_lists_match_runtime_metric_tuples():
    from nolane_ai.experiments.exp331_runtime import _control_matches
    record={
        "pair_id":"P01","mode":"CONTROL","total_optimizer_updates":64,
        "members":[0,1],
        "world_token_accuracies":(1.0,1.0),
        "world_full_answer_exact":(1.0,1.0),
        "model_state_digest":"a"*64,
        "optimizer_state_digest":"b"*64,
        "rng_state_digest":"c"*64,
        "nonfinite_events":0,
        "negative_dot_count":0,
        "projection_event_count":0,
    }
    anchor={
        "members":[0,1],
        "passed":True,
        "world_token_accuracies":[1.0,1.0],
        "world_full_answer_exact":[1.0,1.0],
        "final_state":{
            "model_state_digest":"a"*64,
            "optimizer_state_digest":"b"*64,
            "rng_state_digest":"c"*64,
        },
    }
    assert _control_matches(record,anchor)
