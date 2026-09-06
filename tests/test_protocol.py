from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _schema_module():
    module = import_module("nolane_ai.protocol.schema")
    assert hasattr(module, "load_and_validate_protocol")
    return module


def test_stage_a_protocol_freezes_exactly_six_first_gates():
    protocol = _schema_module().load_and_validate_protocol(ROOT / "protocols/stage_a_v1.json")
    assert tuple(protocol.experiment_ids) == (
        "EXP-277", "EXP-279", "EXP-282", "EXP-286", "EXP-289", "EXP-297"
    )


def test_each_gate_has_required_confirmatory_fields():
    protocol = _schema_module().load_and_validate_protocol(ROOT / "protocols/stage_a_v1.json")
    for exp in protocol.experiments:
        assert len(exp.arms) >= 2
        assert exp.primary_endpoint
        assert exp.mesi["value"] > 0
        assert exp.sample_size_plan["power_target"] >= 0.8
        assert exp.resource_match
        assert exp.decision_rule
        assert exp.failure_policy


def test_result_states_are_finite_and_include_kill_and_unverified():
    protocol = _schema_module().load_and_validate_protocol(ROOT / "protocols/stage_a_v1.json")
    assert set(protocol.result_states) == {
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "PRACTICALLY_EQUIVALENT_USE_SIMPLER_RIVAL",
        "KILL_SUBSYSTEM",
        "INVALID_RUN",
        "UNVERIFIED",
    }
