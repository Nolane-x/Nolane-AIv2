import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.runner import execute_stage_a
from nolane_ai.protocol.schema import EXPECTED_FIRST_GATES, EXPECTED_RESULT_STATES
from nolane_ai.protocol.seeds import STREAM_NAMES


def _valid_protocol(root_seed: str = "runner") -> dict:
    experiments = []
    for experiment_id in EXPECTED_FIRST_GATES:
        experiments.append(
            {
                "experiment_id": experiment_id,
                "hypothesis_id": f"H-{experiment_id}",
                "question": "fixture",
                "arms": [{"id": "a"}, {"id": "b"}],
                "primary_endpoint": {"metric": "fixture"},
                "protected_endpoints": [],
                "mesi": {"value": 0.1},
                "sample_size_plan": {"power_target": 0.9},
                "analysis_method": "paired",
                "multiplicity_family": "fixture",
                "resource_match": {"compute": "matched"},
                "failure_policy": {"scientific_failure_kept": True},
                "challenge_generator": {"lane": "future"},
                "decision_rule": {"promote_if": "fixture"},
            }
        )
    return {
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "status": "FROZEN_V1",
        "source_revision": "NLM-V0.16.1",
        "result_states": list(EXPECTED_RESULT_STATES),
        "rng": {"root_seed": root_seed, "streams": list(STREAM_NAMES)},
        "experiments": experiments,
    }


def _write_protocol(tmp_path: Path, payload: dict) -> tuple[Path, Path]:
    protocol = tmp_path / "stage_a.json"
    digest = tmp_path / "stage_a.sha256"
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest(), encoding="utf-8")
    return protocol, digest


def test_execute_stage_a_refuses_protocol_digest_drift(tmp_path: Path):
    protocol, digest = _write_protocol(tmp_path, _valid_protocol("x"))
    digest.write_text("bad", encoding="utf-8")
    try:
        execute_stage_a(protocol_path=protocol, digest_path=digest, code_root=tmp_path, replicates=1)
    except RuntimeError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("protocol drift must be rejected")


def test_execute_stage_a_refuses_digest_matching_but_schema_invalid_protocol(tmp_path: Path):
    payload = _valid_protocol()
    payload["experiments"] = payload["experiments"][:-1]
    protocol, digest = _write_protocol(tmp_path, payload)
    try:
        execute_stage_a(protocol_path=protocol, digest_path=digest, code_root=tmp_path, replicates=1)
    except RuntimeError as error:
        assert "protocol validation" in str(error)
    else:
        raise AssertionError("schema-invalid frozen bytes must be rejected")


def test_execute_stage_a_returns_ev_e2_packet_with_analysis(tmp_path: Path):
    protocol, digest = _write_protocol(tmp_path, _valid_protocol())
    packet = execute_stage_a(
        protocol_path=protocol,
        digest_path=digest,
        code_root=tmp_path,
        replicates=2,
        bootstrap_samples=100,
    )
    assert packet["evidence_level"] == "EV-E2"
    assert packet["decision"] == "UNVERIFIED"
    assert set(packet["analysis"]) == set(EXPECTED_FIRST_GATES)
