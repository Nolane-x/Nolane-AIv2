import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.runner import execute_stage_a


def test_execute_stage_a_refuses_protocol_digest_drift(tmp_path: Path):
    protocol = tmp_path / "stage_a.json"
    digest = tmp_path / "stage_a.sha256"
    protocol.write_text(json.dumps({"protocol_id":"NLM-REASONING-STAGE-A-CONFIRMATORY-V1","status":"FROZEN_V1","rng":{"root_seed":"x"}}), encoding="utf-8")
    digest.write_text("bad", encoding="utf-8")
    try:
        execute_stage_a(protocol_path=protocol, digest_path=digest, code_root=tmp_path, replicates=1)
    except RuntimeError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("protocol drift must be rejected")


def test_execute_stage_a_returns_ev_e2_packet_with_analysis(tmp_path: Path):
    protocol = tmp_path / "stage_a.json"
    digest = tmp_path / "stage_a.sha256"
    payload = {"protocol_id":"NLM-REASONING-STAGE-A-CONFIRMATORY-V1","status":"FROZEN_V1","rng":{"root_seed":"runner"}}
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest(), encoding="utf-8")
    packet = execute_stage_a(protocol_path=protocol, digest_path=digest, code_root=tmp_path, replicates=2, bootstrap_samples=100)
    assert packet["evidence_level"] == "EV-E2"
    assert packet["decision"] == "UNVERIFIED"
    assert set(packet["analysis"]) == {"EXP-277","EXP-279","EXP-282","EXP-286","EXP-289","EXP-297"}
