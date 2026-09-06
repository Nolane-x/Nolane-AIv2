from importlib import import_module


def _evidence_module():
    module = import_module("nolane_ai.protocol.evidence")
    assert hasattr(module, "canonical_sha256")
    assert hasattr(module, "validate_evidence_packet")
    return module


def test_canonical_hash_ignores_mapping_insertion_order():
    evidence = _evidence_module()
    assert evidence.canonical_sha256({"a": 1, "b": 2}) == evidence.canonical_sha256({"b": 2, "a": 1})


def test_evidence_packet_cannot_claim_neural_maturity_without_raw_metrics():
    evidence = _evidence_module()
    packet = {
        "protocol_digest": "a" * 64,
        "code_digest": "b" * 64,
        "claim_id": "H-CBRF-01",
        "evidence_level": "EV-E3",
        "decision": "PROMOTE_TO_NEXT_STAGE",
        "raw_per_replicate_metrics": [],
    }
    errors = evidence.validate_evidence_packet(packet)
    assert any("raw_per_replicate_metrics" in error for error in errors)
