from __future__ import annotations

from copy import deepcopy

import pytest

from nolane_ai.protocol.evidence import canonical_sha256

FREEZE_SHA = "1" * 40
FREEZE_TIME = "2026-09-10T10:45:00Z"
BEACON_TIME = "2026-09-10T10:45:01Z"


def _api():
    try:
        from nolane_ai.experiments.exp289_beacon import (
            build_test_beacon_receipt,
            derive_exp289_challenge_seed,
            validate_exp289_beacon_receipt,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-289 post-freeze beacon authority is missing")
    return build_test_beacon_receipt, derive_exp289_challenge_seed, validate_exp289_beacon_receipt


def test_exp289_test_beacon_is_explicitly_non_scientific_and_self_hashing() -> None:
    build, _, validate = _api()
    receipt = build(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp289-test-beacon-1",
        published_at_utc=BEACON_TIME,
        entropy_hex="ab" * 32,
        evidence_reference="unit-test-only",
    )
    assert receipt["schema"] == "NLM-EXP-289-BEACON-RECEIPT-V1"
    assert receipt["experiment_id"] == "EXP-289"
    assert receipt["test_only"] is True
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["source"] == "TEST-ONLY deterministic beacon"
    assert receipt["beacon_id"] == "exp289-test-beacon-1"
    assert receipt["published_at_utc"] == BEACON_TIME
    assert receipt["entropy_hex"] == "ab" * 32
    assert validate(receipt) == []


def test_exp289_challenge_seed_is_domain_separated_and_bound_to_freeze_and_beacon() -> None:
    build, derive, _ = _api()
    receipt = build(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp289-test-beacon-1",
        published_at_utc=BEACON_TIME,
        entropy_hex="ab" * 32,
        evidence_reference="unit-test-only",
    )
    kwargs = {
        "protocol_digest": "c" * 64,
        "freeze_commit_sha": FREEZE_SHA,
        "freeze_commit_timestamp_utc": FREEZE_TIME,
        "beacon_receipt": receipt,
        "stream": "challenge",
        "replicate": 32,
    }
    first = derive(**kwargs)
    assert isinstance(first, int)
    assert first >= 0
    assert first == derive(**kwargs)
    assert first != derive(**{**kwargs, "stream": "world"})
    assert first != derive(**{**kwargs, "replicate": 33})
    changed = deepcopy(receipt)
    changed["entropy_hex"] = "cd" * 32
    clean = deepcopy(changed)
    clean.pop("receipt_digest", None)
    changed["receipt_digest"] = canonical_sha256(clean)
    assert first != derive(**{**kwargs, "beacon_receipt": changed})


def test_exp289_beacon_validator_rejects_rehashed_semantic_tampering() -> None:
    build, _, validate = _api()
    receipt = build(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp289-test-beacon-1",
        published_at_utc=BEACON_TIME,
        entropy_hex="ab" * 32,
        evidence_reference="unit-test-only",
    )
    changed = deepcopy(receipt)
    changed["scientific_evidence_eligible"] = True
    clean = deepcopy(changed)
    clean.pop("receipt_digest", None)
    changed["receipt_digest"] = canonical_sha256(clean)
    assert validate(changed)


def test_exp289_seed_derivation_rejects_invalid_or_tampered_beacon() -> None:
    build, derive, _ = _api()
    receipt = build(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp289-test-beacon-1",
        published_at_utc=BEACON_TIME,
        entropy_hex="ab" * 32,
        evidence_reference="unit-test-only",
    )
    receipt["receipt_digest"] = "0" * 64
    with pytest.raises(ValueError, match="beacon"):
        derive(
            protocol_digest="c" * 64,
            freeze_commit_sha=FREEZE_SHA,
            freeze_commit_timestamp_utc=FREEZE_TIME,
            beacon_receipt=receipt,
            stream="challenge",
            replicate=32,
        )
