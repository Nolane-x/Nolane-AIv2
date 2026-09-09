from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import inspect

import pytest


FREEZE = "2026-09-08T10:00:00Z"
CHECKPOINT_SEAL = "2026-09-08T10:00:30Z"
PUBLISHED = "2026-09-08T10:01:00Z"


def _real_receipt(*, published_at_utc: str = PUBLISHED, authenticity_status: str = "EXTERNAL_EVIDENCE_RECORDED"):
    from nolane_ai.experiments.exp277_beacon import _receipt_digest

    payload = {
        "schema": "NLM-EXP-277-PUBLIC-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-277",
        "source": "public-beacon.example",
        "beacon_id": "round-123",
        "published_at_utc": published_at_utc,
        "entropy_hex": "ab" * 32,
        "evidence_reference": "https://public-beacon.example/round/123",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": authenticity_status,
        "receipt_digest": "",
    }
    payload["receipt_digest"] = _receipt_digest(payload)
    return payload


def test_exp277_test_beacon_is_explicitly_scientifically_ineligible() -> None:
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt, validate_exp277_beacon_receipt

    receipt = build_test_beacon_receipt(
        source="ci-synthetic",
        beacon_id="exp277-test-round",
        published_at_utc=PUBLISHED,
        entropy_hex="cd" * 32,
        evidence_reference="TEST-ONLY",
    )
    assert receipt["schema"] == "NLM-EXP-277-PUBLIC-BEACON-RECEIPT-V1"
    assert receipt["experiment_id"] == "EXP-277"
    assert receipt["test_only"] is True
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["authenticity_status"] == "TEST_ONLY_SYNTHETIC"
    assert validate_exp277_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc=FREEZE,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
    ) == []


def test_exp277_real_beacon_requires_scientific_classification_and_external_authenticity() -> None:
    from nolane_ai.experiments.exp277_beacon import _receipt_digest, validate_exp277_beacon_receipt

    receipt = _real_receipt()
    assert validate_exp277_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc=FREEZE,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
    ) == []

    not_scientific = deepcopy(receipt)
    not_scientific["scientific_evidence_eligible"] = False
    not_scientific["receipt_digest"] = _receipt_digest(not_scientific)
    assert any("scientific" in error.lower() for error in validate_exp277_beacon_receipt(not_scientific))

    bad_auth = deepcopy(receipt)
    bad_auth["authenticity_status"] = "CRYPTOGRAPHICALLY_VERIFIED_BY_CEREMONY"
    bad_auth["receipt_digest"] = _receipt_digest(bad_auth)
    assert any("authenticity" in error.lower() for error in validate_exp277_beacon_receipt(bad_auth))


def test_exp277_beacon_requires_utc_iso_timestamp_and_at_least_256_bits_entropy() -> None:
    from nolane_ai.experiments.exp277_beacon import _receipt_digest, validate_exp277_beacon_receipt

    non_utc = _real_receipt(published_at_utc="2026-09-08T12:01:00+02:00")
    assert any("timestamp" in error.lower() or "utc" in error.lower() for error in validate_exp277_beacon_receipt(non_utc))

    short_entropy = _real_receipt()
    short_entropy["entropy_hex"] = "aa" * 31
    short_entropy["receipt_digest"] = _receipt_digest(short_entropy)
    assert any("256" in error for error in validate_exp277_beacon_receipt(short_entropy))


def test_exp277_beacon_must_be_strictly_after_both_freeze_boundaries() -> None:
    from nolane_ai.experiments.exp277_beacon import validate_exp277_beacon_receipt

    at_freeze = _real_receipt(published_at_utc=FREEZE)
    errors = validate_exp277_beacon_receipt(
        at_freeze,
        freeze_commit_timestamp_utc=FREEZE,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
    )
    assert any("freeze" in error.lower() for error in errors)

    at_checkpoint = _real_receipt(published_at_utc=CHECKPOINT_SEAL)
    errors = validate_exp277_beacon_receipt(
        at_checkpoint,
        freeze_commit_timestamp_utc=FREEZE,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
    )
    assert any("checkpoint" in error.lower() for error in errors)


def test_exp277_beacon_digest_tamper_fails_even_when_fields_remain_well_formed() -> None:
    from nolane_ai.experiments.exp277_beacon import validate_exp277_beacon_receipt

    receipt = _real_receipt()
    receipt["beacon_id"] = "round-124"
    errors = validate_exp277_beacon_receipt(receipt)
    assert any("digest" in error.lower() for error in errors)


def test_exp277_seed_derivation_uses_exact_frozen_material_and_no_freeze_sha_parameter() -> None:
    from nolane_ai.experiments.exp277_beacon import derive_exp277_challenge_seed

    receipt = _real_receipt()
    protocol_digest = "11" * 32
    seed = derive_exp277_challenge_seed(
        protocol_digest=protocol_digest,
        beacon_receipt=receipt,
        stream="confirmatory",
        replicate=50000,
        freeze_commit_timestamp_utc=FREEZE,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
    )
    material = f"{protocol_digest}|{receipt['receipt_digest']}|EXP-277|confirmatory|50000"
    expected = int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")
    assert seed == expected
    assert set(inspect.signature(derive_exp277_challenge_seed).parameters) == {
        "protocol_digest",
        "beacon_receipt",
        "stream",
        "replicate",
        "freeze_commit_timestamp_utc",
        "checkpoint_seal_created_at_utc",
    }


def test_exp277_seed_derivation_rejects_invalid_receipt_stream_or_replicate() -> None:
    from nolane_ai.experiments.exp277_beacon import derive_exp277_challenge_seed

    receipt = _real_receipt()
    tampered = deepcopy(receipt)
    tampered["entropy_hex"] = "ff" * 32
    with pytest.raises(ValueError, match="beacon"):
        derive_exp277_challenge_seed(
            protocol_digest="22" * 32,
            beacon_receipt=tampered,
            stream="confirmatory",
            replicate=50000,
            freeze_commit_timestamp_utc=FREEZE,
            checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
        )
    with pytest.raises(ValueError, match="stream"):
        derive_exp277_challenge_seed(
            protocol_digest="22" * 32,
            beacon_receipt=receipt,
            stream="",
            replicate=50000,
            freeze_commit_timestamp_utc=FREEZE,
            checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
        )
    with pytest.raises(ValueError, match="replicate"):
        derive_exp277_challenge_seed(
            protocol_digest="22" * 32,
            beacon_receipt=receipt,
            stream="confirmatory",
            replicate=-1,
            freeze_commit_timestamp_utc=FREEZE,
            checkpoint_seal_created_at_utc=CHECKPOINT_SEAL,
        )
