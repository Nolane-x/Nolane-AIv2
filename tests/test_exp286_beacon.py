from __future__ import annotations

from copy import deepcopy

import pytest

from nolane_ai.protocol.evidence import canonical_sha256


def _api():
    try:
        from nolane_ai.experiments.exp286_beacon import (
            build_test_beacon_receipt,
            derive_exp286_challenge_seed,
            validate_exp286_beacon_receipt,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 post-freeze beacon barrier is missing")
    return (
        build_test_beacon_receipt,
        derive_exp286_challenge_seed,
        validate_exp286_beacon_receipt,
    )


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    clean = deepcopy(payload)
    clean.pop("receipt_digest", None)
    payload["receipt_digest"] = canonical_sha256(clean)
    return payload


def test_test_beacon_is_deterministic_but_never_scientific() -> None:
    build, derive, validate = _api()
    receipt = build(
        source="test-fixture",
        beacon_id="fixture-1",
        published_at_utc="2026-09-10T05:00:00Z",
        entropy_hex="ab" * 32,
        evidence_reference="synthetic://fixture-1",
    )

    assert receipt["experiment_id"] == "EXP-286"
    assert receipt["test_only"] is True
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["authenticity_status"] == "TEST_ONLY_SYNTHETIC"
    assert validate(receipt) == []

    first = derive(
        protocol_digest="protocol-digest",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=30032,
    )
    second = derive(
        protocol_digest="protocol-digest",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=30032,
    )
    assert first == second


def test_real_beacon_must_be_strictly_post_freeze_and_post_authorization() -> None:
    _, _, validate = _api()
    payload: dict[str, object] = {
        "schema": "NLM-EXP-286-PUBLIC-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-286",
        "source": "external-public-source",
        "beacon_id": "public-1",
        "published_at_utc": "2026-09-10T05:00:00Z",
        "entropy_hex": "cd" * 32,
        "evidence_reference": "external-evidence-record",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": "EXTERNAL_EVIDENCE_RECORDED",
        "receipt_digest": "",
    }
    _rehash(payload)

    errors = validate(
        payload,
        freeze_commit_timestamp_utc="2026-09-10T05:00:00Z",
        authorization_created_at_utc="2026-09-10T04:59:59Z",
    )
    assert any("strictly after Gate A freeze" in item for item in errors)

    errors = validate(
        payload,
        freeze_commit_timestamp_utc="2026-09-10T04:59:58Z",
        authorization_created_at_utc="2026-09-10T05:00:00Z",
    )
    assert any("strictly after execution authorization" in item for item in errors)


def test_real_beacon_requires_external_authenticity_classification() -> None:
    _, _, validate = _api()
    payload: dict[str, object] = {
        "schema": "NLM-EXP-286-PUBLIC-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-286",
        "source": "external-public-source",
        "beacon_id": "public-2",
        "published_at_utc": "2026-09-10T05:00:01Z",
        "entropy_hex": "cd" * 32,
        "evidence_reference": "external-evidence-record",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": "UNVERIFIED",
        "receipt_digest": "",
    }
    _rehash(payload)

    errors = validate(payload)
    assert any("authenticity" in item for item in errors)


def test_seed_derivation_rejects_tampered_receipt() -> None:
    build, derive, _ = _api()
    receipt = build(
        source="test-fixture",
        beacon_id="fixture-2",
        published_at_utc="2026-09-10T05:00:00Z",
        entropy_hex="ef" * 32,
        evidence_reference="synthetic://fixture-2",
    )
    receipt["entropy_hex"] = "00" * 32

    with pytest.raises(ValueError, match="invalid EXP-286 beacon receipt"):
        derive(
            protocol_digest="protocol-digest",
            beacon_receipt=receipt,
            stream="challenge",
            replicate=30032,
        )


def test_seed_derivation_is_domain_separated() -> None:
    build, derive, _ = _api()
    receipt = build(
        source="test-fixture",
        beacon_id="fixture-3",
        published_at_utc="2026-09-10T05:00:00Z",
        entropy_hex="12" * 32,
        evidence_reference="synthetic://fixture-3",
    )

    seed_a = derive(
        protocol_digest="protocol-a",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=30032,
    )
    seed_b = derive(
        protocol_digest="protocol-b",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=30032,
    )
    seed_c = derive(
        protocol_digest="protocol-a",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=30033,
    )
    assert len({seed_a, seed_b, seed_c}) == 3
