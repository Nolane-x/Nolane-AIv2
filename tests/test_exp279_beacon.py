from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest


FREEZE_TIME = "2026-09-09T12:00:00Z"
CHECKPOINT_SEAL_TIME = "2026-09-09T12:00:30Z"
BEACON_TIME = "2026-09-09T12:01:00Z"
PROTOCOL_DIGEST = "p" * 64


def _test_receipt():
    from nolane_ai.experiments.exp279_beacon import build_test_beacon_receipt

    return build_test_beacon_receipt(
        source="synthetic-exp279-beacon-test",
        beacon_id="test-round-279",
        published_at_utc=BEACON_TIME,
        entropy_hex="42" * 32,
        evidence_reference="test-only://exp279-beacon",
    )


def test_exp279_test_beacon_is_explicitly_non_scientific() -> None:
    from nolane_ai.experiments.exp279_beacon import validate_exp279_beacon_receipt

    receipt = _test_receipt()
    assert receipt["schema"] == "NLM-EXP-279-PUBLIC-BEACON-RECEIPT-V1"
    assert receipt["experiment_id"] == "EXP-279"
    assert receipt["test_only"] is True
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["authenticity_status"] == "TEST_ONLY_SYNTHETIC"
    assert validate_exp279_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    ) == []


def test_exp279_beacon_must_be_strictly_post_freeze_and_post_checkpoint_seal() -> None:
    from nolane_ai.experiments.exp279_beacon import (
        build_test_beacon_receipt,
        validate_exp279_beacon_receipt,
    )

    at_freeze = build_test_beacon_receipt(
        source="synthetic",
        beacon_id="at-freeze",
        published_at_utc=FREEZE_TIME,
        entropy_hex="11" * 32,
        evidence_reference="test-only://at-freeze",
    )
    errors = validate_exp279_beacon_receipt(
        at_freeze,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc="2026-09-09T11:59:59Z",
    )
    assert any("strictly after" in error.lower() and "freeze" in error.lower() for error in errors)

    at_checkpoint = build_test_beacon_receipt(
        source="synthetic",
        beacon_id="at-checkpoint",
        published_at_utc=CHECKPOINT_SEAL_TIME,
        entropy_hex="22" * 32,
        evidence_reference="test-only://at-checkpoint",
    )
    errors = validate_exp279_beacon_receipt(
        at_checkpoint,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    )
    assert any("strictly after" in error.lower() and "checkpoint" in error.lower() for error in errors)


def test_exp279_real_beacon_classification_requires_external_evidence_status() -> None:
    from nolane_ai.experiments.exp279_beacon import (
        _receipt_digest,
        validate_exp279_beacon_receipt,
    )

    receipt = _test_receipt()
    real = deepcopy(receipt)
    real["test_only"] = False
    real["scientific_evidence_eligible"] = True
    real["authenticity_status"] = "EXTERNAL_EVIDENCE_RECORDED"
    real["evidence_reference"] = "https://example.invalid/public-beacon/round/279"
    real["receipt_digest"] = _receipt_digest(real)
    assert validate_exp279_beacon_receipt(
        real,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    ) == []

    bad = deepcopy(real)
    bad["authenticity_status"] = "CRYPTOGRAPHICALLY_VERIFIED_WITHOUT_PROOF"
    bad["receipt_digest"] = _receipt_digest(bad)
    errors = validate_exp279_beacon_receipt(bad)
    assert any("authenticity" in error.lower() for error in errors)


def test_exp279_challenge_seed_formula_is_preregistered_and_binds_stratum() -> None:
    from nolane_ai.experiments.exp279_beacon import derive_exp279_challenge_seed

    receipt = _test_receipt()
    replicate = 400
    stratum = "MIXED_RESIDUAL"
    material = (
        f"{PROTOCOL_DIGEST}|{receipt['receipt_digest']}|EXP-279|challenge|"
        f"{replicate}|{stratum}"
    )
    expected = int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")
    observed = derive_exp279_challenge_seed(
        protocol_digest=PROTOCOL_DIGEST,
        beacon_receipt=receipt,
        stream="challenge",
        replicate=replicate,
        stratum=stratum,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    )
    assert observed == expected

    assert derive_exp279_challenge_seed(
        protocol_digest=PROTOCOL_DIGEST,
        beacon_receipt=receipt,
        stream="challenge",
        replicate=replicate + 1,
        stratum=stratum,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    ) != observed
    assert derive_exp279_challenge_seed(
        protocol_digest=PROTOCOL_DIGEST,
        beacon_receipt=receipt,
        stream="challenge",
        replicate=replicate,
        stratum="PROPAGATION_FIT",
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    ) != observed


def test_exp279_seed_rejects_non_frozen_stream_stratum_and_tampered_receipt() -> None:
    from nolane_ai.experiments.exp279_beacon import derive_exp279_challenge_seed

    receipt = _test_receipt()
    common = dict(
        protocol_digest=PROTOCOL_DIGEST,
        beacon_receipt=receipt,
        replicate=400,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        checkpoint_seal_created_at_utc=CHECKPOINT_SEAL_TIME,
    )
    with pytest.raises(ValueError, match="stream"):
        derive_exp279_challenge_seed(
            **common,
            stream="evaluation",
            stratum="PROPAGATION_FIT",
        )
    with pytest.raises(ValueError, match="stratum"):
        derive_exp279_challenge_seed(
            **common,
            stream="challenge",
            stratum="POSTHOC_STRATUM",
        )

    bad = deepcopy(receipt)
    bad["entropy_hex"] = "99" * 32
    with pytest.raises(ValueError, match="beacon receipt"):
        derive_exp279_challenge_seed(
            **{**common, "beacon_receipt": bad},
            stream="challenge",
            stratum="PROPAGATION_FIT",
        )
