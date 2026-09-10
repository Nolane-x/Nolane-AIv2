from __future__ import annotations

from nolane_ai.experiments.exp286_beacon import _receipt_digest, validate_exp286_beacon_receipt


def _real_receipt(published_at_utc: str) -> dict:
    receipt = {
        "schema": "NLM-EXP-286-PUBLIC-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-286",
        "source": "drand",
        "beacon_id": "drand-round-1",
        "published_at_utc": published_at_utc,
        "entropy_hex": "ab" * 32,
        "evidence_reference": "https://api.drand.sh/public/1",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": "SOURCE_VERIFIED",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


def test_exp286_real_beacon_must_be_strictly_after_actual_gate_a_seal_creation() -> None:
    receipt = _real_receipt("2026-09-10T10:00:01Z")
    errors = validate_exp286_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc="2026-09-10T09:59:00Z",
        seal_created_at_utc="2026-09-10T10:00:01Z",
    )
    assert any("actual Gate A seal creation" in error for error in errors)

    receipt = _real_receipt("2026-09-10T10:00:02Z")
    assert validate_exp286_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc="2026-09-10T09:59:00Z",
        seal_created_at_utc="2026-09-10T10:00:01Z",
    ) == []


def test_exp286_actual_gate_a_seal_creation_cannot_precede_freeze() -> None:
    receipt = _real_receipt("2026-09-10T10:00:02Z")
    errors = validate_exp286_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc="2026-09-10T10:00:00Z",
        seal_created_at_utc="2026-09-10T09:59:59Z",
    )
    assert any("seal creation cannot precede" in error for error in errors)
