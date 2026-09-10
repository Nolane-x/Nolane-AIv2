from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


SCHEMA = "NLM-EXP-286-PUBLIC-BEACON-RECEIPT-V1"
EXPERIMENT_ID = "EXP-286"
_REAL_AUTHENTICITY_STATUSES = {"EXTERNAL_EVIDENCE_RECORDED", "SOURCE_VERIFIED"}


def _receipt_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("receipt_digest", None)
    return canonical_sha256(clean)


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("timestamp is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _valid_entropy_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 64 or len(value) % 2 != 0:
        return False
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        return False
    return len(raw) >= 32


def build_test_beacon_receipt(
    *,
    source: str,
    beacon_id: str,
    published_at_utc: str,
    entropy_hex: str,
    evidence_reference: str,
) -> dict[str, Any]:
    """Build deterministic synthetic entropy for tests only.

    The returned receipt is structurally valid but explicitly ineligible for
    scientific evidence. Production code must ingest externally recorded
    public-beacon evidence instead of calling an internal entropy generator.
    """

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source": source,
        "beacon_id": beacon_id,
        "published_at_utc": published_at_utc,
        "entropy_hex": entropy_hex.lower(),
        "evidence_reference": evidence_reference,
        "test_only": True,
        "scientific_evidence_eligible": False,
        "authenticity_status": "TEST_ONLY_SYNTHETIC",
        "receipt_digest": "",
    }
    payload["receipt_digest"] = _receipt_digest(payload)
    errors = validate_exp286_beacon_receipt(payload)
    if errors:
        raise ValueError("invalid synthetic EXP-286 beacon receipt: " + "; ".join(errors))
    return payload


def validate_exp286_beacon_receipt(
    payload: dict[str, Any],
    *,
    freeze_commit_timestamp_utc: str | None = None,
    authorization_created_at_utc: str | None = None,
) -> list[str]:
    """Validate an EXP-286 public-beacon evidence receipt fail closed.

    Timestamp arguments are authority boundaries supplied by the caller. When
    present, the beacon must have been published strictly after both. This
    validator never fetches, generates, or authenticates external entropy by
    itself; real receipts must carry an allowed external authenticity state.
    """

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-286 beacon receipt must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-286 beacon receipt identity")

    for field in ("source", "beacon_id", "evidence_reference"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            errors.append(f"EXP-286 beacon {field} missing")

    try:
        published = _parse_utc(payload.get("published_at_utc"))
    except ValueError:
        published = None
        errors.append("EXP-286 beacon publication timestamp invalid or not UTC")

    if not _valid_entropy_hex(payload.get("entropy_hex")):
        errors.append("EXP-286 beacon entropy must contain at least 256 bits of hex entropy")

    test_only = payload.get("test_only")
    scientific = payload.get("scientific_evidence_eligible")
    authenticity = payload.get("authenticity_status")
    if not isinstance(test_only, bool) or not isinstance(scientific, bool):
        errors.append("EXP-286 beacon evidence classification invalid")
    if test_only is True:
        if scientific is not False or authenticity != "TEST_ONLY_SYNTHETIC":
            errors.append("TEST-ONLY EXP-286 beacon cannot be scientific evidence")
    elif test_only is False:
        if scientific is not True:
            errors.append("real EXP-286 beacon must be scientifically eligible")
        if authenticity not in _REAL_AUTHENTICITY_STATUSES:
            errors.append("external EXP-286 beacon authenticity status invalid")
    else:
        errors.append("EXP-286 beacon test_only flag missing")

    digest = payload.get("receipt_digest")
    if not isinstance(digest, str) or digest != _receipt_digest(payload):
        errors.append("EXP-286 beacon receipt digest mismatch")

    if freeze_commit_timestamp_utc is not None:
        try:
            frozen = _parse_utc(freeze_commit_timestamp_utc)
        except ValueError:
            errors.append("EXP-286 Gate A freeze timestamp invalid")
        else:
            if published is not None and published <= frozen:
                errors.append("EXP-286 beacon must be published strictly after Gate A freeze")

    if authorization_created_at_utc is not None:
        try:
            authorized = _parse_utc(authorization_created_at_utc)
        except ValueError:
            errors.append("EXP-286 execution authorization timestamp invalid")
        else:
            if published is not None and published <= authorized:
                errors.append("EXP-286 beacon must be published strictly after execution authorization")

    return errors


def derive_exp286_challenge_seed(
    *,
    protocol_digest: str,
    beacon_receipt: dict[str, Any],
    stream: str,
    replicate: int,
    freeze_commit_timestamp_utc: str | None = None,
    authorization_created_at_utc: str | None = None,
) -> int:
    """Derive one domain-separated post-freeze seed from a validated receipt."""

    if not isinstance(protocol_digest, str) or not protocol_digest:
        raise ValueError("protocol_digest is required")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream is required")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")

    errors = validate_exp286_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
        authorization_created_at_utc=authorization_created_at_utc,
    )
    if errors:
        raise ValueError("invalid EXP-286 beacon receipt: " + "; ".join(errors))

    material = (
        f"{protocol_digest}|{beacon_receipt['receipt_digest']}|"
        f"{EXPERIMENT_ID}|{stream}|{replicate}"
    )
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
