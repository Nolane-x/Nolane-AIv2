from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


SCHEMA = "NLM-EXP-279-PUBLIC-BEACON-RECEIPT-V1"
EXPERIMENT_ID = "EXP-279"
CHALLENGE_STREAM = "challenge"
# Keep the public-beacon/seed court importable in the core (non-model) runtime.
# The same predeclared schedule is independently sealed and validated by Gate A.
STRATA = ("PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL")
SEED_RULE = (
    "SHA256(protocol_digest|beacon_receipt_digest|EXP-279|challenge|replicate|stratum)"
)
_REAL_AUTHENTICITY_STATUSES = {
    "EXTERNAL_EVIDENCE_RECORDED",
    "SOURCE_VERIFIED",
}


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
        "seed_rule": SEED_RULE,
        "receipt_digest": "",
    }
    payload["receipt_digest"] = _receipt_digest(payload)
    errors = validate_exp279_beacon_receipt(payload)
    if errors:
        raise ValueError(
            "invalid synthetic EXP-279 beacon receipt: " + "; ".join(errors)
        )
    return payload


def validate_exp279_beacon_receipt(
    payload: dict[str, Any],
    *,
    freeze_commit_timestamp_utc: str | None = None,
    checkpoint_seal_created_at_utc: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-279 beacon receipt must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 beacon receipt identity")

    for field in ("source", "beacon_id", "evidence_reference"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            errors.append(f"EXP-279 beacon {field} missing")

    try:
        published = _parse_utc(payload.get("published_at_utc"))
    except ValueError:
        published = None
        errors.append("EXP-279 beacon publication timestamp invalid or not UTC")

    if not _valid_entropy_hex(payload.get("entropy_hex")):
        errors.append(
            "EXP-279 beacon entropy must contain at least 256 bits of hex entropy"
        )

    if payload.get("seed_rule") != SEED_RULE:
        errors.append("EXP-279 beacon seed-rule declaration drift")

    test_only = payload.get("test_only")
    scientific = payload.get("scientific_evidence_eligible")
    authenticity = payload.get("authenticity_status")
    if not isinstance(test_only, bool) or not isinstance(scientific, bool):
        errors.append("EXP-279 beacon evidence classification invalid")
    if test_only is True:
        if scientific is not False or authenticity != "TEST_ONLY_SYNTHETIC":
            errors.append("TEST-ONLY EXP-279 beacon cannot be scientific evidence")
    elif test_only is False:
        if scientific is not True:
            errors.append("real EXP-279 beacon must be scientifically eligible")
        if authenticity not in _REAL_AUTHENTICITY_STATUSES:
            errors.append("external EXP-279 beacon authenticity status invalid")
    else:
        errors.append("EXP-279 beacon test_only flag missing")

    digest = payload.get("receipt_digest")
    if not isinstance(digest, str) or digest != _receipt_digest(payload):
        errors.append("EXP-279 beacon receipt digest mismatch")

    if freeze_commit_timestamp_utc is not None:
        try:
            frozen = _parse_utc(freeze_commit_timestamp_utc)
        except ValueError:
            errors.append("EXP-279 freeze commit timestamp invalid")
        else:
            if published is not None and published <= frozen:
                errors.append(
                    "EXP-279 beacon must be published strictly after Gate A freeze"
                )

    if checkpoint_seal_created_at_utc is not None:
        try:
            checkpoint_seal = _parse_utc(checkpoint_seal_created_at_utc)
        except ValueError:
            errors.append("EXP-279 checkpoint seal timestamp invalid")
        else:
            if published is not None and published <= checkpoint_seal:
                errors.append(
                    "EXP-279 beacon must be published strictly after checkpoint seal"
                )
    return errors


def derive_exp279_challenge_seed(
    *,
    protocol_digest: str,
    beacon_receipt: dict[str, Any],
    stream: str,
    replicate: int,
    stratum: str,
    freeze_commit_timestamp_utc: str | None = None,
    checkpoint_seal_created_at_utc: str | None = None,
) -> int:
    if not isinstance(protocol_digest, str) or not protocol_digest:
        raise ValueError("protocol_digest is required")
    if stream != CHALLENGE_STREAM:
        raise ValueError("EXP-279 confirmatory stream must be challenge")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}")

    errors = validate_exp279_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
        checkpoint_seal_created_at_utc=checkpoint_seal_created_at_utc,
    )
    if errors:
        raise ValueError("invalid EXP-279 beacon receipt: " + "; ".join(errors))

    # Freeze/checkpoint timestamps are eligibility gates only. They are
    # deliberately not mixed into scientific randomness. The sealed stratum is
    # included so a post-hoc schedule substitution cannot preserve a seed.
    material = (
        f"{protocol_digest}|{beacon_receipt['receipt_digest']}|{EXPERIMENT_ID}|"
        f"{CHALLENGE_STREAM}|{replicate}|{stratum}"
    )
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
