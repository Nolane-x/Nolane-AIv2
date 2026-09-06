from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

VALID_EVIDENCE_LEVELS = {"EV-H0", "EV-E1", "EV-E2", "EV-E3", "EV-E4", "EV-E5"}
VALID_DECISIONS = {
    "PROMOTE_TO_NEXT_STAGE",
    "HOLD_UNSTABLE",
    "PRACTICALLY_EQUIVALENT_USE_SIMPLER_RIVAL",
    "KILL_SUBSYSTEM",
    "INVALID_RUN",
    "UNVERIFIED",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_evidence_packet(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("protocol_digest", "code_digest", "claim_id", "evidence_level", "decision"):
        if not packet.get(field):
            errors.append(f"missing {field}")
    if packet.get("evidence_level") not in VALID_EVIDENCE_LEVELS:
        errors.append("invalid evidence_level")
    if packet.get("decision") not in VALID_DECISIONS:
        errors.append("invalid decision")
    if packet.get("evidence_level") in {"EV-E3", "EV-E4", "EV-E5"}:
        raw = packet.get("raw_per_replicate_metrics")
        if not isinstance(raw, list) or not raw:
            errors.append("raw_per_replicate_metrics required for EV-E3+")
    return errors
