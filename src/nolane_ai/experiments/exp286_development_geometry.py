from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "NLM-EXP-286-DEVELOPMENT-GEOMETRY-V1"
AUTHORITY_SCOPE = "DEVELOPMENT_PILOT_ONLY"


def load_exp286_development_geometry(
    manifest_path: str | Path,
    digest_path: str | Path,
    *,
    protocol_digest: str,
) -> tuple[dict[str, Any], str]:
    manifest_path = Path(manifest_path)
    digest_path = Path(digest_path)
    raw = manifest_path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = digest_path.read_text(encoding="utf-8").strip()
    if actual_digest != expected_digest:
        raise ValueError(
            f"EXP-286 geometry digest mismatch: expected {expected_digest!r}, got {actual_digest!r}"
        )

    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != "EXP-286":
        raise ValueError("EXP-286 development geometry identity drift")
    if payload.get("authority_scope") != AUTHORITY_SCOPE:
        raise ValueError("EXP-286 development geometry authority scope drift")
    if payload.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-286 development geometry protocol lineage mismatch")
    if payload.get("predeclaration") != "BEFORE_AUTHORITATIVE_DEVELOPMENT_RUN":
        raise ValueError("EXP-286 development geometry predeclaration boundary drift")
    if payload.get("confirmatory_authority") is not False:
        raise ValueError("EXP-286 development geometry cannot claim confirmatory authority")
    if payload.get("challenge_materialized") is not False:
        raise ValueError("EXP-286 development geometry cannot materialize challenge entropy")
    if payload.get("pilot_reuse_as_confirmatory") is not False:
        raise ValueError("EXP-286 development geometry cannot authorize pilot reuse")
    if "beacon" in payload or "challenge_seed" in payload:
        raise ValueError("EXP-286 development geometry cannot contain confirmatory entropy")

    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("EXP-286 development geometry payload is missing")
    if geometry.get("tiny") is not False:
        raise ValueError("EXP-286 authoritative DEVELOPMENT geometry cannot be tiny smoke")
    try:
        if int(geometry["eval_replicates"]) < 32:
            raise ValueError("EXP-286 authoritative DEVELOPMENT geometry requires at least 32 eval replicates")
        if int(geometry["eval_start_replicate"]) < int(geometry["train_replicates"]):
            raise ValueError("EXP-286 DEVELOPMENT train/eval lineage must be disjoint")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("EXP-286"):
            raise
        raise ValueError("EXP-286 development geometry counts are invalid") from exc

    return dict(geometry), actual_digest
