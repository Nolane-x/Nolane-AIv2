from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import validate_evidence_packet
from nolane_ai.protocol.identity import file_sha256, source_tree_digest

from .analysis import build_smoke_evidence_packet, summarize_bundle
from .harness import run_stage_a_smoke


def execute_stage_a(
    *,
    protocol_path: str | Path,
    digest_path: str | Path,
    code_root: str | Path,
    replicates: int,
    bootstrap_samples: int = 1000,
    root_seed: str | None = None,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path)
    digest_path = Path(digest_path)
    actual_digest = file_sha256(protocol_path)
    expected_digest = digest_path.read_text(encoding="utf-8").strip()
    if actual_digest != expected_digest:
        raise RuntimeError(f"protocol digest mismatch: expected {expected_digest!r}, got {actual_digest}")
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    if raw.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1" or raw.get("status") != "FROZEN_V1":
        raise RuntimeError("runner requires the frozen Stage-A v1 protocol")
    if root_seed is None:
        root_seed = str(raw.get("rng", {}).get("root_seed", "20260906"))
    bundle = run_stage_a_smoke(replicates=replicates, root_seed=root_seed)
    packet = build_smoke_evidence_packet(
        bundle,
        protocol_digest=actual_digest,
        code_digest=source_tree_digest(code_root),
    )
    packet["protocol_id"] = raw["protocol_id"]
    packet["protocol_status"] = raw["status"]
    packet["analysis"] = {
        experiment_id: asdict(summary)
        for experiment_id, summary in summarize_bundle(bundle, bootstrap_samples=bootstrap_samples).items()
    }
    errors = validate_evidence_packet(packet)
    if errors:
        raise RuntimeError("invalid evidence packet: " + "; ".join(errors))
    return packet
