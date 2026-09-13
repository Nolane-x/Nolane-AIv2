from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import (
    require_frozen_stage_a_v1_sha256,
    source_tree_digest,
)

STAGE_A_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"
GEOMETRY_MANIFEST = ROOT / "protocols" / "exp298_cross_domain_fidelity_geometry_v1.json"
GEOMETRY_DIGEST = ROOT / "protocols" / "exp298_cross_domain_fidelity_geometry_v1.sha256"
GEOMETRY_SCHEMA = "NLM-EXP-298-CROSS-DOMAIN-FIDELITY-GEOMETRY-V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen DEVELOPMENT-only EXP-298 root or four-root cross reduction"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    root = subparsers.add_parser("root", help="run one canonical EXP-298 root")
    root.add_argument("--canonical-index", type=int, choices=(0, 1, 2, 3), required=True)
    root.add_argument("--output", type=Path, required=True)

    cross = subparsers.add_parser("cross", help="reduce exactly four canonical EXP-298 roots")
    cross.add_argument("--root-receipt", type=Path, action="append", required=True)
    cross.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "cross" and len(args.root_receipt) != 4:
        parser.error("cross mode requires exactly four --root-receipt paths")
    return args


def _repository_head() -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if len(head) != 40:
        raise RuntimeError("repository HEAD is not a full git SHA")
    int(head, 16)
    return head


def _load_frozen_geometry() -> tuple[dict[str, Any], str]:
    payload = json.loads(GEOMETRY_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EXP-298 frozen geometry manifest must be an object")
    observed_digest = canonical_sha256(payload)
    declared_digest = GEOMETRY_DIGEST.read_text(encoding="ascii").strip()
    if observed_digest != declared_digest:
        raise ValueError("EXP-298 frozen geometry digest mismatch")

    protocol_digest = STAGE_A_DIGEST.read_text(encoding="ascii").strip()
    require_frozen_stage_a_v1_sha256(protocol_digest)
    if payload.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("EXP-298 frozen geometry schema mismatch")
    if payload.get("experiment_id") != "EXP-298":
        raise ValueError("EXP-298 frozen geometry experiment mismatch")
    if payload.get("authority_scope") != "DEVELOPMENT_EV_E2_ONLY":
        raise ValueError("EXP-298 frozen geometry authority scope mismatch")
    if payload.get("canonical_indices") != [0, 1, 2, 3]:
        raise ValueError("EXP-298 frozen geometry canonical roots mismatch")
    if payload.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-298 frozen geometry protocol digest mismatch")
    if payload.get("domains") != [
        "code_invariant",
        "causal_diagnosis",
        "grounded_language_ambiguity",
    ]:
        raise ValueError("EXP-298 frozen geometry domain identity mismatch")
    for flag in (
        "scientific_evidence_eligible",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
        "stage_a_protocol_modified",
        "exp300_authorized",
    ):
        if payload.get(flag) is not False:
            raise ValueError(f"EXP-298 forbidden frozen geometry flag enabled: {flag}")
    if payload.get("successor_scope_if_recurrent") != "DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY":
        raise ValueError("EXP-298 frozen successor scope mismatch")
    return payload, declared_digest


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_temp(parent: Path, prefix: str, payload: bytes) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=parent)
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_canonical_atomic(output: Path, receipt: dict[str, Any]) -> None:
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"output or sidecar already exists: {output}")

    payload = _canonical_bytes(receipt)
    digest_payload = (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii")
    receipt_tmp = _write_temp(output.parent, f".{output.name}.", payload)
    sidecar_tmp = _write_temp(output.parent, f".{sidecar.name}.", digest_payload)
    published_output = False
    try:
        os.link(receipt_tmp, output)
        published_output = True
        try:
            os.link(sidecar_tmp, sidecar)
        except BaseException:
            output.unlink(missing_ok=True)
            published_output = False
            raise
    except FileExistsError as exc:
        if published_output:
            output.unlink(missing_ok=True)
        raise FileExistsError(f"output or sidecar already exists: {output}") from exc
    finally:
        receipt_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)


def _load_root(path: Path) -> dict[str, Any]:
    from nolane_ai.experiments.exp298_paired_runner import validate_exp298_root

    geometry, geometry_digest = _load_frozen_geometry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"EXP-298 root receipt must be an object: {path}")
    errors = validate_exp298_root(payload)
    if errors:
        raise ValueError(f"invalid EXP-298 root receipt {path}: {'; '.join(errors)}")
    if payload.get("geometry_digest") != geometry_digest:
        raise ValueError(f"EXP-298 root geometry digest mismatch: {path}")
    if payload.get("protocol_digest") != geometry["protocol_digest"]:
        raise ValueError(f"EXP-298 root protocol digest mismatch: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        observed = sidecar.read_text(encoding="ascii").strip()
        if observed != expected:
            raise ValueError(f"EXP-298 root sidecar mismatch: {path}")
    return payload


def _run_root(canonical_index: int) -> dict[str, Any]:
    from nolane_ai.experiments.exp298_paired_runner import run_exp298_root

    geometry, geometry_digest = _load_frozen_geometry()
    protocol_digest = str(geometry["protocol_digest"])
    return run_exp298_root(
        root_seed=f"{geometry['root_seed_prefix']}-root-{canonical_index}",
        canonical_index=canonical_index,
        eval_replicates=int(geometry["eval_replicates"]),
        eval_start_replicate=int(geometry["eval_start_replicate"]),
        d_model=int(geometry["d_model"]),
        hidden_size=int(geometry["hidden_size"]),
        target_parameters=int(geometry["target_parameters"]),
        max_exact_probes=int(geometry["max_exact_probes"]),
        protocol_digest=protocol_digest,
        geometry_digest=geometry_digest,
        code_digest=source_tree_digest(ROOT),
        repository_head=_repository_head(),
    )


def main() -> int:
    args = parse_args()
    if args.mode == "root":
        from nolane_ai.experiments.exp298_paired_runner import validate_exp298_root

        receipt = _run_root(args.canonical_index)
        errors = validate_exp298_root(receipt)
        if errors:
            raise RuntimeError("EXP-298 root validation failed: " + "; ".join(errors))
    else:
        from nolane_ai.experiments.exp298_cross_reducer import (
            reduce_exp298_roots,
            validate_exp298_cross,
        )

        roots = [_load_root(path) for path in args.root_receipt]
        receipt = reduce_exp298_roots(roots)
        errors = validate_exp298_cross(receipt)
        if errors:
            raise RuntimeError("EXP-298 cross validation failed: " + "; ".join(errors))

    _write_canonical_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "decision": receipt["decision"],
                "mode": args.mode,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
