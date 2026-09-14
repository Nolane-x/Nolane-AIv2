from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nolane_ai.experiments.exp301r_recovery import (
    build_cross_root_recovery_envelope,
    write_once_json,
)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"EXP-301R requires workflow-bound {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize EXP-301R recovery provenance")
    parser.add_argument("--cross-root-evidence", required=True)
    parser.add_argument("--root-receipts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = _env("EXP301R_RUN_ID")
    workflow_digest = _env("EXP301R_WORKFLOW_DIGEST")
    cross_root = json.loads(Path(args.cross_root_evidence).read_text(encoding="utf-8"))
    paths = tuple(sorted(Path(args.root_receipts_dir).rglob("root-recovery-receipt.json")))
    if len(paths) != 4:
        raise SystemExit(f"EXP-301R finalizer requires exactly 4 root recovery receipts, found {len(paths)}")
    receipts = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    envelope = build_cross_root_recovery_envelope(
        cross_root_artifact=cross_root,
        root_recovery_receipts=receipts,
        run_id=run_id,
        workflow_digest=workflow_digest,
    )
    write_once_json(Path(args.output_dir) / "exp301r-cross-root-recovery-envelope.json", envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
