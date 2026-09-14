from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nolane_ai.experiments.exp301_ceremony import materialize_root_challenge
from nolane_ai.experiments.exp301_evidence import _selection_from_raw
from nolane_ai.experiments.exp301_runner import load_frozen_implementation_identity
from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_IMPLEMENTATION_DIGEST,
    build_challenge_manifest,
    recovery_beacon,
    write_once_json,
)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"EXP-301R requires workflow-bound {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the EXP-301R challenge identity")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-implementation-identity", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = int(_env("EXP301R_ROOT"))
    run_id = _env("EXP301R_RUN_ID")
    workflow_digest = _env("EXP301R_WORKFLOW_DIGEST")

    frozen = load_frozen_implementation_identity(args.frozen_implementation_identity)
    if frozen.frozen_implementation_digest != FROZEN_IMPLEMENTATION_DIGEST:
        raise SystemExit("EXP-301R frozen implementation digest mismatch")

    selection_raw = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    selection = _selection_from_raw(selection_raw)
    if selection.root != root:
        raise SystemExit("selection manifest root mismatch")

    beacon = recovery_beacon(run_id)
    challenge = materialize_root_challenge(
        root=root,
        beacon=beacon,
        frozen_implementation_digest=frozen.frozen_implementation_digest,
    )
    manifest = build_challenge_manifest(
        challenge,
        run_id=run_id,
        workflow_digest=workflow_digest,
        selection_manifest_digest=selection.selection_manifest_digest,
    )
    write_once_json(Path(args.output_dir) / f"root-{root}" / "challenge-manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
