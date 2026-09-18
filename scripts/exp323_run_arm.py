from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from nolane_ai.experiments.exp323_contract import canonical_json_bytes
from nolane_ai.experiments.exp323_identity import (
    Exp323ExecutionIdentity,
    validate_exp323_execution_identity,
)
from nolane_ai.experiments.exp323_runtime import run_intervention_arm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen EXP-323 continuation arm")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--parent-exp322", required=True)
    parser.add_argument("--arm", choices=("HOLD_5E5", "DECAY_2P5E5"), required=True)
    parser.add_argument("--execution-identity", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_identity = json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    expected = {item.name for item in fields(Exp323ExecutionIdentity)}
    if set(raw_identity) != expected:
        raise SystemExit("EXP-323 execution identity field mismatch")
    identity = Exp323ExecutionIdentity(**raw_identity)
    validate_exp323_execution_identity(identity)
    payload = run_intervention_arm(
        checkpoint_path=args.checkpoint,
        receipt_path=args.receipt,
        parent_exp322_path=args.parent_exp322,
        arm=args.arm,
        execution_identity=identity,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
