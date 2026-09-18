from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp323_contract import canonical_json_bytes
from nolane_ai.experiments.exp323_runtime import (
    reconstruct_exp322_decay_state,
    validate_exp322_parent_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify exact EXP-323 reconstruction authority without post-2048 updates"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--parent-exp322", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parent = json.loads(Path(args.parent_exp322).read_text(encoding="utf-8"))
    if not isinstance(parent, dict):
        raise SystemExit("EXP-323 parent evidence must be a JSON object")
    validate_exp322_parent_evidence(parent)

    state = reconstruct_exp322_decay_state(
        checkpoint_path=args.checkpoint,
        receipt_path=args.receipt,
    )
    payload = {
        "schema": "EXP323-REPLAY-AUTHORITY-RECEIPT-V1",
        **state.reconstruction,
        "post_2048_optimizer_steps": 0,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.write(b"\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
