from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp323r_reconstruction import (
    materialize_verified_reconstruction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize an exact EXP-323 step-2048 reconstruction candidate"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--parent-exp322", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempt-index", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    verified = materialize_verified_reconstruction(
        checkpoint_path=args.checkpoint,
        receipt_path=args.receipt,
        parent_exp322_path=args.parent_exp322,
        output_checkpoint_path=out / "reconstruction.pt",
        output_receipt_path=out / "reconstruction-receipt.json",
        output_diagnostic_path=out / "diagnostic.json",
        attempt_index=args.attempt_index,
    )
    print(json.dumps({"attempt_index": args.attempt_index, "verified": verified}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
