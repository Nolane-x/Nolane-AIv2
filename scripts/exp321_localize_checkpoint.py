from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp321_contract import canonical_json_bytes
from nolane_ai.experiments.exp321_measure import run_localization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sealed EXP-321 A_FIXED failure localization"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_localization(
        checkpoint_path=args.checkpoint,
        receipt_path=args.receipt,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(result))
        handle.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
