from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_powered_support_stability_v8_cross_cell import (
    classify_exp279_powered_support_stability_v8_cross_cell,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify frozen EXP-279 V8 powered support train60/train120 receipts"
    )
    parser.add_argument("--train60", type=Path, required=True)
    parser.add_argument("--train120", type=Path, required=True)
    parser.add_argument("--expected-scientific-branch-head", required=True)
    parser.add_argument("--expected-executed-commit", required=True)
    parser.add_argument("--expected-code-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    cells = {
        60: json.loads(args.train60.read_text(encoding="utf-8")),
        120: json.loads(args.train120.read_text(encoding="utf-8")),
    }
    receipt = classify_exp279_powered_support_stability_v8_cross_cell(
        cells,
        expected_scientific_branch_head=args.expected_scientific_branch_head,
        expected_executed_commit=args.expected_executed_commit,
        expected_code_digest=args.expected_code_digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "decision": receipt["decision"],
                "scientific_branch_head": receipt["scientific_branch_head"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
