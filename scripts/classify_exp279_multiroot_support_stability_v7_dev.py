from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_multiroot_support_stability_v7_cross_cell import (
    classify_exp279_multiroot_support_stability_v7_cross_cell,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the frozen EXP-279 V7 train60/train120 support receipts"
    )
    parser.add_argument("--train60", type=Path, required=True)
    parser.add_argument("--train120", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    train60 = json.loads(args.train60.read_text(encoding="utf-8"))
    train120 = json.loads(args.train120.read_text(encoding="utf-8"))
    receipt = classify_exp279_multiroot_support_stability_v7_cross_cell(
        {60: train60, 120: train120}
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "evidence_level": receipt["evidence_level"],
                "decision": receipt["decision"],
                "artifact_digest": receipt["artifact_digest"],
                "successor_design_authorized": receipt["successor_design_authorized"],
                "authorization_scope": receipt["authorization_scope"],
                "future_probe_episodes_per_canonical_root": receipt[
                    "future_probe_episodes_per_canonical_root"
                ],
                "fresh_evaluation_lineage_may_be_reserved": receipt[
                    "fresh_evaluation_lineage_may_be_reserved"
                ],
                "fresh_evaluation_lineage_consumed": receipt[
                    "fresh_evaluation_lineage_consumed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
