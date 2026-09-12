from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_multiroot_support_stability_v7 import (
    aggregate_exp279_multiroot_support_stability_v7_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate exactly four EXP-279 V7 canonical shard receipts into one frozen budget receipt"
    )
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if len(args.shard) != 4:
        raise SystemExit("V7 budget reducer requires exactly four --shard paths")

    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.shard]
    receipt = aggregate_exp279_multiroot_support_stability_v7_budget(shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "evidence_level": receipt["evidence_level"],
                "decision": receipt["decision"],
                "train_replicates": receipt["train_replicates"],
                "train_cell_classification": receipt["train_cell_classification"],
                "canonical_prevalence_floor": receipt["canonical_prevalence_floor"],
                "artifact_digest": receipt["artifact_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
