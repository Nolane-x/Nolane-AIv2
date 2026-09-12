from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9_receipts import (
    build_budget_receipt,
    canonical_receipt_bytes,
    validate_shard_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate four frozen EXP-279 V9 shard receipts")
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_receipt(output: Path, receipt: dict) -> None:
    payload = canonical_receipt_bytes(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    output.with_name(output.name + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="utf-8",
    )


def _load_verified(path: Path) -> dict:
    payload = path.read_bytes()
    sidecar = path.with_name(path.name + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"V9 shard sidecar mismatch for {path}: expected {expected}, got {actual}")
    receipt = json.loads(payload.decode("utf-8"))
    if payload != canonical_receipt_bytes(receipt):
        raise RuntimeError(f"V9 shard receipt is not canonical JSON: {path}")
    errors = validate_shard_receipt(receipt)
    if errors:
        raise RuntimeError(f"V9 shard receipt invalid at {path}: " + "; ".join(errors))
    return receipt


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".sha256").exists():
        raise SystemExit(f"output already exists: {args.output}")
    if len(args.shard) != 4:
        raise SystemExit("V9 budget reducer requires exactly four --shard inputs")
    shards = [_load_verified(path) for path in args.shard]
    receipt = build_budget_receipt(shards)
    _write_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "classification": receipt["classification"],
                "train_replicates": receipt["train_replicates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
