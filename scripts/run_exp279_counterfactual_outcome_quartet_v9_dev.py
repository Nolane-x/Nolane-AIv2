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
    canonical_receipt_bytes,
    validate_shard_receipt,
)
from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9_scientific_runner import (
    run_exp279_counterfactual_outcome_quartet_v9_shard,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DEVELOPMENT-only EXP-279 V9 outcome-quartet shard")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--train-replicates", type=int, choices=(60, 120), required=True)
    parser.add_argument("--canonical-index", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--scientific-branch-head", required=True)
    parser.add_argument("--executed-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _verified_protocol(protocol_path: Path, digest_path: Path) -> str:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_canonical_stage_a_v1_digest(actual)
    return actual


def _write_receipt(output: Path, receipt: dict) -> None:
    payload = canonical_receipt_bytes(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    output.with_name(output.name + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".sha256").exists():
        raise SystemExit(f"output already exists: {args.output}")
    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    receipt = run_exp279_counterfactual_outcome_quartet_v9_shard(
        train_replicates=args.train_replicates,
        canonical_index=args.canonical_index,
        protocol_digest=protocol_digest,
        code_digest=source_tree_digest(ROOT),
        scientific_branch_head=args.scientific_branch_head,
        executed_commit=args.executed_commit,
    )
    errors = validate_shard_receipt(receipt)
    if errors:
        raise RuntimeError("V9 shard receipt validation failed: " + "; ".join(errors))
    _write_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "canonical_index": receipt["canonical_index"],
                "root_classification": receipt["root_classification"],
                "scientific_branch_head": receipt["scientific_branch_head"],
                "train_replicates": receipt["train_replicates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
