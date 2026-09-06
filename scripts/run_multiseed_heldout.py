from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.model.config import NLMConfig
from nolane_ai.protocol.identity import file_sha256, source_tree_digest
from nolane_ai.protocol.schema import load_and_validate_protocol
from nolane_ai.training.multiseed import run_multiseed_heldout_development

PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def _verified_protocol_digest() -> str:
    load_and_validate_protocol(PROTOCOL)
    actual = file_sha256(PROTOCOL)
    expected = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EV-E2 multi-seed held-out development analysis")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variables", type=int, default=4)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--root-seed", default="20260906-multiseed-heldout")
    parser.add_argument("--train-start-replicate", type=int, default=0)
    parser.add_argument("--eval-start-replicate", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = NLMConfig.stage_a_pilot_tiny_for_tests() if args.tiny else NLMConfig.stage_a_pilot_16m()
    payload = run_multiseed_heldout_development(
        config=config,
        root_seed=args.root_seed,
        seeds=args.seeds,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        variables=args.variables,
        constraints=args.constraints,
        bootstrap_samples=args.bootstrap_samples,
        protocol_digest=_verified_protocol_digest(),
        code_digest=source_tree_digest(ROOT),
        train_start_replicate=args.train_start_replicate,
        eval_start_replicate=args.eval_start_replicate,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "evidence_level": payload["evidence_level"],
        "decision": payload["decision"],
        "seed_count": payload["seed_count"],
        "successful_seed_count": payload["successful_seed_count"],
        "failed_seed_count": payload["failed_seed_count"],
        "aggregate_valid": payload["aggregate_valid"],
        "multiseed_digest": payload["multiseed_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
