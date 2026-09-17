from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp319_contract import CONTRACT
from nolane_ai.experiments.exp319_training import run_training_chunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen EXP-319 Stage B training chunk")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-receipt")
    parser.add_argument("--checkpoint-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_index not in range(4):
        raise SystemExit("Stage B chunk-index must be in 0..3")
    chunk = CONTRACT.stage_b.max_chunk_steps
    start = args.chunk_index * chunk
    end = min(start + chunk, max(CONTRACT.stage_b.checkpoints))
    parent_checkpoint = args.parent_checkpoint if args.chunk_index else None
    parent_receipt = args.parent_receipt if args.chunk_index else None
    if args.chunk_index and (not parent_checkpoint or not parent_receipt):
        raise SystemExit("continuation Stage B chunk requires parent checkpoint and receipt")
    result = run_training_chunk(
        stage="B_TRAIN",
        arm_id=args.arm,
        root=args.root,
        learning_rate=args.learning_rate,
        run_identity=args.run_identity,
        source_commit_sha=args.source_commit_sha,
        chunk_start_step=start,
        chunk_end_step=end,
        checkpoint_path=args.checkpoint_output,
        receipt_path=args.receipt_output,
        device=args.device,
        parent_checkpoint_path=parent_checkpoint,
        parent_receipt_path=parent_receipt,
    )
    _write_once(args.summary_output, {
        "schema": "EXP319-TRAINING-CHUNK-SUMMARY-V1",
        "stage": "B_TRAIN",
        "chunk_index": args.chunk_index,
        "receipt": result.receipt.to_json_dict(),
        "initial_answer_only_loss": result.initial_answer_only_loss,
        "snapshots": [item.to_json_dict() for item in result.snapshots],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
