from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp319_contract import canonical_json_bytes
from nolane_ai.experiments.exp319_selection import StageBSelectionRecord, select_stage_b_checkpoint_with_disposition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal one EXP-319 Stage B root selection")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[StageBSelectionRecord] = []
    checkpoint_by_step: dict[int, tuple[str, float]] = {}
    for path in args.summaries:
        summary = _load(path)
        if summary.get("schema") != "EXP319-TRAINING-CHUNK-SUMMARY-V1" or summary.get("stage") != "B_TRAIN":
            raise SystemExit(f"not a Stage B chunk summary: {path}")
        receipt = summary["receipt"]
        if str(receipt["arm_id"]) != args.arm or int(receipt["root"]) != args.root:
            continue
        for snapshot in summary.get("snapshots", []):
            step = int(snapshot["step"])
            family_exact = tuple((str(name), float(score)) for name, score in snapshot["family_exact"])
            records.append(StageBSelectionRecord(
                arm_id=args.arm,
                root=args.root,
                step=step,
                train_exact_match=float(snapshot["train_exact_match"]),
                iid_answer_only_loss=float(snapshot["answer_only_loss"]),
                iid_answer_token_accuracy=float(snapshot["answer_token_accuracy"]),
                iid_family_balanced_exact_match=float(snapshot["family_balanced_exact_match"]),
                iid_family_exact=family_exact,
                eos_correctness=float(snapshot["eos_correctness"]),
                invalid_output_rate=float(snapshot["invalid_output_rate"]),
            ))
            checkpoint_by_step[step] = (str(receipt["artifact_digest"]), float(receipt["learning_rate"]))
    result = select_stage_b_checkpoint_with_disposition(records)
    checkpoint_digest, learning_rate = checkpoint_by_step[result.selected.step]
    record_payload = asdict(result.selected)
    selection_digest = hashlib.sha256(canonical_json_bytes({
        "record": record_payload,
        "passes_floor": result.passes_floor,
        "checkpoint_digest": checkpoint_digest,
    })).hexdigest()
    payload = {
        "schema": "EXP319-STAGE-B-ROOT-SELECTION-V1",
        "arm_id": args.arm,
        "root": args.root,
        "learning_rate": learning_rate,
        "passes_floor": result.passes_floor,
        "used_passing_checkpoint_rule": result.used_passing_checkpoint_rule,
        "selected_record": record_payload,
        "seal": {
            "arm_id": args.arm,
            "root": args.root,
            "selected_step": result.selected.step,
            "passes_floor": result.passes_floor,
            "checkpoint_digest": checkpoint_digest,
            "selection_digest": selection_digest,
        },
    }
    _write_once(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
