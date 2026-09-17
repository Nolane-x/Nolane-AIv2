from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp319_contract import DIAGNOSTIC_ARMS, canonical_json_bytes
from nolane_ai.experiments.exp319_selection import StageASelectionRecord, select_stage_a_learning_rate_with_disposition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal EXP-319 Stage A learning-rate selections")
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
    records: list[StageASelectionRecord] = []
    for path in args.summaries:
        summary = _load(path)
        if summary.get("schema") != "EXP319-TRAINING-CHUNK-SUMMARY-V1" or summary.get("stage") != "A_SANITY":
            raise SystemExit(f"not a Stage A chunk summary: {path}")
        receipt = summary["receipt"]
        for snapshot in summary.get("snapshots", []):
            if int(snapshot["step"]) != 1024:
                continue
            records.append(StageASelectionRecord(
                arm_id=str(receipt["arm_id"]),
                root=int(receipt["root"]),
                learning_rate=float(receipt["learning_rate"]),
                step=int(snapshot["step"]),
                initial_answer_only_loss=float(summary["initial_answer_only_loss"]),
                answer_only_loss=float(snapshot["answer_only_loss"]),
                answer_token_accuracy=float(snapshot["answer_token_accuracy"]),
                exact_match=float(snapshot["exact_match"]),
                eos_correctness=float(snapshot["eos_correctness"]),
                invalid_output_rate=float(snapshot["invalid_output_rate"]),
                nonfinite_events=int(snapshot["nonfinite_events"]),
            ))
    sealed = []
    for arm_id in DIAGNOSTIC_ARMS:
        result = select_stage_a_learning_rate_with_disposition(record for record in records if record.arm_id == arm_id)
        record_payload = asdict(result.selected)
        digest = hashlib.sha256(canonical_json_bytes({"record": record_payload, "passes_floor": result.passes_floor})).hexdigest()
        sealed.append({
            "arm_id": arm_id,
            "passes_floor": result.passes_floor,
            "selected_learning_rate": result.selected.learning_rate,
            "selected_record": record_payload,
            "selection_digest": digest,
        })
    authority = {
        "schema": "EXP319-STAGE-A-SELECTION-AUTHORITY-V1",
        "selections": sealed,
    }
    authority["authority_digest"] = hashlib.sha256(canonical_json_bytes(authority)).hexdigest()
    _write_once(args.output, authority)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
