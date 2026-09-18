from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp319_stage_a_postmortem import (
    analyze_stage_a_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only EXP-319D1 Stage-A effort-path postmortem"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--scientific-run-id", type=int, required=True)
    parser.add_argument("--expected-selection-authority-digest", required=True)
    parser.add_argument("--expected-source-commit-sha", required=True)
    parser.add_argument("--expected-run-identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_stage_a_checkpoint(
        checkpoint_path=args.checkpoint,
        receipt_path=args.receipt,
        selection_path=args.selection,
        scientific_run_id=args.scientific_run_id,
        expected_selection_authority_digest=args.expected_selection_authority_digest,
        expected_source_commit_sha=args.expected_source_commit_sha,
        expected_run_identity=args.expected_run_identity,
        device=args.device,
    )
    _write_once(args.output, report.to_json_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
