from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from nolane_ai.experiments.exp279_stop_path_ablation import (
    build_exp279_stop_path_ablation,
    build_exp279_stop_path_sweep,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DEVELOPMENT-only EXP-279 zero-route stop-path cost decomposition receipt"
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        dest="inputs",
        help="Path to an NLM-EXP-279-PAIRED-DEV-EVAL-V1 JSON artifact; repeat for a sweep",
    )
    parser.add_argument("--output", required=True, help="Output receipt JSON path")
    return parser


def _load(path: str) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"input {source} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payloads = [_load(path) for path in args.inputs]
    if len(payloads) == 1:
        receipt = build_exp279_stop_path_ablation(payloads[0])
    else:
        receipt = build_exp279_stop_path_sweep(payloads)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
