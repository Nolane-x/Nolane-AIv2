from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp322_contract import canonical_json_bytes
from nolane_ai.experiments.exp322_evidence import build_final_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce frozen EXP-322 arm evidence")
    parser.add_argument("--hold", required=True)
    parser.add_argument("--decay", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"EXP-322 evidence is not an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_final_evidence(_load(args.hold), _load(args.decay))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
