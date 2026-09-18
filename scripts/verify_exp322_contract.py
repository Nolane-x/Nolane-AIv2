from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp322_contract import (
    canonical_json_bytes,
    preregistration_digest,
    preregistration_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify frozen EXP-322 preregistration")
    parser.add_argument("--json", required=True)
    parser.add_argument("--sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    if payload != preregistration_payload():
        raise SystemExit("EXP-322 preregistration payload does not match frozen contract")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest != preregistration_digest():
        raise SystemExit("EXP-322 preregistration implementation digest mismatch")
    sidecar = Path(args.sha256).read_text(encoding="ascii").strip().split()[0]
    if sidecar != digest:
        raise SystemExit("EXP-322 preregistration digest mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
