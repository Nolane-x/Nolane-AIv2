from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp321_contract import (
    canonical_json_bytes,
    preregistration_digest,
    preregistration_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify frozen EXP-321 preregistration")
    parser.add_argument("--json", required=True)
    parser.add_argument("--sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path = Path(args.json)
    sha_path = Path(args.sha256)
    expected = canonical_json_bytes(preregistration_payload()) + b"\n"
    if json_path.read_bytes() != expected:
        raise SystemExit("EXP-321 preregistration JSON mismatch")
    digest = hashlib.sha256(expected).hexdigest()
    if digest != preregistration_digest():
        raise SystemExit("EXP-321 preregistration digest mismatch")
    expected_sidecar = f"{digest}  {json_path.as_posix()}\n".encode("ascii")
    if sha_path.read_bytes() != expected_sidecar:
        raise SystemExit("EXP-321 preregistration sidecar mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
