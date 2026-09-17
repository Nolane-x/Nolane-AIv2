from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp319_contract import canonical_json_bytes, preregistration_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify frozen EXP-319 preregistration")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--digest-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    expected = preregistration_payload()
    if payload != expected:
        raise SystemExit("EXP-319 preregistration payload does not match frozen contract")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    sidecar = Path(args.digest_file).read_text(encoding="ascii").strip().split()[0]
    if sidecar != digest:
        raise SystemExit("EXP-319 preregistration digest mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
