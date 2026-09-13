from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp287_receipts import (
    canonical_json_bytes,
    reduce_exp287_cross_root,
    validate_exp287_cross_receipt,
    validate_exp287_root_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce exactly four frozen EXP-287 DEVELOPMENT root receipts"
    )
    parser.add_argument("--root-receipts", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_root_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"EXP-287 root receipt must be a JSON object: {path}")
    canonical = canonical_json_bytes(parsed)
    if raw != canonical:
        raise ValueError(f"EXP-287 root receipt is not canonical JSON: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"EXP-287 root receipt sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8")
    actual = hashlib.sha256(raw).hexdigest() + "\n"
    if expected != actual:
        raise ValueError(f"EXP-287 root receipt sidecar mismatch: {path}")
    validate_exp287_root_receipt(parsed)
    return parsed


def _write_temp(parent: Path, prefix: str, payload: bytes) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=parent)
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_receipt_atomic(output: Path, receipt: dict[str, Any]) -> None:
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"output or sidecar already exists: {output}")
    payload = canonical_json_bytes(receipt)
    digest_payload = (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii")
    receipt_tmp = _write_temp(output.parent, f".{output.name}.", payload)
    sidecar_tmp = _write_temp(output.parent, f".{sidecar.name}.", digest_payload)
    published_output = False
    try:
        os.link(receipt_tmp, output)
        published_output = True
        try:
            os.link(sidecar_tmp, sidecar)
        except BaseException:
            output.unlink(missing_ok=True)
            published_output = False
            raise
    except FileExistsError as exc:
        if published_output:
            output.unlink(missing_ok=True)
        raise FileExistsError(f"output or sidecar already exists: {output}") from exc
    finally:
        receipt_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    roots = [_load_root_receipt(path) for path in args.root_receipts]
    receipt = reduce_exp287_cross_root(roots)
    validate_exp287_cross_receipt(receipt)
    _write_receipt_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "decision": receipt["decision"],
                "successor_design_authorized": receipt["successor_design_authorized"],
                "authorization_scope": receipt["authorization_scope"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
