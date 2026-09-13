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

from nolane_ai.experiments.exp287_learned_conflict_localization import (
    run_exp287_root_development,
)
from nolane_ai.experiments.exp287_receipts import (
    PROTOCOL_DIGEST,
    build_exp287_root_receipt,
    canonical_json_bytes,
    validate_exp287_root_receipt,
)
from nolane_ai.protocol.identity import source_tree_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen DEVELOPMENT-only EXP-287 canonical root"
    )
    parser.add_argument("--canonical-index", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    primitive = run_exp287_root_development(
        args.canonical_index,
        protocol_digest=PROTOCOL_DIGEST,
    )
    primitive["code_digest"] = source_tree_digest(ROOT)
    receipt = build_exp287_root_receipt(primitive)
    validate_exp287_root_receipt(receipt)
    _write_receipt_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "canonical_index": receipt["canonical_index"],
                "classification": receipt["decision"]["classification"],
                "code_digest": receipt["code_digest"],
                "output": str(args.output),
                "protocol_digest": receipt["protocol_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
