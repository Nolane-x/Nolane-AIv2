from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp282_confirmatory_executor import execute_exp282_confirmatory_open
from nolane_ai.protocol.identity import (
    file_sha256,
    require_frozen_stage_a_v1_sha256,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a sealed EV-E2 EXP-282 confirmatory-open raw lane from an authorized checkpoint"
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_protocol(protocol_path: Path, digest_path: Path) -> tuple[dict, str]:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_frozen_stage_a_v1_sha256(actual)
    return _load_json(protocol_path), actual


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    execution = _load_json(args.execution)
    reconstruction = _load_json(args.reconstruction)
    raw = execute_exp282_confirmatory_open(
        protocol=protocol,
        protocol_digest=protocol_digest,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        checkpoint_path=args.checkpoint,
        executor_code_digest=source_tree_digest(ROOT),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_digest": raw["artifact_digest"],
                "confirmatory_n": raw["confirmatory_n"],
                "schema": raw["schema"],
                "status": raw["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
