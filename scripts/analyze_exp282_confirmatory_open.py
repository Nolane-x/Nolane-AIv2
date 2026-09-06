from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an already executed EXP-282 confirmatory-open raw artifact under the frozen Stage-A court"
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
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
    require_canonical_stage_a_v1_digest(actual)
    return _load_json(protocol_path), actual


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    prep = _load_json(args.prep)
    execution = _load_json(args.execution)
    reconstruction = _load_json(args.reconstruction)
    raw = _load_json(args.raw)

    result = build_exp282_confirmatory_analysis(
        protocol=protocol,
        protocol_digest=protocol_digest,
        prep_artifact=prep,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        raw_artifact=raw,
        analysis_code_digest=source_tree_digest(ROOT),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    primary = result["primary_effect"]
    protected = result["protected_endpoints"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "evidence_level": result["evidence_level"],
                "decision": result["decision"],
                "one_sided_lower_95": primary["one_sided_lower_95"],
                "one_sided_upper_95": primary["one_sided_upper_95"],
                "brier_guard_pass": protected["brier"]["pass"],
                "compute_guard_pass": protected["compute"]["pass"],
                "analysis_digest": result["analysis_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
