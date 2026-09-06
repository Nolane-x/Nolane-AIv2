from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp282_confirmatory_ceremony import (
    execute_exp282_confirmatory_ceremony,
    validate_exp282_confirmatory_ceremony_result,
)
from nolane_ai.protocol.identity import file_sha256, require_canonical_stage_a_v1_digest, source_tree_digest
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a previously sealed EXP-282 confirmatory-open ceremony")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument("--protocol-digest-file", type=Path, default=ROOT / "protocols" / "stage_a_v1.sha256")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
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
    outputs = (args.raw_output, args.analysis_output, args.bundle_output)
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None:
        raise SystemExit(f"output already exists: {existing}")

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    bundle = execute_exp282_confirmatory_ceremony(
        protocol=protocol,
        protocol_digest=protocol_digest,
        ceremony_seal=_load_json(args.seal),
        paired_execution_artifact=_load_json(args.execution),
        checkpoint_path=args.checkpoint,
        ceremony_code_digest=source_tree_digest(ROOT),
    )
    errors = validate_exp282_confirmatory_ceremony_result(bundle)
    if errors:
        raise RuntimeError("invalid ceremony result before persistence: " + "; ".join(errors))

    raw = bundle["artifacts"]["raw"]
    analysis = bundle["artifacts"]["analysis"]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    args.analysis_output.write_text(json.dumps(analysis, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    args.bundle_output.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "schema": bundle["schema"],
        "status": bundle["status"],
        "decision": bundle["decision"],
        "evidence_level": bundle["evidence_level"],
        "raw_artifact_digest": raw["artifact_digest"],
        "analysis_digest": analysis["analysis_digest"],
        "ceremony_result_digest": bundle["ceremony_result_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
