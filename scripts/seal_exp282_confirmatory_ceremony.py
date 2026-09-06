from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp282_confirmatory_ceremony import seal_exp282_confirmatory_ceremony
from nolane_ai.experiments.exp282_confirmatory_execution_court import authorize_exp282_confirmatory_execution
from nolane_ai.experiments.exp282_reconstruction_court import authorize_exp282_confirmatory_reconstruction
from nolane_ai.protocol.identity import file_sha256, require_canonical_stage_a_v1_digest, source_tree_digest
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal EXP-282 confirmatory-open lineage without consuming confirmatory data")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument("--protocol-digest-file", type=Path, default=ROOT / "protocols" / "stage_a_v1.sha256")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_protocol(protocol_path: Path, digest_path: Path) -> str:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_canonical_stage_a_v1_digest(actual)
    return actual


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    execution = _load_json(args.execution)
    prep = _load_json(args.prep)
    ceremony_code_digest = source_tree_digest(ROOT)

    execution_authorization = authorize_exp282_confirmatory_execution(
        prep_artifact=prep,
        execution_code_digest=ceremony_code_digest,
    )
    reconstruction_authorization = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=execution_authorization,
    )
    seal = seal_exp282_confirmatory_ceremony(
        protocol_digest=protocol_digest,
        paired_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=execution_authorization,
        reconstruction_authorization=reconstruction_authorization,
        ceremony_code_digest=ceremony_code_digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(seal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": seal["schema"],
        "status": seal["status"],
        "confirmatory_n": seal["confirmatory_n"],
        "ceremony_seal_digest": seal["ceremony_seal_digest"],
        "confirmatory_data_consumed": seal["confirmatory_data_consumed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
