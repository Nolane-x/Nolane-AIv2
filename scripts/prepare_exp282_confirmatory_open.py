from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep
from nolane_ai.experiments.exp282_paired_runner import validate_exp282_paired_development
from nolane_ai.experiments.neural_arm_registry import validate_neural_arm_registry
from nolane_ai.protocol.identity import file_sha256, source_tree_digest
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze EXP-282 confirmatory-open preparation without consuming confirmatory data")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument("--protocol-digest-file", type=Path, default=ROOT / "protocols" / "stage_a_v1.sha256")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_verified_protocol(path: Path, digest_path: Path) -> tuple[dict, str]:
    load_and_validate_protocol(path)
    actual = file_sha256(path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    return json.loads(path.read_text(encoding="utf-8")), actual


def _exp282(protocol: dict) -> dict:
    for experiment in protocol.get("experiments", []):
        if experiment.get("experiment_id") == "EXP-282":
            return experiment
    raise RuntimeError("frozen protocol does not contain EXP-282")


def main() -> int:
    args = parse_args()
    protocol, protocol_digest = _load_verified_protocol(args.protocol, args.protocol_digest_file)
    execution = json.loads(args.execution.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))

    execution_errors = validate_exp282_paired_development(execution)
    if execution_errors:
        raise RuntimeError("invalid EXP-282 paired execution artifact: " + "; ".join(execution_errors))
    registry_errors = validate_neural_arm_registry(registry)
    if registry_errors:
        raise RuntimeError("invalid neural arm registry: " + "; ".join(registry_errors))
    if execution.get("protocol_digest") != protocol_digest or registry.get("protocol_digest") != protocol_digest:
        raise RuntimeError("input artifacts do not bind the verified frozen protocol digest")

    payload = build_exp282_confirmatory_prep(
        experiment=_exp282(protocol),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=source_tree_digest(ROOT),
        familywise_alpha=float(protocol["global_sample_size_plan"]["familywise_alpha"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "evidence_level": payload["evidence_level"],
        "decision": payload["decision"],
        "status": payload["status"],
        "pilot_n": payload["pilot_summary"]["n"],
        "confirmatory_n": payload["sample_size_freeze"]["confirmatory_n"],
        "unclamped_required_n": payload["sample_size_freeze"]["unclamped_required_n"],
        "prep_digest": payload["prep_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
