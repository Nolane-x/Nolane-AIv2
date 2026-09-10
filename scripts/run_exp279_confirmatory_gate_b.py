from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_confirmatory_ceremony import (
    execute_exp279_gate_b_ceremony,
    validate_exp279_gate_b_ceremony,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def _reject_raw_seed_argument(argv: Sequence[str]) -> None:
    forbidden = [arg for arg in argv if arg == "--seed" or arg.startswith("--seed=")]
    if forbidden:
        raise SystemExit("unrecognized arguments: " + " ".join(forbidden))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    _reject_raw_seed_argument(values)
    parser = argparse.ArgumentParser(
        description=(
            "Execute frozen EXP-279 Gate B from an explicitly supplied post-freeze "
            "beacon receipt. Challenge seeds are derived internally from the sealed "
            "lineage; this command exposes no raw seed surface."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.json",
    )
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--beacon", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--test-only",
        action="store_true",
        help="require a TEST-ONLY synthetic beacon and forbid EV-E3 promotion",
    )
    mode.add_argument(
        "--arm-scientific-lane",
        action="store_true",
        help=(
            "explicitly arm the real scientific lane; requires a scientifically "
            "eligible post-freeze beacon receipt"
        ),
    )
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    return parser.parse_args(values)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} JSON must contain an object")
    return payload


def _verified_protocol(protocol_path: Path, digest_path: Path) -> str:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(
            f"protocol digest mismatch: expected {expected!r}, got {actual!r}"
        )
    require_canonical_stage_a_v1_digest(actual)
    return actual


def _preflight_outputs(paths: tuple[Path, ...]) -> None:
    if len(set(paths)) != len(paths):
        raise SystemExit("output paths must be distinct")
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise SystemExit(f"output already exists: {existing}")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _stage_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _commit_outputs(payloads: tuple[tuple[Path, dict[str, Any]], ...]) -> None:
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, payload in payloads:
            staged.append((_stage_payload(destination, payload), destination))
        for temp_path, destination in staged:
            os.link(temp_path, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)


def _sealed_executor_digest(seal: dict[str, Any]) -> str:
    authorization = seal.get("authorization_snapshot")
    if not isinstance(authorization, dict):
        raise RuntimeError("EXP-279 Gate A seal authorization snapshot missing")
    machinery = authorization.get("machinery_digests")
    if not isinstance(machinery, dict):
        raise RuntimeError("EXP-279 Gate A machinery digest snapshot missing")
    digest = machinery.get("executor_code_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("EXP-279 frozen executor code digest missing")
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = (args.raw_output, args.analysis_output)
    _preflight_outputs(outputs)

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    seal = _load_json(args.seal, label="EXP-279 Gate A seal")
    reconstruction = _load_json(
        args.reconstruction,
        label="EXP-279 reconstruction authorization",
    )
    beacon = _load_json(args.beacon, label="EXP-279 public beacon receipt")
    checkpoint_receipt = _load_json(
        args.checkpoint_receipt,
        label="EXP-279 checkpoint receipt",
    )

    authorization = seal.get("authorization_snapshot")
    if not isinstance(authorization, dict):
        raise RuntimeError("EXP-279 Gate A seal authorization snapshot missing")
    if authorization.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-279 Gate B protocol/seal lineage mismatch")
    if reconstruction.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-279 Gate B protocol/reconstruction lineage mismatch")

    current_source = source_tree_digest(ROOT)
    executor_code_digest = _sealed_executor_digest(seal)

    def publish_raw(raw: dict[str, Any]) -> None:
        # The validated raw artifact is the first durable post-beacon record.
        # Publish it atomically before INVALID_RUN handling and before analysis so
        # no later failure can erase first-attempt evidence.
        _preflight_outputs((args.raw_output,))
        _commit_outputs(((args.raw_output, raw),))

    ceremony = execute_exp279_gate_b_ceremony(
        seal=seal,
        reconstruction_authorization=reconstruction,
        beacon_receipt=beacon,
        checkpoint_path=args.checkpoint,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source,
        executor_code_digest=executor_code_digest,
        test_only=bool(args.test_only),
        arm_scientific_lane=bool(args.arm_scientific_lane),
        raw_publisher=publish_raw,
    )
    ceremony_errors = validate_exp279_gate_b_ceremony(
        ceremony,
        checkpoint_path=args.checkpoint,
        checkpoint_receipt=checkpoint_receipt,
    )
    if ceremony_errors:
        raise RuntimeError(
            "invalid EXP-279 Gate B ceremony: " + "; ".join(ceremony_errors)
        )

    raw = ceremony["raw"]
    analysis = ceremony["analysis"]
    persisted_raw = _load_json(args.raw_output, label="persisted EXP-279 raw evidence")
    if persisted_raw.get("artifact_digest") != raw.get("artifact_digest") or persisted_raw != raw:
        raise RuntimeError("persisted EXP-279 raw evidence differs from analyzed raw artifact")

    _preflight_outputs((args.analysis_output,))
    _commit_outputs(((args.analysis_output, analysis),))

    print(
        json.dumps(
            {
                "ceremony_status": ceremony["status"],
                "test_only": ceremony["test_only"],
                "scientific_lane_armed": ceremony["scientific_lane_armed"],
                "raw_status": raw["status"],
                "analysis_status": analysis["status"],
                "evidence_level": analysis["evidence_level"],
                "decision": analysis["decision"],
                "scientific_evidence_eligible": analysis[
                    "scientific_evidence_eligible"
                ],
                "confirmatory_data_consumed": analysis[
                    "confirmatory_data_consumed"
                ],
                "synthetic_challenge_data_consumed": analysis.get(
                    "synthetic_challenge_data_consumed"
                ),
                "decision_rule_executed": analysis["decision_rule_executed"],
                "test_only_would_be_decision": analysis.get(
                    "test_only_would_be_decision"
                ),
                "raw_artifact_digest": raw["artifact_digest"],
                "analysis_digest": analysis["analysis_digest"],
                "ceremony_digest": ceremony["ceremony_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
