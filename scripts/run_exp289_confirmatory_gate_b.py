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

from nolane_ai.experiments.exp289_beacon import validate_exp289_beacon_receipt
from nolane_ai.experiments.exp289_checkpoint import validate_exp289_checkpoint_receipt
from nolane_ai.experiments.exp289_confirmatory_analysis import (
    build_exp289_confirmatory_analysis,
    validate_exp289_confirmatory_analysis,
)
from nolane_ai.experiments.exp289_confirmatory_ceremony import (
    validate_exp289_confirmatory_gate_a_seal,
)
from nolane_ai.experiments.exp289_confirmatory_executor import (
    execute_exp289_confirmatory_challenge,
    validate_exp289_confirmatory_raw,
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
            "Execute EXP-289 Gate B from an immutable Gate-A seal. Challenge seeds "
            "are derived only from the sealed lineage plus a post-freeze beacon."
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
    parser.add_argument("--beacon", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="require TEST-ONLY synthetic beacon and forbid EV-E3 promotion",
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


def _require_valid(label: str, errors: list[str]) -> None:
    if errors:
        raise RuntimeError(f"invalid {label}: " + "; ".join(errors))


def _enforce_beacon_mode(beacon: dict[str, Any], *, test_only: bool) -> None:
    beacon_test_only = beacon.get("test_only")
    scientific = beacon.get("scientific_evidence_eligible")
    if test_only:
        if beacon_test_only is not True or scientific is not False:
            raise RuntimeError("--test-only requires a TEST-ONLY non-scientific beacon")
        return
    if beacon_test_only is not False or scientific is not True:
        raise RuntimeError(
            "non-test Gate B requires a real beacon explicitly marked "
            "scientific_evidence_eligible"
        )


def _enforce_output_mode(
    raw: dict[str, Any],
    analysis: dict[str, Any],
    *,
    test_only: bool,
) -> None:
    if test_only:
        if raw.get("test_only") is not True or raw.get("scientific_evidence_eligible") is not False:
            raise RuntimeError("EXP-289 TEST-ONLY raw artifact crossed scientific boundary")
        if raw.get("confirmatory_data_consumed") is not False:
            raise RuntimeError("EXP-289 TEST-ONLY raw artifact consumed confirmatory data")
        if analysis.get("evidence_level") != "EV-E2" or analysis.get("decision") != "UNVERIFIED":
            raise RuntimeError("EXP-289 TEST-ONLY analysis attempted scientific promotion")
        if analysis.get("decision_rule_executed") is not False:
            raise RuntimeError("EXP-289 TEST-ONLY analysis executed scientific decision rule")
        return

    if raw.get("test_only") is not False or raw.get("scientific_evidence_eligible") is not True:
        raise RuntimeError("EXP-289 real raw artifact lost scientific eligibility")
    if raw.get("evidence_level") != "EV-E2" or raw.get("decision") != "UNVERIFIED":
        raise RuntimeError("EXP-289 raw executor must remain pre-analysis EV-E2")
    if raw.get("confirmatory_data_consumed") is not True or raw.get("decision_rule_executed") is not False:
        raise RuntimeError("EXP-289 real raw boundary drift")
    if analysis.get("test_only") is not False or analysis.get("scientific_evidence_eligible") is not True:
        raise RuntimeError("EXP-289 real analysis lost scientific eligibility")
    if analysis.get("evidence_level") != "EV-E3":
        raise RuntimeError("EXP-289 real analysis must be EV-E3")
    if analysis.get("decision") not in {
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "KILL_SUBSYSTEM",
    }:
        raise RuntimeError("EXP-289 real analysis decision invalid")
    if analysis.get("confirmatory_data_consumed") is not True:
        raise RuntimeError("EXP-289 real analysis did not bind consumed confirmatory data")
    if analysis.get("decision_rule_executed") is not True:
        raise RuntimeError("EXP-289 real analysis did not execute frozen decision rule")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = (args.raw_output, args.analysis_output)
    _preflight_outputs(outputs)

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    code_tree_digest = source_tree_digest(ROOT)
    seal = _load_json(args.seal, label="EXP-289 Gate-A seal")
    _require_valid("EXP-289 Gate-A seal", validate_exp289_confirmatory_gate_a_seal(seal))
    if (seal.get("lineage") or {}).get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-289 Gate B protocol/seal lineage mismatch")
    if seal.get("code_tree_digest") != code_tree_digest:
        raise RuntimeError(
            "EXP-289 Gate-A seal code tree no longer matches current src/scripts; "
            "production-code drift requires a new pre-beacon freeze"
        )

    checkpoint_receipt = _load_json(
        args.checkpoint_receipt,
        label="EXP-289 checkpoint receipt",
    )
    _require_valid(
        "EXP-289 checkpoint receipt",
        validate_exp289_checkpoint_receipt(checkpoint_receipt),
    )

    beacon = _load_json(args.beacon, label="EXP-289 public beacon receipt")
    beacon_errors = validate_exp289_beacon_receipt(
        beacon,
        freeze_commit_timestamp_utc=seal.get("freeze_commit_timestamp_utc"),
        seal_created_at_utc=seal.get("seal_created_at_utc"),
    )
    if beacon_errors:
        raise RuntimeError("invalid EXP-289 public beacon receipt: " + "; ".join(beacon_errors))
    _enforce_beacon_mode(beacon, test_only=args.test_only)

    raw = execute_exp289_confirmatory_challenge(
        seal=seal,
        beacon_receipt=beacon,
        checkpoint_path=args.checkpoint,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=code_tree_digest,
        executor_code_digest=code_tree_digest,
    )
    _require_valid(
        "EXP-289 confirmatory raw artifact",
        validate_exp289_confirmatory_raw(
            raw,
            seal=seal,
            checkpoint_path=args.checkpoint,
            checkpoint_receipt=checkpoint_receipt,
        ),
    )

    analysis = build_exp289_confirmatory_analysis(
        raw_artifact=raw,
        seal=seal,
        checkpoint_path=args.checkpoint,
        checkpoint_receipt=checkpoint_receipt,
        analysis_code_digest=code_tree_digest,
    )
    _require_valid(
        "EXP-289 confirmatory analysis",
        validate_exp289_confirmatory_analysis(
            analysis,
            raw_artifact=raw,
            seal=seal,
            checkpoint_path=args.checkpoint,
            checkpoint_receipt=checkpoint_receipt,
        ),
    )
    _enforce_output_mode(raw, analysis, test_only=args.test_only)

    _preflight_outputs(outputs)
    _commit_outputs(
        (
            (args.raw_output, raw),
            (args.analysis_output, analysis),
        )
    )

    print(
        json.dumps(
            {
                "raw_schema": raw["schema"],
                "raw_status": raw["status"],
                "analysis_schema": analysis["schema"],
                "analysis_status": analysis["status"],
                "test_only": raw["test_only"],
                "scientific_evidence_eligible": analysis["scientific_evidence_eligible"],
                "evidence_level": analysis["evidence_level"],
                "decision": analysis["decision"],
                "confirmatory_data_consumed": analysis["confirmatory_data_consumed"],
                "challenge_materialized": raw["challenge_materialized"],
                "decision_rule_executed": analysis["decision_rule_executed"],
                "test_only_would_be_decision": analysis.get("test_only_would_be_decision"),
                "raw_artifact_digest": raw["artifact_digest"],
                "analysis_digest": analysis["analysis_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
