from __future__ import annotations

import argparse
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

from nolane_ai.experiments.exp297_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp297_confirmatory_ceremony import (
    seal_exp297_confirmatory_gate_a,
    validate_exp297_confirmatory_gate_a_seal,
)
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    authorize_exp297_confirmatory_execution,
    validate_exp297_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    build_exp297_confirmatory_prep,
    validate_exp297_confirmatory_prep,
)
from nolane_ai.experiments.exp297_paired_runner import validate_exp297_execution
from nolane_ai.experiments.exp297_reconstruction_court import (
    authorize_exp297_confirmatory_reconstruction,
    validate_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and atomically seal EXP-297 confirmatory Gate A without "
            "consuming any beacon or confirmatory challenge data"
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
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze-commit-sha", required=True)
    parser.add_argument("--freeze-commit-timestamp-utc", required=True)
    parser.add_argument("--prep-output", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    parser.add_argument("--reconstruction-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    return parser.parse_args()


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


def _verified_protocol(
    protocol_path: Path,
    digest_path: Path,
) -> tuple[dict[str, Any], str]:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(
            f"protocol digest mismatch: expected {expected!r}, got {actual!r}"
        )
    require_canonical_stage_a_v1_digest(actual)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Stage-A protocol JSON must contain an object")
    return payload, actual


def _exp297(protocol: dict[str, Any]) -> dict[str, Any]:
    for item in protocol.get("experiments") or []:
        if isinstance(item, dict) and item.get("experiment_id") == "EXP-297":
            return item
    raise RuntimeError("canonical Stage-A protocol does not contain EXP-297")


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


def _commit_outputs(
    payloads: tuple[tuple[Path, dict[str, Any]], ...],
) -> None:
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


def main() -> int:
    args = parse_args()
    outputs = (
        args.prep_output,
        args.authorization_output,
        args.reconstruction_output,
        args.seal_output,
    )
    _preflight_outputs(outputs)

    protocol, protocol_digest = _verified_protocol(
        args.protocol,
        args.protocol_digest_file,
    )
    code_tree_digest = source_tree_digest(ROOT)
    execution = _load_json(args.execution, label="EXP-297 DEVELOPMENT execution")
    registry = _load_json(args.registry, label="EXP-297 arm registry")

    _require_valid("EXP-297 DEVELOPMENT execution", validate_exp297_execution(execution))
    if execution.get("code_digest") != code_tree_digest:
        raise RuntimeError(
            "EXP-297 DEVELOPMENT execution code digest does not match the current "
            "Gate A code tree; regenerate the 32+ replicate DEVELOPMENT pilot from "
            "the frozen Gate A source tree before sealing"
        )
    if execution.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-297 DEVELOPMENT execution protocol lineage mismatch")

    prep = build_exp297_confirmatory_prep(
        experiment=_exp297(protocol),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=code_tree_digest,
    )
    _require_valid("EXP-297 confirmatory prep", validate_exp297_confirmatory_prep(prep))

    authorization = authorize_exp297_confirmatory_execution(
        prep_artifact=prep,
        challenge_contract_digest=challenge_contract_digest(),
        evaluator_code_digest=code_tree_digest,
        execution_code_digest=code_tree_digest,
    )
    _require_valid(
        "EXP-297 execution authorization",
        validate_exp297_confirmatory_execution_authorization(authorization),
    )

    reconstruction = authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_code_digest=code_tree_digest,
    )
    _require_valid(
        "EXP-297 reconstruction authorization",
        validate_exp297_confirmatory_reconstruction(reconstruction),
    )

    seal = seal_exp297_confirmatory_gate_a(
        protocol_digest=protocol_digest,
        development_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_authorization=reconstruction,
        code_tree_digest=code_tree_digest,
        freeze_commit_sha=args.freeze_commit_sha,
        freeze_commit_timestamp_utc=args.freeze_commit_timestamp_utc,
    )
    _require_valid(
        "EXP-297 Gate A seal",
        validate_exp297_confirmatory_gate_a_seal(seal),
    )

    _preflight_outputs(outputs)
    _commit_outputs(
        (
            (args.prep_output, prep),
            (args.authorization_output, authorization),
            (args.reconstruction_output, reconstruction),
            (args.seal_output, seal),
        )
    )

    print(
        json.dumps(
            {
                "schema": seal["schema"],
                "status": seal["status"],
                "evidence_level": seal["evidence_level"],
                "decision": seal["decision"],
                "confirmatory_ready": seal["confirmatory_ready"],
                "confirmatory_ready_scope": seal["confirmatory_ready_scope"],
                "confirmatory_data_consumed": seal["confirmatory_data_consumed"],
                "challenge_materialized": seal["challenge_materialized"],
                "decision_rule_executed": seal["decision_rule_executed"],
                "semantic_authority_promoted": seal["semantic_authority_promoted"],
                "code_tree_digest": code_tree_digest,
                "prep_digest": prep["prep_digest"],
                "authorization_digest": authorization["authorization_digest"],
                "reconstruction_digest": reconstruction["reconstruction_digest"],
                "seal_digest": seal["seal_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
