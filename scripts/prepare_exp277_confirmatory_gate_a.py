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

from nolane_ai.experiments.exp277_checkpoint import (
    build_exp277_trained_checkpoint,
    validate_exp277_checkpoint_receipt,
)
from nolane_ai.experiments.exp277_confirmatory_authorization import (
    build_exp277_gate_a_authorization,
    build_exp277_gate_a_seal,
    validate_exp277_gate_a_authorization,
    validate_exp277_gate_a_seal,
)
from nolane_ai.experiments.exp277_confirmatory_prep import (
    build_exp277_confirmatory_prep,
    validate_exp277_confirmatory_prep,
)
from nolane_ai.experiments.exp277_paired_runner import validate_exp277_paired_development
from nolane_ai.experiments.exp277_reconstruction_court import (
    build_exp277_reconstruction_authorization,
    validate_exp277_reconstruction_authorization,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


_FORBIDDEN_PREFREEZE_FLAGS = ("--seed", "--beacon")


def _reject_prefreeze_entropy_arguments(argv: Sequence[str]) -> None:
    forbidden = [
        arg
        for arg in argv
        if any(arg == flag or arg.startswith(flag + "=") for flag in _FORBIDDEN_PREFREEZE_FLAGS)
    ]
    if forbidden:
        raise SystemExit("unrecognized arguments: " + " ".join(forbidden))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    _reject_prefreeze_entropy_arguments(values)
    parser = argparse.ArgumentParser(
        description=(
            "Prepare EXP-277 confirmatory Gate A transactionally. This command is "
            "strictly pre-beacon and cannot accept challenge entropy or a beacon receipt."
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
    parser.add_argument("--checkpoint-seal-created-at-utc", required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt-output", type=Path, required=True)
    parser.add_argument("--prep-output", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    parser.add_argument("--reconstruction-output", type=Path, required=True)
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


def _exp277(protocol: dict[str, Any]) -> dict[str, Any]:
    for item in protocol.get("experiments") or []:
        if isinstance(item, dict) and item.get("experiment_id") == "EXP-277":
            return item
    raise RuntimeError("canonical Stage-A protocol does not contain EXP-277")


def _preflight_outputs(paths: tuple[Path, ...]) -> None:
    if len(set(paths)) != len(paths):
        raise SystemExit("output paths must be distinct")
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise SystemExit(f"output already exists: {existing}")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
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


def _publish_staged_files(staged: tuple[tuple[Path, Path], ...]) -> None:
    published: list[Path] = []
    try:
        for temp_path, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(temp_path, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise


def _require_valid(label: str, errors: list[str]) -> None:
    if errors:
        raise RuntimeError(f"invalid {label}: " + "; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = (
        args.checkpoint_output,
        args.checkpoint_receipt_output,
        args.prep_output,
        args.authorization_output,
        args.seal_output,
        args.reconstruction_output,
    )

    # The entire output surface is checked before training replay creates even a
    # temporary checkpoint. This prevents a late sentinel from causing a partial
    # Gate A publication.
    _preflight_outputs(outputs)

    protocol, protocol_digest = _verified_protocol(
        args.protocol,
        args.protocol_digest_file,
    )
    code_tree_digest = source_tree_digest(ROOT)
    execution = _load_json(args.execution, label="EXP-277 DEVELOPMENT execution")
    registry = _load_json(args.registry, label="EXP-277 arm registry")

    _require_valid(
        "EXP-277 DEVELOPMENT execution",
        validate_exp277_paired_development(execution),
    )
    if execution.get("code_digest") != code_tree_digest:
        raise RuntimeError(
            "EXP-277 DEVELOPMENT execution code digest does not match the current "
            "Gate A code tree; regenerate the 32+ replicate DEVELOPMENT pilot from "
            "the frozen Gate A source tree before sealing"
        )
    if execution.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-277 DEVELOPMENT execution protocol lineage mismatch")

    checkpoint_parent = args.checkpoint_output.parent
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    json_staged: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{args.checkpoint_output.name}.gate-a.",
        dir=checkpoint_parent,
    ) as checkpoint_stage_dir:
        checkpoint_stage = Path(checkpoint_stage_dir) / args.checkpoint_output.name
        checkpoint_receipt = build_exp277_trained_checkpoint(
            execution_artifact=execution,
            checkpoint_path=checkpoint_stage,
        )
        _require_valid(
            "EXP-277 trained checkpoint receipt",
            validate_exp277_checkpoint_receipt(checkpoint_receipt),
        )
        if file_sha256(checkpoint_stage) != checkpoint_receipt.get("checkpoint_file_sha256"):
            raise RuntimeError("EXP-277 staged checkpoint SHA256 does not match its receipt")

        prep = build_exp277_confirmatory_prep(
            experiment=_exp277(protocol),
            execution_artifact=execution,
            arm_registry=registry,
            analysis_code_digest=code_tree_digest,
        )
        _require_valid("EXP-277 confirmatory prep", validate_exp277_confirmatory_prep(prep))
        if prep.get("status") != "CONFIRMATORY_GATE_A_PREPARED":
            raise RuntimeError(
                f"EXP-277 Gate A is not ready to seal: {prep.get('status')!r}"
            )

        machinery_digests = {
            "challenge_generator_digest": code_tree_digest,
            "beacon_seed_derivation_digest": code_tree_digest,
            "executor_code_digest": code_tree_digest,
            "reconstruction_code_digest": code_tree_digest,
        }
        authorization = build_exp277_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=checkpoint_receipt,
            development_execution_artifact=execution,
            arm_registry=registry,
            source_tree_digest=code_tree_digest,
            freeze_commit_sha=args.freeze_commit_sha,
            freeze_commit_timestamp_utc=args.freeze_commit_timestamp_utc,
            checkpoint_seal_created_at_utc=args.checkpoint_seal_created_at_utc,
            machinery_digests=machinery_digests,
        )
        _require_valid(
            "EXP-277 Gate A authorization",
            validate_exp277_gate_a_authorization(authorization),
        )

        seal = build_exp277_gate_a_seal(authorization=authorization)
        _require_valid("EXP-277 Gate A seal", validate_exp277_gate_a_seal(seal))

        reconstruction = build_exp277_reconstruction_authorization(
            seal=seal,
            reconstruction_code_digest=code_tree_digest,
        )
        _require_valid(
            "EXP-277 reconstruction authorization",
            validate_exp277_reconstruction_authorization(reconstruction),
        )

        # Recheck immediately before publication. Another process creating any
        # target while Gate A was being assembled must fail closed without
        # overwriting or publishing the remaining artifacts.
        _preflight_outputs(outputs)
        try:
            staged_pairs: list[tuple[Path, Path]] = [
                (checkpoint_stage, args.checkpoint_output),
            ]
            for destination, payload in (
                (args.checkpoint_receipt_output, checkpoint_receipt),
                (args.prep_output, prep),
                (args.authorization_output, authorization),
                (args.seal_output, seal),
                (args.reconstruction_output, reconstruction),
            ):
                temp_path = _stage_json(destination, payload)
                json_staged.append(temp_path)
                staged_pairs.append((temp_path, destination))
            _publish_staged_files(tuple(staged_pairs))
        finally:
            for temp_path in json_staged:
                temp_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "status": seal["status"],
                "evidence_level": seal["evidence_level"],
                "decision": seal["decision"],
                "confirmatory_data_consumed": seal["confirmatory_data_consumed"],
                "challenge_materialized": seal["challenge_materialized"],
                "decision_rule_executed": seal["decision_rule_executed"],
                "protocol_digest": protocol_digest,
                "source_tree_digest": code_tree_digest,
                "checkpoint_file_sha256": checkpoint_receipt["checkpoint_file_sha256"],
                "checkpoint_receipt_digest": checkpoint_receipt["receipt_digest"],
                "prep_digest": prep["prep_digest"],
                "authorization_digest": authorization["authorization_digest"],
                "seal_digest": seal["seal_digest"],
                "reconstruction_digest": reconstruction["reconstruction_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
