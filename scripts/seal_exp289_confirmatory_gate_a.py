from __future__ import annotations

import argparse
from copy import deepcopy
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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_exp289_confirmatory_gate_a import (
    _exp289,
    _familywise_alpha,
    _unwrap_authoritative_execution,
    _verified_protocol,
)
from nolane_ai.experiments.exp289_checkpoint import (
    build_exp289_trained_checkpoint,
    validate_exp289_checkpoint_receipt,
)
from nolane_ai.experiments.exp289_confirmatory_authorization import (
    authorize_exp289_confirmatory_execution,
    validate_exp289_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp289_confirmatory_ceremony import (
    seal_exp289_confirmatory_gate_a,
    validate_exp289_confirmatory_gate_a_seal,
)
from nolane_ai.experiments.exp289_confirmatory_prep import (
    _prep_digest,
    build_exp289_confirmatory_prep,
    validate_exp289_confirmatory_prep,
)
from nolane_ai.experiments.exp289_development_geometry import (
    load_exp289_development_geometry,
)
from nolane_ai.experiments.exp289_paired_runner import validate_exp289_paired_development
from nolane_ai.experiments.neural_arm_registry import validate_neural_arm_registry
from nolane_ai.protocol.identity import source_tree_digest


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
            "Build the exact EXP-289 trained checkpoint, authorize execution, and "
            "atomically publish the immutable pre-beacon Gate-A seal. This command "
            "cannot accept beacon entropy or challenge seeds."
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
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        default=ROOT / "protocols" / "exp289_authoritative_development_v1.json",
    )
    parser.add_argument(
        "--geometry-digest-file",
        type=Path,
        default=ROOT / "protocols" / "exp289_authoritative_development_v1.sha256",
    )
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze-commit-sha", required=True)
    parser.add_argument("--freeze-commit-timestamp-utc", required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt-output", type=Path, required=True)
    parser.add_argument("--prep-output", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
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


def _require_valid(label: str, errors: list[str]) -> None:
    if errors:
        raise RuntimeError(f"invalid {label}: " + "; ".join(errors))


def _geometry_matches_execution(
    execution_configuration: Any,
    manifest_geometry: Any,
) -> bool:
    """Compare only scientific runner fields; ``tiny`` is manifest authority metadata.

    The authoritative runner deliberately does not serialize the CLI-only ``tiny``
    switch into its scientific execution configuration. Every other predeclared
    geometry field must match exactly and no extra scientific execution field is
    accepted here.
    """

    if not isinstance(execution_configuration, dict) or not isinstance(manifest_geometry, dict):
        return False
    normalized_manifest = {
        key: value for key, value in manifest_geometry.items() if key != "tiny"
    }
    return execution_configuration == normalized_manifest


def _preflight_outputs(paths: tuple[Path, ...]) -> None:
    if len(set(paths)) != len(paths):
        raise SystemExit("output paths must be distinct")
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise SystemExit(f"output already exists: {existing}")


def _stage_bytes(destination: Path, data: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _checkpoint_stage_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".checkpoint.tmp",
        dir=destination.parent,
    )
    os.close(fd)
    path = Path(temp_name)
    path.unlink()
    return path


def _publish_transaction(staged: tuple[tuple[Path, Path], ...]) -> None:
    published: list[Path] = []
    try:
        for source, destination in staged:
            os.link(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for source, _ in staged:
            source.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = (
        args.checkpoint_output,
        args.checkpoint_receipt_output,
        args.prep_output,
        args.authorization_output,
        args.seal_output,
    )
    _preflight_outputs(outputs)

    protocol, protocol_digest = _verified_protocol(
        args.protocol,
        args.protocol_digest_file,
    )
    geometry, geometry_digest = load_exp289_development_geometry(
        args.geometry_manifest,
        args.geometry_digest_file,
        protocol_digest=protocol_digest,
    )
    code_digest = source_tree_digest(ROOT)

    authoritative_execution = _load_json(
        args.execution,
        label="EXP-289 authoritative DEVELOPMENT execution",
    )
    registry = _load_json(args.registry, label="EXP-289 arm registry")
    scientific_execution, geometry_binding = _unwrap_authoritative_execution(
        authoritative_execution,
        protocol_digest=protocol_digest,
    )
    if geometry_binding is None:
        raise RuntimeError("EXP-289 Gate-A seal requires authoritative DEVELOPMENT geometry")

    _require_valid(
        "EXP-289 DEVELOPMENT execution",
        validate_exp289_paired_development(scientific_execution),
    )
    _require_valid(
        "EXP-289 arm registry",
        validate_neural_arm_registry(registry),
    )
    if scientific_execution.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-289 Gate-A seal protocol/development lineage mismatch")
    if scientific_execution.get("code_digest") != code_digest:
        raise RuntimeError(
            "EXP-289 DEVELOPMENT execution code digest does not match the current "
            "pre-beacon source tree; regenerate authoritative DEVELOPMENT evidence"
        )
    if not _geometry_matches_execution(scientific_execution.get("configuration"), geometry):
        raise RuntimeError("EXP-289 authoritative DEVELOPMENT geometry configuration drift")
    if geometry_binding.get("manifest_digest") != geometry_digest:
        raise RuntimeError("EXP-289 authoritative DEVELOPMENT geometry digest drift")

    prep = build_exp289_confirmatory_prep(
        experiment=_exp289(protocol),
        execution_artifact=scientific_execution,
        arm_registry=registry,
        analysis_code_digest=code_digest,
        familywise_alpha=_familywise_alpha(protocol),
    )
    prep["development_geometry_authority"] = deepcopy(geometry_binding)
    prep["prep_digest"] = _prep_digest(prep)
    _require_valid("EXP-289 confirmatory prep", validate_exp289_confirmatory_prep(prep))
    if prep.get("status") != "CONFIRMATORY_GATE_A_PREPARED" or prep.get("confirmatory_ready") is not True:
        raise RuntimeError(
            "EXP-289 immutable seal requires Gate A READY; refusing to freeze a NOT_READY pilot"
        )

    checkpoint_stage = _checkpoint_stage_path(args.checkpoint_output)
    json_stages: list[Path] = []
    try:
        checkpoint_receipt = build_exp289_trained_checkpoint(
            execution_artifact=authoritative_execution,
            checkpoint_path=checkpoint_stage,
        )
        _require_valid(
            "EXP-289 checkpoint receipt",
            validate_exp289_checkpoint_receipt(checkpoint_receipt),
        )

        authorization = authorize_exp289_confirmatory_execution(
            execution_artifact=authoritative_execution,
            prep_artifact=prep,
            checkpoint_path=checkpoint_stage,
            checkpoint_receipt=checkpoint_receipt,
            expected_geometry_digest=geometry_digest,
            expected_geometry_configuration=geometry,
            execution_code_digest=code_digest,
        )
        _require_valid(
            "EXP-289 execution authorization",
            validate_exp289_confirmatory_execution_authorization(authorization),
        )

        seal = seal_exp289_confirmatory_gate_a(
            protocol_digest=protocol_digest,
            development_execution_artifact=authoritative_execution,
            prep_artifact=prep,
            checkpoint_receipt=checkpoint_receipt,
            execution_authorization=authorization,
            expected_geometry_digest=geometry_digest,
            code_tree_digest=code_digest,
            freeze_commit_sha=args.freeze_commit_sha,
            freeze_commit_timestamp_utc=args.freeze_commit_timestamp_utc,
        )
        _require_valid(
            "EXP-289 Gate-A seal",
            validate_exp289_confirmatory_gate_a_seal(seal),
        )

        _preflight_outputs(outputs)
        for destination, payload in (
            (args.checkpoint_receipt_output, checkpoint_receipt),
            (args.prep_output, prep),
            (args.authorization_output, authorization),
            (args.seal_output, seal),
        ):
            json_stages.append(_stage_bytes(destination, _json_bytes(payload)))

        staged = (
            (checkpoint_stage, args.checkpoint_output),
            (json_stages[0], args.checkpoint_receipt_output),
            (json_stages[1], args.prep_output),
            (json_stages[2], args.authorization_output),
            (json_stages[3], args.seal_output),
        )
        _publish_transaction(staged)
    except BaseException:
        checkpoint_stage.unlink(missing_ok=True)
        for path in json_stages:
            path.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "schema": seal["schema"],
                "status": seal["status"],
                "evidence_level": seal["evidence_level"],
                "decision": seal["decision"],
                "confirmatory_n": seal["confirmatory_n"],
                "confirmatory_data_consumed": seal["confirmatory_data_consumed"],
                "seed_materialization_status": seal["seed_materialization_status"],
                "challenge_materialized": seal["challenge_materialized"],
                "code_tree_digest": code_digest,
                "checkpoint_receipt_digest": checkpoint_receipt["receipt_digest"],
                "checkpoint_scientific_identity_digest": checkpoint_receipt["scientific_identity_digest"],
                "authorization_digest": authorization["authorization_digest"],
                "pre_beacon_binding_digest": seal["pre_beacon_binding_digest"],
                "seal_digest": seal["seal_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
