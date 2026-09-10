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

from nolane_ai.experiments.exp289_confirmatory_prep import (
    _prep_digest,
    build_exp289_confirmatory_prep,
    validate_exp289_confirmatory_prep,
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
            "Prepare EXP-289 confirmatory Gate A from DEVELOPMENT RDER evidence. "
            "This command is pre-beacon and never consumes challenge randomness."
        )
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _verified_protocol(protocol_path: Path, digest_path: Path) -> tuple[dict[str, Any], str]:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual!r}")
    require_canonical_stage_a_v1_digest(actual)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Stage-A protocol JSON must contain an object")
    return payload, actual


def _exp289(protocol: dict[str, Any]) -> dict[str, Any]:
    for item in protocol.get("experiments") or []:
        if isinstance(item, dict) and item.get("experiment_id") == "EXP-289":
            return item
    raise RuntimeError("canonical Stage-A protocol does not contain EXP-289")


def _familywise_alpha(protocol: dict[str, Any]) -> float:
    plan = protocol.get("global_sample_size_plan") or {}
    value = plan.get("familywise_alpha")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("canonical Stage-A protocol lacks numeric familywise_alpha")
    return float(value)


def _require_valid(label: str, errors: list[str]) -> None:
    if errors:
        raise RuntimeError(f"invalid {label}: " + "; ".join(errors))


def _unwrap_authoritative_execution(
    execution: dict[str, Any],
    *,
    protocol_digest: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate and remove only the external geometry authority envelope.

    EXP-289's deterministic reconstruction court owns the scientific execution
    payload. A predeclared geometry manifest is external authority, so it is
    validated against the canonical repository manifest and then stripped for
    deterministic reconstruction. Both the scientific and enveloped artifact
    digests are subsequently bound into Gate-A prep.
    """

    authority = execution.get("development_geometry_authority")
    if authority is None:
        return execution, None
    if not isinstance(authority, dict):
        raise RuntimeError("EXP-289 development geometry authority must be an object")

    from nolane_ai.experiments.exp289_development_geometry import (
        AUTHORITY_SCOPE,
        SCHEMA as GEOMETRY_SCHEMA,
        load_exp289_development_geometry,
    )
    from nolane_ai.experiments.exp289_paired_runner import _artifact_digest

    if authority.get("schema") != GEOMETRY_SCHEMA:
        raise RuntimeError("EXP-289 development geometry authority schema drift")
    if authority.get("authority_scope") != AUTHORITY_SCOPE:
        raise RuntimeError("EXP-289 development geometry authority scope drift")
    if authority.get("confirmatory_authority") is not False:
        raise RuntimeError("EXP-289 DEVELOPMENT geometry cannot claim confirmatory authority")

    full_digest = execution.get("artifact_digest")
    if not isinstance(full_digest, str) or full_digest != _artifact_digest(execution):
        raise RuntimeError("EXP-289 authoritative DEVELOPMENT artifact self-hash mismatch")

    manifest = ROOT / "protocols" / "exp289_authoritative_development_v1.json"
    digest_file = ROOT / "protocols" / "exp289_authoritative_development_v1.sha256"
    geometry, geometry_digest = load_exp289_development_geometry(
        manifest,
        digest_file,
        protocol_digest=protocol_digest,
    )
    if authority.get("manifest_digest") != geometry_digest:
        raise RuntimeError("EXP-289 DEVELOPMENT geometry manifest digest drift")

    scientific = deepcopy(execution)
    scientific.pop("development_geometry_authority", None)
    scientific["artifact_digest"] = _artifact_digest(scientific)
    if authority.get("scientific_execution_digest") != scientific["artifact_digest"]:
        raise RuntimeError("EXP-289 DEVELOPMENT geometry scientific execution digest drift")

    configuration = scientific.get("configuration") or {}
    for key, expected in geometry.items():
        if key == "tiny":
            continue
        if configuration.get(key) != expected:
            raise RuntimeError(f"EXP-289 authoritative DEVELOPMENT geometry execution drift: {key}")

    binding = deepcopy(authority)
    binding["authoritative_execution_digest"] = full_digest
    binding["geometry_configuration_match"] = True
    return scientific, binding


def _publish_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise SystemExit(f"output already exists: {path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Refuse a destination collision before importing torch-backed DEVELOPMENT
    # validators, reading evidence, or computing any Gate-A state.
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    from nolane_ai.experiments.exp289_paired_runner import validate_exp289_paired_development
    from nolane_ai.experiments.neural_arm_registry import validate_neural_arm_registry

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    code_digest = source_tree_digest(ROOT)
    raw_execution = _load_json(args.execution, label="EXP-289 DEVELOPMENT execution")
    registry = _load_json(args.registry, label="EXP-289 arm registry")
    execution, geometry_binding = _unwrap_authoritative_execution(
        raw_execution,
        protocol_digest=protocol_digest,
    )

    _require_valid("EXP-289 DEVELOPMENT execution", validate_exp289_paired_development(execution))
    _require_valid("EXP-289 arm registry", validate_neural_arm_registry(registry))
    if execution.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-289 DEVELOPMENT execution protocol lineage mismatch")
    if registry.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-289 arm registry protocol lineage mismatch")
    if execution.get("code_digest") != code_digest:
        raise RuntimeError(
            "EXP-289 DEVELOPMENT execution code digest does not match the current "
            "Gate A source tree; regenerate the 32+ replicate DEVELOPMENT pilot "
            "before preparing Gate A"
        )

    prep = build_exp289_confirmatory_prep(
        experiment=_exp289(protocol),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=code_digest,
        familywise_alpha=_familywise_alpha(protocol),
    )
    if geometry_binding is not None:
        prep["development_geometry_authority"] = geometry_binding
        prep["prep_digest"] = _prep_digest(prep)
    _require_valid("EXP-289 confirmatory prep", validate_exp289_confirmatory_prep(prep))
    _publish_json_exclusive(args.output, prep)

    freeze = prep["sample_size_freeze"]
    pilot = prep["pilot_summary"]
    print(
        "EXP-289 Gate A: "
        f"status={prep['status']}; evidence={prep['evidence_level']}; "
        f"decision={prep['decision']}; pilot_n={pilot['n']}; "
        f"paired_rder_sd={pilot['paired_relative_rder_reduction_sd']:.12g}; "
        f"unclamped_required_n={freeze['unclamped_required_n']}; "
        f"confirmatory_n={freeze['confirmatory_n']}; challenge=NOT_EXECUTED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
