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

from nolane_ai.experiments.exp286_confirmatory_prep import (
    build_exp286_confirmatory_prep,
    validate_exp286_confirmatory_prep,
)
from nolane_ai.experiments.exp286_paired_runner import validate_exp286_paired_development
from nolane_ai.experiments.neural_arm_registry import validate_neural_arm_registry
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
            "Prepare EXP-286 confirmatory Gate A from DEVELOPMENT pilot evidence. "
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


def _exp286(protocol: dict[str, Any]) -> dict[str, Any]:
    for item in protocol.get("experiments") or []:
        if isinstance(item, dict) and item.get("experiment_id") == "EXP-286":
            return item
    raise RuntimeError("canonical Stage-A protocol does not contain EXP-286")


def _familywise_alpha(protocol: dict[str, Any]) -> float:
    plan = protocol.get("global_sample_size_plan") or {}
    value = plan.get("familywise_alpha")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("canonical Stage-A protocol lacks numeric familywise_alpha")
    return float(value)


def _require_valid(label: str, errors: list[str]) -> None:
    if errors:
        raise RuntimeError(f"invalid {label}: " + "; ".join(errors))


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

    # Refuse an existing destination before reading DEVELOPMENT evidence or
    # computing any Gate-A planning state.
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    code_digest = source_tree_digest(ROOT)
    execution = _load_json(args.execution, label="EXP-286 DEVELOPMENT execution")
    registry = _load_json(args.registry, label="EXP-286 arm registry")

    _require_valid("EXP-286 DEVELOPMENT execution", validate_exp286_paired_development(execution))
    _require_valid("EXP-286 arm registry", validate_neural_arm_registry(registry))
    if execution.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-286 DEVELOPMENT execution protocol lineage mismatch")
    if registry.get("protocol_digest") != protocol_digest:
        raise RuntimeError("EXP-286 arm registry protocol lineage mismatch")
    if execution.get("code_digest") != code_digest:
        raise RuntimeError(
            "EXP-286 DEVELOPMENT execution code digest does not match the current "
            "Gate A source tree; regenerate the 32+ replicate DEVELOPMENT pilot "
            "before preparing Gate A"
        )

    prep = build_exp286_confirmatory_prep(
        experiment=_exp286(protocol),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=code_digest,
        familywise_alpha=_familywise_alpha(protocol),
    )
    _require_valid("EXP-286 confirmatory prep", validate_exp286_confirmatory_prep(prep))
    _publish_json_exclusive(args.output, prep)

    freeze = prep["sample_size_freeze"]
    print(
        "EXP-286 Gate A: "
        f"status={prep['status']}; evidence={prep['evidence_level']}; "
        f"decision={prep['decision']}; pilot_n={prep['pilot_summary']['n']}; "
        f"paired_log_cost_sd={prep['pilot_summary']['paired_log_cost_sd']:.12g}; "
        f"unclamped_required_n={freeze['unclamped_required_n']}; "
        f"confirmatory_n={freeze['confirmatory_n']}; challenge=NOT_EXECUTED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
