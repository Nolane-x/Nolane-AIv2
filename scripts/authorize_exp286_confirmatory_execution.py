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

from nolane_ai.experiments.exp286_confirmatory_authorization import (
    authorize_exp286_confirmatory_execution,
    validate_exp286_confirmatory_execution_authorization,
)
from nolane_ai.protocol.identity import source_tree_digest


_FORBIDDEN_PREFREEZE_FLAGS = ("--seed", "--beacon", "--challenge-seed")


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
            "Authorize EXP-286 confirmatory execution at the pre-beacon boundary. "
            "This command never materializes challenge entropy or consumes confirmatory data."
        )
    )
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument(
        "--geometry-digest-file",
        type=Path,
        default=ROOT / "protocols" / "exp286_development_geometry_v1.sha256",
    )
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
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    execution = _load_json(args.execution, label="EXP-286 DEVELOPMENT execution")
    prep = _load_json(args.prep, label="EXP-286 Gate A prep")
    expected_geometry_digest = args.geometry_digest_file.read_text(encoding="utf-8").strip()
    code_digest = source_tree_digest(ROOT)

    authorization = authorize_exp286_confirmatory_execution(
        execution_artifact=execution,
        prep_artifact=prep,
        expected_geometry_digest=expected_geometry_digest,
        execution_code_digest=code_digest,
    )
    errors = validate_exp286_confirmatory_execution_authorization(authorization)
    if errors:
        raise RuntimeError("invalid EXP-286 confirmatory execution authorization: " + "; ".join(errors))
    _publish_json_exclusive(args.output, authorization)

    print(
        "EXP-286 execution court: "
        f"status={authorization['status']}; evidence={authorization['evidence_level']}; "
        f"decision={authorization['decision']}; confirmatory_n={authorization['confirmatory_n']}; "
        "challenge=NOT_EXECUTED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
