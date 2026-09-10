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

from nolane_ai.experiments.exp286_confirmatory_ceremony import (
    seal_exp286_confirmatory_gate_a,
    validate_exp286_confirmatory_gate_a_seal,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seal EXP-286 confirmatory Gate A after protocol, code, DEVELOPMENT geometry, "
            "prep, and execution authorization are fixed. This command remains pre-challenge."
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
        "--geometry-digest-file",
        type=Path,
        default=ROOT / "protocols" / "exp286_development_geometry_v1.sha256",
    )
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--freeze-commit-sha", required=True)
    parser.add_argument("--freeze-commit-timestamp-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


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


def _verified_protocol_digest(protocol: Path, digest_file: Path) -> str:
    actual = file_sha256(protocol)
    expected = digest_file.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(
            f"protocol digest mismatch: expected {expected!r}, got {actual!r}"
        )
    require_canonical_stage_a_v1_digest(actual)
    return actual


def _publish_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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

    protocol_digest = _verified_protocol_digest(
        args.protocol, args.protocol_digest_file
    )
    geometry_digest = args.geometry_digest_file.read_text(encoding="utf-8").strip()
    execution = _load_json(args.execution, label="EXP-286 DEVELOPMENT execution")
    prep = _load_json(args.prep, label="EXP-286 Gate A prep")
    authorization = _load_json(
        args.authorization, label="EXP-286 execution authorization"
    )
    code_tree_digest = source_tree_digest(ROOT)

    seal = seal_exp286_confirmatory_gate_a(
        protocol_digest=protocol_digest,
        development_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        expected_geometry_digest=geometry_digest,
        code_tree_digest=code_tree_digest,
        freeze_commit_sha=args.freeze_commit_sha,
        freeze_commit_timestamp_utc=args.freeze_commit_timestamp_utc,
    )
    errors = validate_exp286_confirmatory_gate_a_seal(seal)
    if errors:
        raise RuntimeError("invalid EXP-286 Gate-A seal: " + "; ".join(errors))

    _publish_json_exclusive(args.output, seal)
    print(
        "EXP-286 Gate-A seal: "
        f"status={seal['status']}; evidence={seal['evidence_level']}; "
        f"decision={seal['decision']}; confirmatory_n={seal['confirmatory_n']}; "
        "challenge=NOT_EXECUTED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
