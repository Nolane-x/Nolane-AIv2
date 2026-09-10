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

from nolane_ai.experiments.exp289_development_geometry import (
    AUTHORITY_SCOPE as DEVELOPMENT_AUTHORITY_SCOPE,
    SCHEMA as DEVELOPMENT_GEOMETRY_SCHEMA,
    load_exp289_development_geometry,
)
from nolane_ai.experiments.exp289_paired_runner import (
    _artifact_digest,
    run_exp289_paired_development,
    validate_exp289_execution_artifact,
    validate_exp289_paired_development,
)
from nolane_ai.experiments.neural_arm_registry import (
    build_neural_arm_registry,
    validate_neural_arm_registry,
)
from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


_GEOMETRY_CLI_DEFAULTS = {
    "root_seed": "20260906-exp289-paired-dev",
    "d_model": 64,
    "hidden_size": 48,
    "target_parameters": 500_000,
    "train_replicates": 16,
    "eval_replicates": 16,
    "eval_start_replicate": 10_000,
    "batch_size": 8,
    "timesteps": 4,
    "restarts": 4,
    "variables": 8,
    "decoys": 3,
    "max_search_steps": 24,
    "noise_std": 0.05,
    "lr": 2e-3,
    "weight_decay": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched EV-E2 EXP-289 no-nogood/episode-local-nogood DEVELOPMENT evaluation"
        )
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="use CPU-safe tiny matched episode-local nogood arms",
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
    parser.add_argument("--geometry-manifest", type=Path, default=None)
    parser.add_argument(
        "--geometry-digest-file",
        type=Path,
        default=ROOT / "protocols" / "exp289_authoritative_development_v1.sha256",
    )
    parser.add_argument("--root-seed", default=_GEOMETRY_CLI_DEFAULTS["root_seed"])
    parser.add_argument("--d-model", type=int, default=_GEOMETRY_CLI_DEFAULTS["d_model"])
    parser.add_argument("--hidden-size", type=int, default=_GEOMETRY_CLI_DEFAULTS["hidden_size"])
    parser.add_argument("--target-parameters", type=int, default=_GEOMETRY_CLI_DEFAULTS["target_parameters"])
    parser.add_argument("--train-replicates", type=int, default=_GEOMETRY_CLI_DEFAULTS["train_replicates"])
    parser.add_argument("--eval-replicates", type=int, default=_GEOMETRY_CLI_DEFAULTS["eval_replicates"])
    parser.add_argument("--eval-start-replicate", type=int, default=_GEOMETRY_CLI_DEFAULTS["eval_start_replicate"])
    parser.add_argument("--batch-size", type=int, default=_GEOMETRY_CLI_DEFAULTS["batch_size"])
    parser.add_argument("--timesteps", type=int, default=_GEOMETRY_CLI_DEFAULTS["timesteps"])
    parser.add_argument("--restarts", type=int, default=_GEOMETRY_CLI_DEFAULTS["restarts"])
    parser.add_argument("--variables", type=int, default=_GEOMETRY_CLI_DEFAULTS["variables"])
    parser.add_argument("--decoys", type=int, default=_GEOMETRY_CLI_DEFAULTS["decoys"])
    parser.add_argument("--max-search-steps", type=int, default=_GEOMETRY_CLI_DEFAULTS["max_search_steps"])
    parser.add_argument("--noise-std", type=float, default=_GEOMETRY_CLI_DEFAULTS["noise_std"])
    parser.add_argument("--lr", type=float, default=_GEOMETRY_CLI_DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=_GEOMETRY_CLI_DEFAULTS["weight_decay"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, default=None)
    return parser.parse_args()


def _verified_protocol(protocol_path: Path, digest_path: Path) -> tuple[dict[str, Any], str]:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_canonical_stage_a_v1_digest(actual)
    return json.loads(protocol_path.read_text(encoding="utf-8")), actual


def _resolve_geometry(args: argparse.Namespace, protocol_digest: str) -> tuple[dict[str, Any], str | None]:
    if args.geometry_manifest is None:
        geometry = {key: getattr(args, key) for key in _GEOMETRY_CLI_DEFAULTS}
        if args.tiny:
            geometry["d_model"] = 8
            geometry["hidden_size"] = 6
            geometry["target_parameters"] = 5_000
        return geometry, None

    if args.tiny:
        raise SystemExit("geometry manifest cannot be combined with --tiny")
    overridden = [
        name
        for name, default in _GEOMETRY_CLI_DEFAULTS.items()
        if getattr(args, name) != default
    ]
    if overridden:
        raise SystemExit(
            "geometry manifest cannot be combined with geometry overrides: " + ", ".join(overridden)
        )
    geometry, digest = load_exp289_development_geometry(
        args.geometry_manifest,
        args.geometry_digest_file,
        protocol_digest=protocol_digest,
    )
    return geometry, digest


def _preflight_outputs(paths: tuple[Path, ...]) -> None:
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


def main() -> int:
    args = parse_args()
    outputs = (args.output,) if args.registry_output is None else (args.output, args.registry_output)
    _preflight_outputs(outputs)

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    geometry, geometry_digest = _resolve_geometry(args, protocol_digest)
    code_digest = source_tree_digest(ROOT)

    execution = run_exp289_paired_development(
        root_seed=str(geometry["root_seed"]),
        d_model=int(geometry["d_model"]),
        hidden_size=int(geometry["hidden_size"]),
        target_parameters=int(geometry["target_parameters"]),
        train_replicates=int(geometry["train_replicates"]),
        eval_replicates=int(geometry["eval_replicates"]),
        eval_start_replicate=int(geometry["eval_start_replicate"]),
        batch_size=int(geometry["batch_size"]),
        timesteps=int(geometry["timesteps"]),
        restarts=int(geometry["restarts"]),
        variables=int(geometry["variables"]),
        decoys=int(geometry["decoys"]),
        max_search_steps=int(geometry["max_search_steps"]),
        noise_std=float(geometry["noise_std"]),
        lr=float(geometry["lr"]),
        weight_decay=float(geometry["weight_decay"]),
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )
    if geometry_digest is not None:
        execution["development_geometry_authority"] = {
            "schema": DEVELOPMENT_GEOMETRY_SCHEMA,
            "authority_scope": DEVELOPMENT_AUTHORITY_SCOPE,
            "manifest_digest": geometry_digest,
            "confirmatory_authority": False,
        }
        execution["artifact_digest"] = _artifact_digest(execution)

    execution_errors = validate_exp289_paired_development(execution)
    if execution_errors:
        raise RuntimeError(
            "invalid EXP-289 paired development artifact before write: "
            + "; ".join(execution_errors)
        )
    validate_exp289_execution_artifact(execution, protocol)

    registry: dict[str, Any] | None = None
    if args.registry_output is not None:
        pair_audit = execution["resource_match"]["pair_audit"]
        pilot_audit = audit_model(
            NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta")
        )
        registry = build_neural_arm_registry(
            protocol=protocol,
            protocol_digest=protocol_digest,
            model_audit=pilot_audit,
            exp289_pair_audit=pair_audit,
            exp289_execution_artifact=execution,
        )
        registry_errors = validate_neural_arm_registry(registry)
        if registry_errors:
            raise RuntimeError(
                "invalid EXP-289 neural arm registry before write: "
                + "; ".join(registry_errors)
            )

    payloads: list[tuple[Path, dict[str, Any]]] = [(args.output, execution)]
    if args.registry_output is not None and registry is not None:
        payloads.append((args.registry_output, registry))

    _preflight_outputs(outputs)
    _commit_outputs(tuple(payloads))

    aggregate = execution["evaluation"]["aggregate"]
    exp289 = registry["experiments"]["EXP-289"] if registry is not None else None
    paired = exp289["paired_execution_evidence"] if exp289 is not None else None
    print(
        json.dumps(
            {
                "schema": execution["schema"],
                "evidence_level": execution["evidence_level"],
                "decision": execution["decision"],
                "execution_digest": execution["artifact_digest"],
                "development_geometry_digest": geometry_digest,
                "registry_digest": registry["registry_digest"] if registry is not None else None,
                "match_court": exp289["match_court"] if exp289 is not None else None,
                "development_match_status": (
                    exp289["development_match_status"] if exp289 is not None else None
                ),
                "mean_descriptive_relative_rder_reduction": aggregate[
                    "mean_descriptive_relative_rder_reduction"
                ],
                "confirmatory_data_consumed": execution["confirmatory_data_consumed"],
                "challenge_materialized": execution["challenge_materialized"],
                "learned_clause_transfer_validated": (
                    paired["learned_clause_transfer_validated"]
                    if paired is not None
                    else False
                ),
                "lifelong_lemma_economy_validated": (
                    paired["lifelong_lemma_economy_validated"]
                    if paired is not None
                    else False
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
