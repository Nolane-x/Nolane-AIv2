from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp286_development_geometry import (
    AUTHORITY_SCOPE as DEVELOPMENT_AUTHORITY_SCOPE,
    SCHEMA as DEVELOPMENT_GEOMETRY_SCHEMA,
    load_exp286_development_geometry,
)
from nolane_ai.experiments.exp286_paired_runner import (
    _artifact_digest,
    run_exp286_paired_development,
    validate_exp286_paired_development,
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
    "root_seed": "20260906-exp286-paired-dev",
    "d_model": 64,
    "hidden_size": 48,
    "target_parameters": 500_000,
    "train_replicates": 16,
    "eval_replicates": 16,
    "eval_start_replicate": 10_000,
    "batch_size": 8,
    "timesteps": 4,
    "variables": 8,
    "decoys": 3,
    "max_search_steps": 16,
    "noise_std": 0.05,
    "lr": 2e-3,
    "weight_decay": 0.0,
    "max_accounted_flops_per_episode": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched EV-E2 EXP-286 chronological/oracle-conflict-core DEVELOPMENT evaluation"
        )
    )
    parser.add_argument("--tiny", action="store_true", help="use CPU-safe tiny matched conflict arms")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--geometry-manifest", type=Path, default=None)
    parser.add_argument(
        "--geometry-digest-file",
        type=Path,
        default=ROOT / "protocols" / "exp286_development_geometry_v1.sha256",
    )
    parser.add_argument("--root-seed", default=_GEOMETRY_CLI_DEFAULTS["root_seed"])
    parser.add_argument("--d-model", type=int, default=_GEOMETRY_CLI_DEFAULTS["d_model"])
    parser.add_argument("--hidden-size", type=int, default=_GEOMETRY_CLI_DEFAULTS["hidden_size"])
    parser.add_argument("--target-parameters", type=int, default=_GEOMETRY_CLI_DEFAULTS["target_parameters"])
    parser.add_argument("--train-replicates", type=int, default=_GEOMETRY_CLI_DEFAULTS["train_replicates"])
    parser.add_argument("--eval-replicates", type=int, default=_GEOMETRY_CLI_DEFAULTS["eval_replicates"])
    parser.add_argument(
        "--eval-start-replicate", type=int, default=_GEOMETRY_CLI_DEFAULTS["eval_start_replicate"]
    )
    parser.add_argument("--batch-size", type=int, default=_GEOMETRY_CLI_DEFAULTS["batch_size"])
    parser.add_argument("--timesteps", type=int, default=_GEOMETRY_CLI_DEFAULTS["timesteps"])
    parser.add_argument("--variables", type=int, default=_GEOMETRY_CLI_DEFAULTS["variables"])
    parser.add_argument("--decoys", type=int, default=_GEOMETRY_CLI_DEFAULTS["decoys"])
    parser.add_argument(
        "--max-search-steps", type=int, default=_GEOMETRY_CLI_DEFAULTS["max_search_steps"]
    )
    parser.add_argument("--noise-std", type=float, default=_GEOMETRY_CLI_DEFAULTS["noise_std"])
    parser.add_argument("--lr", type=float, default=_GEOMETRY_CLI_DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=_GEOMETRY_CLI_DEFAULTS["weight_decay"])
    parser.add_argument(
        "--max-accounted-flops-per-episode",
        type=int,
        default=_GEOMETRY_CLI_DEFAULTS["max_accounted_flops_per_episode"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    return parser.parse_args()


def _verified_protocol(protocol_path: Path, digest_path: Path) -> tuple[dict, str]:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_canonical_stage_a_v1_digest(actual)
    return json.loads(protocol_path.read_text(encoding="utf-8")), actual


def _resolve_geometry(args: argparse.Namespace, protocol_digest: str) -> tuple[dict, str | None]:
    if args.geometry_manifest is None:
        if args.tiny:
            return {
                "root_seed": args.root_seed,
                "d_model": 8,
                "hidden_size": 6,
                "target_parameters": 5_000,
                "train_replicates": args.train_replicates,
                "eval_replicates": args.eval_replicates,
                "eval_start_replicate": args.eval_start_replicate,
                "batch_size": args.batch_size,
                "timesteps": args.timesteps,
                "variables": args.variables,
                "decoys": args.decoys,
                "max_search_steps": args.max_search_steps,
                "noise_std": args.noise_std,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_accounted_flops_per_episode": args.max_accounted_flops_per_episode,
            }, None
        return {key: getattr(args, key) for key in _GEOMETRY_CLI_DEFAULTS}, None

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
    geometry, digest = load_exp286_development_geometry(
        args.geometry_manifest,
        args.geometry_digest_file,
        protocol_digest=protocol_digest,
    )
    return geometry, digest


def main() -> int:
    args = parse_args()
    outputs = (args.output, args.registry_output)
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None:
        raise SystemExit(f"output already exists: {existing}")

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    geometry, geometry_digest = _resolve_geometry(args, protocol_digest)
    code_digest = source_tree_digest(ROOT)

    execution = run_exp286_paired_development(
        root_seed=str(geometry["root_seed"]),
        d_model=int(geometry["d_model"]),
        hidden_size=int(geometry["hidden_size"]),
        target_parameters=int(geometry["target_parameters"]),
        train_replicates=int(geometry["train_replicates"]),
        eval_replicates=int(geometry["eval_replicates"]),
        eval_start_replicate=int(geometry["eval_start_replicate"]),
        batch_size=int(geometry["batch_size"]),
        timesteps=int(geometry["timesteps"]),
        variables=int(geometry["variables"]),
        decoys=int(geometry["decoys"]),
        max_search_steps=int(geometry["max_search_steps"]),
        noise_std=float(geometry["noise_std"]),
        lr=float(geometry["lr"]),
        weight_decay=float(geometry["weight_decay"]),
        protocol_digest=protocol_digest,
        code_digest=code_digest,
        max_accounted_flops_per_episode=geometry["max_accounted_flops_per_episode"],
    )
    if geometry_digest is not None:
        execution["development_geometry_authority"] = {
            "schema": DEVELOPMENT_GEOMETRY_SCHEMA,
            "authority_scope": DEVELOPMENT_AUTHORITY_SCOPE,
            "manifest_digest": geometry_digest,
            "confirmatory_authority": False,
        }
        execution["artifact_digest"] = _artifact_digest(execution)

    execution_errors = validate_exp286_paired_development(execution)
    if execution_errors:
        raise RuntimeError(
            "invalid EXP-286 paired development artifact before write: "
            + "; ".join(execution_errors)
        )

    pair_audit = execution["resource_match"]["pair_audit"]
    pilot_audit = audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))
    registry = build_neural_arm_registry(
        protocol=protocol,
        protocol_digest=protocol_digest,
        model_audit=pilot_audit,
        exp286_pair_audit=pair_audit,
        exp286_execution_artifact=execution,
    )
    registry_errors = validate_neural_arm_registry(registry)
    if registry_errors:
        raise RuntimeError(
            "invalid EXP-286 neural arm registry before write: " + "; ".join(registry_errors)
        )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(execution, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.registry_output.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    exp286 = registry["experiments"]["EXP-286"]
    aggregate = execution["evaluation"]["aggregate"]
    print(
        json.dumps(
            {
                "schema": execution["schema"],
                "evidence_level": execution["evidence_level"],
                "decision": execution["decision"],
                "execution_digest": execution["artifact_digest"],
                "development_geometry_digest": geometry_digest,
                "registry_digest": registry["registry_digest"],
                "match_court": exp286["match_court"],
                "development_match_status": exp286["development_match_status"],
                "descriptive_relative_flop_reduction": aggregate[
                    "descriptive_relative_flop_reduction"
                ],
                "confirmatory_data_consumed": execution["confirmatory_data_consumed"],
                "challenge_materialized": execution["challenge_materialized"],
                "learned_conflict_localizer_validated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())