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

from nolane_ai.experiments.exp289_paired_runner import (
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
    parser.add_argument("--root-seed", default="20260906-exp289-paired-dev")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--target-parameters", type=int, default=500_000)
    parser.add_argument("--train-replicates", type=int, default=16)
    parser.add_argument("--eval-replicates", type=int, default=16)
    parser.add_argument("--eval-start-replicate", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--variables", type=int, default=8)
    parser.add_argument("--decoys", type=int, default=3)
    parser.add_argument("--max-search-steps", type=int, default=24)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    """Atomically publish validated JSON files without overwriting existing paths.

    Each file is fully staged and fsynced in its destination directory before an
    atomic hard-link publish. If a later publish fails, files published by this
    invocation are rolled back, leaving no partial output set.
    """

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

    # Fail before protocol/model work and never overwrite an existing artifact.
    _preflight_outputs(outputs)

    protocol, protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    code_digest = source_tree_digest(ROOT)

    if args.tiny:
        d_model = 8
        hidden_size = 6
        target_parameters = 5_000
    else:
        d_model = args.d_model
        hidden_size = args.hidden_size
        target_parameters = args.target_parameters

    execution = run_exp289_paired_development(
        root_seed=args.root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        train_replicates=args.train_replicates,
        eval_replicates=args.eval_replicates,
        eval_start_replicate=args.eval_start_replicate,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        restarts=args.restarts,
        variables=args.variables,
        decoys=args.decoys,
        max_search_steps=args.max_search_steps,
        noise_std=args.noise_std,
        lr=args.lr,
        weight_decay=args.weight_decay,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )
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

    # Recheck immediately before publish to close the normal no-overwrite race.
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
