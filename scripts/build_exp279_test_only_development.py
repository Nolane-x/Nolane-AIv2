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

from nolane_ai.experiments.exp279_paired_runner import (
    ARM_ORDER,
    PRIMARY_METRIC,
    _aggregate_rows,
    _artifact_digest,
    run_exp279_paired_development,
    validate_exp279_paired_development,
)
from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


FIXTURE_PURPOSE = "confirmatory_machinery_ci_only"
ROOT_SEED = "TEST-ONLY-exp279-development-ci-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an explicitly non-scientific EXP-279 EV-E2 DEVELOPMENT fixture "
            "for confirmatory machinery CI. This command never materializes a "
            "confirmatory beacon or challenge."
        )
    )
    parser.add_argument(
        "--test-only-fixture",
        action="store_true",
        required=True,
        help="mandatory acknowledgement that the output is CI-only and non-scientific",
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def _verified_protocol(protocol_path: Path, digest_path: Path) -> tuple[dict[str, Any], str]:
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


def _require_valid(errors: list[str]) -> None:
    if errors:
        raise RuntimeError("invalid EXP-279 TEST-ONLY DEVELOPMENT fixture: " + "; ".join(errors))


def _preflight(outputs: tuple[Path, ...]) -> None:
    if len(set(outputs)) != len(outputs):
        raise SystemExit("output paths must be distinct")
    existing = next((path for path in outputs if path.exists()), None)
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


def _publish_pair(
    first: tuple[Path, dict[str, Any]],
    second: tuple[Path, dict[str, Any]],
) -> None:
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, payload in (first, second):
            staged.append((_stage_json(destination, payload), destination))
        _preflight((first[0], second[0]))
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


def _apply_deterministic_ci_metric_profile(execution: dict[str, Any]) -> None:
    rows = (execution.get("evaluation") or {}).get("per_replicate") or []
    if not isinstance(rows, list) or len(rows) != 33:
        raise RuntimeError("EXP-279 TEST-ONLY fixture requires exactly 33 DEVELOPMENT rows")

    max_flops = max(
        float(row[arm]["accounted_flops_per_episode"])
        for row in rows
        for arm in ARM_ORDER
    )
    if max_flops <= 0.0:
        raise RuntimeError("EXP-279 TEST-ONLY fixture compute ledger is non-positive")

    # This profile is deliberately synthetic and low-variance so Gate A freezes
    # the protocol minimum n=32 deterministically. It is CI machinery input,
    # never empirical evidence about the model.
    base_utility = 0.10 / max_flops
    stratum_scale = {
        "PROPAGATION_FIT": 1.00,
        "BRANCH_FIT": 1.03,
        "MIXED_RESIDUAL": 0.97,
    }
    arm_scale = {
        "propagation_only": 1.00,
        "branch_only": 1.01,
        "hybrid": 1.12,
    }
    for row in rows:
        stratum = row.get("stratum")
        if stratum not in stratum_scale:
            raise RuntimeError(f"unexpected EXP-279 TEST-ONLY stratum: {stratum!r}")
        for arm in ARM_ORDER:
            flops = float(row[arm]["accounted_flops_per_episode"])
            utility = base_utility * stratum_scale[stratum] * arm_scale[arm]
            row[arm]["verified_solution_rate"] = utility * flops
            row[arm][PRIMARY_METRIC] = utility

    execution["evaluation"]["aggregate"] = _aggregate_rows(rows)
    execution["test_only_fixture"] = True
    execution["scientific_evidence_eligible"] = False
    execution["fixture_purpose"] = FIXTURE_PURPOSE
    execution["fixture_metric_profile"] = "deterministic_low_variance_ci_v1"
    execution["artifact_digest"] = _artifact_digest(execution)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.test_only_fixture is not True:
        raise SystemExit("--test-only-fixture is required")
    outputs = (args.output, args.registry_output)
    _preflight(outputs)

    protocol, protocol_digest = _verified_protocol(
        args.protocol,
        args.protocol_digest_file,
    )
    code_digest = source_tree_digest(ROOT)

    execution = run_exp279_paired_development(
        root_seed=ROOT_SEED,
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=33,
        eval_start_replicate=20_000,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )
    _apply_deterministic_ci_metric_profile(execution)
    _require_valid(validate_exp279_paired_development(execution))

    pilot_audit = audit_model(
        NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta")
    )
    registry = build_neural_arm_registry(
        protocol=protocol,
        protocol_digest=protocol_digest,
        model_audit=pilot_audit,
        exp279_pair_audit=execution["resource_match"]["pair_audit"],
        exp279_execution_artifact=execution,
    )

    _preflight(outputs)
    _publish_pair((args.output, execution), (args.registry_output, registry))
    print(
        json.dumps(
            {
                "status": "TEST_ONLY_DEVELOPMENT_FIXTURE_BUILT",
                "evidence_level": execution["evidence_level"],
                "decision": execution["decision"],
                "test_only_fixture": execution["test_only_fixture"],
                "scientific_evidence_eligible": execution[
                    "scientific_evidence_eligible"
                ],
                "fixture_purpose": execution["fixture_purpose"],
                "protocol_digest": protocol_digest,
                "source_tree_digest": code_digest,
                "development_artifact_digest": execution["artifact_digest"],
                "registry_digest": registry["registry_digest"],
                "confirmatory_data_consumed": execution[
                    "confirmatory_data_consumed"
                ],
                "challenge_materialized": execution["challenge_materialized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
