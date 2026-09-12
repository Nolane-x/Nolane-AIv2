from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_multiroot_support_stability_v7 import (
    FROZEN_ARM_GEOMETRY,
    FROZEN_OPTIMIZER,
    FROZEN_ROUTE_THRESHOLD,
    FROZEN_WORLD_GEOMETRY,
    PROBE_REPLICATES,
    run_exp279_multiroot_support_stability_v7_shard,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one DEVELOPMENT-only EXP-279 V7 multi-root support shard"
    )
    parser.add_argument("--tiny-contract", action="store_true")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--train-replicates", type=int, choices=(60, 120), required=True)
    parser.add_argument("--canonical-index", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--target-parameters", type=int, default=500_000)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--probe-replicates", type=int, default=198)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _verified_protocol(protocol_path: Path, digest_path: Path) -> str:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_canonical_stage_a_v1_digest(actual)
    return actual


def _require_real_scientific_configuration(args: argparse.Namespace) -> None:
    supplied = {
        "d_model": args.d_model,
        "hidden_size": args.hidden_size,
        "target_parameters": args.target_parameters,
        "route_threshold": args.route_threshold,
        "probe_replicates": args.probe_replicates,
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "variables": args.variables,
        "constraints": args.constraints,
        "noise_std": args.noise_std,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    expected = {
        "d_model": FROZEN_WORLD_GEOMETRY["d_model"],
        "hidden_size": FROZEN_ARM_GEOMETRY["hidden_size"],
        "target_parameters": FROZEN_ARM_GEOMETRY["target_parameters"],
        "route_threshold": FROZEN_ROUTE_THRESHOLD,
        "probe_replicates": PROBE_REPLICATES,
        "batch_size": FROZEN_WORLD_GEOMETRY["batch_size"],
        "timesteps": FROZEN_WORLD_GEOMETRY["timesteps"],
        "variables": FROZEN_WORLD_GEOMETRY["variables"],
        "constraints": FROZEN_WORLD_GEOMETRY["constraints"],
        "noise_std": FROZEN_WORLD_GEOMETRY["noise_std"],
        "lr": FROZEN_OPTIMIZER["lr"],
        "weight_decay": FROZEN_OPTIMIZER["weight_decay"],
    }
    if supplied != expected:
        raise SystemExit("non-contract V7 execution must use the frozen scientific configuration")


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if not args.tiny_contract:
        _require_real_scientific_configuration(args)

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    receipt = run_exp279_multiroot_support_stability_v7_shard(
        train_replicates=args.train_replicates,
        canonical_index=args.canonical_index,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        target_parameters=args.target_parameters,
        route_threshold=args.route_threshold,
        probe_replicates=args.probe_replicates,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        variables=args.variables,
        constraints=args.constraints,
        noise_std=args.noise_std,
        lr=args.lr,
        weight_decay=args.weight_decay,
        protocol_digest=protocol_digest,
        code_digest=source_tree_digest(ROOT),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "evidence_level": receipt["evidence_level"],
                "decision": receipt["decision"],
                "train_replicates": receipt["train_replicates"],
                "canonical_index": receipt["canonical_index"],
                "artifact_digest": receipt["artifact_digest"],
                "fresh_evaluation_lineage_consumed": receipt[
                    "fresh_evaluation_lineage_consumed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
