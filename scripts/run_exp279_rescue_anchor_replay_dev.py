from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_rescue_anchor_replay import (
    run_exp279_rescue_anchor_replay_development,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DEVELOPMENT-only EXP-279 rescue-anchor replay intervention"
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--root-seed", default="20260906-exp279-paired-dev")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--target-parameters", type=int, default=500_000)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--train-replicates", type=int, default=15)
    parser.add_argument("--eval-replicates", type=int, default=33)
    parser.add_argument("--eval-start-replicate", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-accounted-flops-per-episode", type=int, default=None)
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


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    execution = run_exp279_rescue_anchor_replay_development(
        root_seed=args.root_seed,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        target_parameters=args.target_parameters,
        route_threshold=args.route_threshold,
        train_replicates=args.train_replicates,
        eval_replicates=args.eval_replicates,
        eval_start_replicate=args.eval_start_replicate,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        variables=args.variables,
        constraints=args.constraints,
        noise_std=args.noise_std,
        lr=args.lr,
        weight_decay=args.weight_decay,
        protocol_digest=protocol_digest,
        code_digest=source_tree_digest(ROOT),
        max_accounted_flops_per_episode=args.max_accounted_flops_per_episode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(execution, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    aggregate = execution["evaluation"]["aggregate"]
    route_rows = execution["evaluation"]["per_replicate"]
    routed = sum(
        int(row["hybrid_route_receipt"]["routed_episodes"])
        for row in route_rows
    )
    episodes = sum(
        int(row["hybrid_route_receipt"]["batch_size"])
        for row in route_rows
    )
    print(
        json.dumps(
            {
                "schema": execution["schema"],
                "evidence_level": execution["evidence_level"],
                "decision": execution["decision"],
                "artifact_digest": execution["artifact_digest"],
                "training_replicates": execution["training"]["replicates"],
                "evaluation_start_replicate": execution["evaluation"]["start_replicate"],
                "evaluation_replicates": execution["evaluation"]["replicates"],
                "final_anchor_episodes": execution["training"]["rescue_anchor_replay"]["final_anchor_episodes"],
                "anchor_batches": execution["training"]["rescue_anchor_replay"]["anchor_batches"],
                "routed_episodes": routed,
                "total_episodes": episodes,
                "route_fraction": routed / episodes if episodes else 0.0,
                "hybrid_solution_rate": aggregate["mean_hybrid_solution_rate"],
                "hybrid_relative_utility_gain": aggregate["hybrid_relative_utility_gain"],
                "best_simple_arm": aggregate["best_simple_arm"],
                "confirmatory_data_consumed": execution["confirmatory_data_consumed"],
                "challenge_materialized": execution["challenge_materialized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
