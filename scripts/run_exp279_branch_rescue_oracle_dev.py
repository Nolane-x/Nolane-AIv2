from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_branch_rescue_oracle import (
    run_exp279_branch_rescue_oracle_development,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DEVELOPMENT-only EXP-279 same-weights branch rescue/oracle diagnostic"
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
    parser.add_argument("--eval-start-replicate", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.002)
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

    receipt = run_exp279_branch_rescue_oracle_development(
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
        protocol_digest=_verified_protocol(args.protocol, args.protocol_digest_file),
        code_digest=source_tree_digest(ROOT),
        max_accounted_flops_per_episode=args.max_accounted_flops_per_episode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    aggregate = receipt["aggregate"]
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "evidence_level": receipt["evidence_level"],
                "decision": receipt["decision"],
                "training_replicates": receipt["training_replicates"],
                "final_hybrid_digest": receipt["final_hybrid_digest"],
                "branch_rescues": aggregate["branch_rescues"],
                "branch_harms": aggregate["branch_harms"],
                "rescue_rate_among_stop_failures": aggregate[
                    "rescue_rate_among_stop_failures"
                ],
                "mean_stop_solution_rate": aggregate["mean_stop_solution_rate"],
                "mean_forced_branch_solution_rate": aggregate[
                    "mean_forced_branch_solution_rate"
                ],
                "mean_rescue_oracle_solution_rate": aggregate[
                    "mean_rescue_oracle_solution_rate"
                ],
                "mean_stop_utility": aggregate["mean_stop_utility"],
                "mean_always_branch_utility": aggregate["mean_always_branch_utility"],
                "mean_rescue_oracle_utility": aggregate["mean_rescue_oracle_utility"],
                "rescue_oracle_relative_utility_gain": aggregate[
                    "rescue_oracle_relative_utility_gain"
                ],
                "branch_rescue_classification": aggregate[
                    "branch_rescue_classification"
                ],
                "challenge_materialized": receipt["challenge_materialized"],
                "confirmatory_data_consumed": receipt["confirmatory_data_consumed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
