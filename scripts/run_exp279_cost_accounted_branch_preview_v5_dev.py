from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_cost_accounted_branch_preview_v5 import (
    run_exp279_cost_accounted_branch_preview_v5_court,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EXP-279 cost-accounted branch-preview V5 DEVELOPMENT court"
    )
    parser.add_argument("--tiny", action="store_true", help="use CPU-safe tiny arm geometry")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--root-seed", default="20260911-exp279-cost-accounted-branch-preview-v5-dev")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--target-parameters", type=int, default=500_000)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--train-replicates", type=int, default=60)
    parser.add_argument("--probe-replicates", type=int, default=198)
    parser.add_argument("--probe-folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
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
    code_digest = source_tree_digest(ROOT)

    if args.tiny:
        d_model = 8
        hidden_size = 6
        target_parameters = 5_000
    else:
        d_model = args.d_model
        hidden_size = args.hidden_size
        target_parameters = args.target_parameters

    artifact = run_exp279_cost_accounted_branch_preview_v5_court(
        root_seed=args.root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=args.route_threshold,
        train_replicates=args.train_replicates,
        probe_replicates=args.probe_replicates,
        probe_folds=args.probe_folds,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        variables=args.variables,
        constraints=args.constraints,
        noise_std=args.noise_std,
        lr=args.lr,
        weight_decay=args.weight_decay,
        probe_steps=args.probe_steps,
        probe_lr=args.probe_lr,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
        max_accounted_flops_per_episode=args.max_accounted_flops_per_episode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": artifact["schema"],
                "evidence_level": artifact["evidence_level"],
                "decision": artifact["decision"],
                "artifact_digest": artifact["artifact_digest"],
                "court_classification": artifact["court_classification"],
                "fresh_evaluation_lineage_consumed": artifact["fresh_evaluation_lineage_consumed"],
                "confirmatory_data_consumed": artifact["confirmatory_data_consumed"],
                "challenge_materialized": artifact["challenge_materialized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
