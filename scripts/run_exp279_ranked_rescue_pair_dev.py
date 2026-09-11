from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_paired_runner import (
    PRIMARY_METRIC,
    run_exp279_paired_development,
)
from nolane_ai.experiments.exp279_ranked_rescue_calibration import (
    run_exp279_ranked_rescue_development,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol

SCHEMA = "NLM-EXP-279-RANKED-RESCUE-CALIBRATION-PAIR-DEV-V1"
PREREGISTERED_EVAL_START = 50_000
PREREGISTERED_EVAL_REPLICATES = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired canonical-vs-ranked-rescue EXP-279 DEVELOPMENT control"
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols" / "stage_a_v1.json")
    parser.add_argument(
        "--protocol-digest-file",
        type=Path,
        default=ROOT / "protocols" / "stage_a_v1.sha256",
    )
    parser.add_argument("--root-seed", default="20260911-exp279-ranked-rescue-dev")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--target-parameters", type=int, default=500_000)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--train-replicates", type=int, required=True)
    parser.add_argument("--eval-replicates", type=int, default=PREREGISTERED_EVAL_REPLICATES)
    parser.add_argument("--eval-start-replicate", type=int, default=PREREGISTERED_EVAL_START)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-accounted-flops-per-episode", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _verified_protocol(protocol_path: Path, digest_path: Path) -> str:
    load_and_validate_protocol(protocol_path)
    actual = file_sha256(protocol_path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    require_canonical_stage_a_v1_digest(actual)
    return actual


def _route_fraction(payload: dict) -> float:
    rows = payload["evaluation"]["per_replicate"]
    routed = sum(int(row["hybrid_route_receipt"]["routed_episodes"]) for row in rows)
    total = sum(int(row["hybrid_route_receipt"]["batch_size"]) for row in rows)
    return routed / total if total else 0.0


def _evaluation_batch_digests(payload: dict) -> list[str]:
    return [str(row["paired_batch_digest"]) for row in payload["evaluation"]["per_replicate"]]


def _arm_metrics(payload: dict, arm: str) -> list[dict]:
    return [row[arm] for row in payload["evaluation"]["per_replicate"]]


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    if args.eval_start_replicate != PREREGISTERED_EVAL_START:
        raise SystemExit(
            f"this preregistered intervention requires eval_start_replicate={PREREGISTERED_EVAL_START}"
        )
    if args.eval_replicates != PREREGISTERED_EVAL_REPLICATES:
        raise SystemExit(
            f"this preregistered intervention requires exactly {PREREGISTERED_EVAL_REPLICATES} evaluation replicates"
        )
    if args.route_threshold != 0.5:
        raise SystemExit("EXP-279 frozen route threshold must remain 0.5")
    if args.train_replicates not in (15, 60, 120):
        raise SystemExit("preregistered train_replicates must be one of 15, 60, 120")

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    code_digest = source_tree_digest(ROOT)
    common = dict(
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
        code_digest=code_digest,
        max_accounted_flops_per_episode=args.max_accounted_flops_per_episode,
    )

    baseline = run_exp279_paired_development(**common)
    intervention = run_exp279_ranked_rescue_development(**common)

    control_identity = {
        "initial_state_match": baseline["initial_state"] == intervention["initial_state"],
        "training_batch_lineage_match": (
            baseline["training"]["paired_batch_digests"]
            == intervention["training"]["paired_batch_digests"]
        ),
        "evaluation_batch_lineage_match": (
            _evaluation_batch_digests(baseline) == _evaluation_batch_digests(intervention)
        ),
        "pair_audit_digest_match": (
            baseline["resource_match"]["pair_audit_digest"]
            == intervention["resource_match"]["pair_audit_digest"]
        ),
        "propagation_final_state_match": (
            baseline["final_state"]["propagation_only_digest"]
            == intervention["final_state"]["propagation_only_digest"]
        ),
        "branch_final_state_match": (
            baseline["final_state"]["branch_only_digest"]
            == intervention["final_state"]["branch_only_digest"]
        ),
        "propagation_evaluation_metrics_match": (
            _arm_metrics(baseline, "propagation_only")
            == _arm_metrics(intervention, "propagation_only")
        ),
        "branch_evaluation_metrics_match": (
            _arm_metrics(baseline, "branch_only")
            == _arm_metrics(intervention, "branch_only")
        ),
    }
    if not all(control_identity.values()):
        failed = [key for key, value in control_identity.items() if value is not True]
        raise RuntimeError("paired ranked-rescue control identity failed: " + ", ".join(failed))

    baseline_agg = baseline["evaluation"]["aggregate"]
    intervention_agg = intervention["evaluation"]["aggregate"]
    baseline_utility = float(baseline_agg["mean_hybrid_utility"])
    intervention_utility = float(intervention_agg["mean_hybrid_utility"])
    utility_delta = intervention_utility - baseline_utility
    relative_utility_delta = utility_delta / max(abs(baseline_utility), 1e-12)
    receipt = intervention["training"]["ranked_rescue_calibration"]

    comparison = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "experiment_id": "EXP-279",
        "lineage_role": "fresh_primary_single_use",
        "training_replicates": args.train_replicates,
        "evaluation_start_replicate": args.eval_start_replicate,
        "evaluation_replicates": args.eval_replicates,
        "route_threshold": args.route_threshold,
        "control_identity": control_identity,
        "baseline": {
            "artifact_digest": baseline["artifact_digest"],
            "final_hybrid_digest": baseline["final_state"]["hybrid_digest"],
            "route_fraction": _route_fraction(baseline),
            "mean_hybrid_solution_rate": baseline_agg["mean_hybrid_solution_rate"],
            "mean_hybrid_utility": baseline_utility,
            "hybrid_relative_utility_gain_vs_best_simple": baseline_agg["hybrid_relative_utility_gain"],
            "best_simple_arm": baseline_agg["best_simple_arm"],
        },
        "ranked_rescue": {
            "artifact_digest": intervention["artifact_digest"],
            "final_hybrid_digest": intervention["final_state"]["hybrid_digest"],
            "natural_positive_count": receipt["natural_positive_count"],
            "natural_total_count": receipt["natural_total_count"],
            "natural_positive_fraction": receipt["natural_positive_fraction"],
            "break_even_probability": receipt["break_even_probability"],
            "route_fraction": _route_fraction(intervention),
            "mean_hybrid_solution_rate": intervention_agg["mean_hybrid_solution_rate"],
            "mean_hybrid_utility": intervention_utility,
            "hybrid_relative_utility_gain_vs_best_simple": intervention_agg["hybrid_relative_utility_gain"],
            "best_simple_arm": intervention_agg["best_simple_arm"],
        },
        "delta_ranked_rescue_minus_baseline": {
            "route_fraction": _route_fraction(intervention) - _route_fraction(baseline),
            "mean_hybrid_solution_rate": (
                float(intervention_agg["mean_hybrid_solution_rate"])
                - float(baseline_agg["mean_hybrid_solution_rate"])
            ),
            "mean_hybrid_utility": utility_delta,
            "relative_mean_hybrid_utility": relative_utility_delta,
        },
        "preregistered_survival_rule": {
            "routing": "at least one of train=60 or train=120 must have ranked_rescue.route_fraction > 0",
            "utility": "train=60 and train=120 ranked_rescue mean_hybrid_utility must each be >= paired baseline",
            "rule_changed_after_results": False,
        },
        "primary_metric": PRIMARY_METRIC,
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
        "analysis_boundary": (
            "single-use DEVELOPMENT paired baseline-vs-ranked-rescue comparison on preregistered evaluation IDs 50000..50032; cannot promote EXP-279"
        ),
        "comparison_digest": "",
    }
    comparison["comparison_digest"] = canonical_sha256(
        {key: value for key, value in comparison.items() if key != "comparison_digest"}
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "baseline.json").write_text(
        json.dumps(baseline, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ranked-rescue.json").write_text(
        json.dumps(intervention, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
