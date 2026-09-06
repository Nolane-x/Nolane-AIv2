from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched EV-E2 EXP-279 propagation/branch/hybrid DEVELOPMENT evaluation"
    )
    parser.add_argument("--tiny", action="store_true", help="use CPU-safe tiny matched routing arms")
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
    parser.add_argument("--train-replicates", type=int, default=18)
    parser.add_argument("--eval-replicates", type=int, default=18)
    parser.add_argument("--eval-start-replicate", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-accounted-flops-per-episode", type=int, default=None)
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


def main() -> int:
    args = parse_args()
    outputs = (args.output, args.registry_output)
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None:
        raise SystemExit(f"output already exists: {existing}")

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

    execution = run_exp279_paired_development(
        root_seed=args.root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
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

    triplet_audit = execution["resource_match"]["triplet_audit"]
    pilot_audit = audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))
    registry = build_neural_arm_registry(
        protocol=protocol,
        protocol_digest=protocol_digest,
        model_audit=pilot_audit,
        exp279_pair_audit=triplet_audit,
        exp279_execution_artifact=execution,
    )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(execution, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    args.registry_output.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    exp279 = registry["experiments"]["EXP-279"]
    print(
        json.dumps(
            {
                "schema": execution["schema"],
                "evidence_level": execution["evidence_level"],
                "decision": execution["decision"],
                "execution_digest": execution["artifact_digest"],
                "registry_digest": registry["registry_digest"],
                "match_court": exp279["match_court"],
                "development_match_status": exp279["development_match_status"],
                "confirmatory_data_consumed": execution["confirmatory_data_consumed"],
                "challenge_materialized": execution["challenge_materialized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
