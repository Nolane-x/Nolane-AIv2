from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.model.config import NLMConfig
from nolane_ai.protocol.identity import file_sha256, source_tree_digest
from nolane_ai.protocol.schema import load_and_validate_protocol
from nolane_ai.training.curriculum import StageACurriculum
from nolane_ai.training.evaluation import run_heldout_development_evaluation
from nolane_ai.training.heldout_artifact import build_heldout_artifact
from nolane_ai.training.pilot import (
    StageAPilotTrainer,
    build_seeded_pilot_model,
    pilot_model_init_seed,
)

PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def _verified_protocol_digest() -> str:
    load_and_validate_protocol(PROTOCOL)
    actual = file_sha256(PROTOCOL)
    expected = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"protocol digest mismatch: expected {expected!r}, got {actual}")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EV-E2 held-out neural development evaluation for the Stage-A pilot"
    )
    parser.add_argument("--tiny", action="store_true", help="use tiny CPU-safe pilot config")
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--train-start-replicate", type=int, default=0)
    parser.add_argument("--eval-start-replicate", type=int, default=10000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variables", type=int, default=4)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--root-seed", default="20260906-neural-heldout")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_steps <= 0 or args.eval_batches <= 0:
        raise SystemExit("--train-steps and --eval-batches must be positive")
    if args.train_start_replicate < 0 or args.eval_start_replicate < 0:
        raise SystemExit("replicate indices must be non-negative")

    config = (
        NLMConfig.stage_a_pilot_tiny_for_tests()
        if args.tiny
        else NLMConfig.stage_a_pilot_16m()
    )
    protocol_digest = _verified_protocol_digest()
    code_digest = source_tree_digest(ROOT)
    init_seed = pilot_model_init_seed(args.root_seed)
    model = build_seeded_pilot_model(config, root_seed=args.root_seed, device="cpu")
    trainer = StageAPilotTrainer(
        model,
        curriculum=StageACurriculum(root_seed=args.root_seed),
        lr=args.lr,
        weight_decay=args.weight_decay,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
        model_init_seed=init_seed,
    )
    report = run_heldout_development_evaluation(
        trainer,
        train_steps=args.train_steps,
        train_start_replicate=args.train_start_replicate,
        eval_start_replicate=args.eval_start_replicate,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        variables=args.variables,
        constraints=args.constraints,
    )
    payload = build_heldout_artifact(
        report,
        trainer=trainer,
        train_steps=args.train_steps,
        train_start_replicate=args.train_start_replicate,
        eval_start_replicate=args.eval_start_replicate,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        variables=args.variables,
        constraints=args.constraints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "evidence_level": report.evidence_level,
                "decision": report.decision,
                "report_digest": report.report_digest,
                "artifact_digest": payload["artifact_digest"],
                "belief_accuracy_delta": report.delta.belief_accuracy,
                "conflict_balanced_accuracy_delta": report.delta.conflict_balanced_accuracy,
                "fidelity_balanced_accuracy_delta": report.delta.fidelity_balanced_accuracy,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
