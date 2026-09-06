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
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.protocol.identity import file_sha256, source_tree_digest
from nolane_ai.protocol.schema import load_and_validate_protocol
from nolane_ai.training.curriculum import StageACurriculum
from nolane_ai.training.pilot import (
    StageAPilotTrainer,
    build_seeded_pilot_model,
    pilot_model_init_seed,
    save_pilot_checkpoint,
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
    parser = argparse.ArgumentParser(description="Run an EV-E2 Stage-A neural pilot training smoke")
    parser.add_argument("--tiny", action="store_true", help="use tiny CPU-safe pilot config")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variables", type=int, default=4)
    parser.add_argument("--constraints", type=int, default=3)
    parser.add_argument("--root-seed", default="20260906-neural-pilot")
    parser.add_argument("--start-replicate", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    config = (
        NLMConfig.stage_a_pilot_tiny_for_tests()
        if args.tiny
        else NLMConfig.stage_a_pilot_16m()
    )
    protocol_digest = _verified_protocol_digest()
    code_digest = source_tree_digest(ROOT)
    init_seed = pilot_model_init_seed(args.root_seed)
    model = build_seeded_pilot_model(config, root_seed=args.root_seed, device="cpu")
    curriculum = StageACurriculum(root_seed=args.root_seed)
    trainer = StageAPilotTrainer(
        model,
        curriculum=curriculum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
        model_init_seed=init_seed,
    )
    telemetry = trainer.train_steps(
        steps=args.steps,
        start_replicate=args.start_replicate,
        batch_size=args.batch_size,
        variables=args.variables,
        constraints=args.constraints,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = save_pilot_checkpoint(
        trainer,
        tensor_path=args.output_dir / "pilot.pt",
        manifest_path=args.output_dir / "pilot.manifest.json",
        last_telemetry=telemetry[-1],
    )
    print(
        json.dumps(
            {
                "training_step": manifest["training_step"],
                "evidence_level": manifest["evidence_level"],
                "decision": manifest["decision"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "total_parameters": manifest["total_parameters"],
                "functional_parameters": manifest["functional_parameters"],
                "reserved_parameters": manifest["reserved_parameters"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
