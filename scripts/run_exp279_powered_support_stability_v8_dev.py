from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.exp279_powered_support_stability_v8 import (
    FROZEN_ARM_GEOMETRY,
    FROZEN_OPTIMIZER,
    FROZEN_ROUTE_THRESHOLD,
    FROZEN_WORLD_GEOMETRY,
    PROBE_REPLICATES,
    run_exp279_powered_support_stability_v8_shard,
)
from nolane_ai.protocol.identity import (
    file_sha256,
    require_canonical_stage_a_v1_digest,
    source_tree_digest,
)
from nolane_ai.protocol.schema import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one DEVELOPMENT-only EXP-279 V8 powered support shard"
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
    parser.add_argument("--scientific-branch-head", required=True)
    parser.add_argument("--executed-commit", required=True)
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


def _geometry(tiny: bool) -> dict[str, int | float]:
    if tiny:
        return {
            "d_model": 8,
            "hidden_size": 6,
            "target_parameters": 5_000,
            "route_threshold": 0.5,
            "probe_replicates": 9,
            "batch_size": 2,
            "timesteps": 3,
            "variables": 4,
            "constraints": 2,
            "noise_std": 0.05,
            "lr": 0.001,
            "weight_decay": 0.0,
        }
    return {
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


def _write_receipt(output: Path, receipt: dict) -> None:
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    output.write_bytes(payload)
    output.with_name(output.name + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    protocol_digest = _verified_protocol(args.protocol, args.protocol_digest_file)
    geometry = _geometry(args.tiny_contract)
    receipt = run_exp279_powered_support_stability_v8_shard(
        train_replicates=args.train_replicates,
        canonical_index=args.canonical_index,
        protocol_digest=protocol_digest,
        code_digest=source_tree_digest(ROOT),
        scientific_branch_head=args.scientific_branch_head,
        executed_commit=args.executed_commit,
        **geometry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "artifact_digest": receipt["artifact_digest"],
                "canonical_index": receipt["canonical_index"],
                "scientific_branch_head": receipt["scientific_branch_head"],
                "train_replicates": receipt["train_replicates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
