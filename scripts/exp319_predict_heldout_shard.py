from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nolane_ai.experiments.exp301_scientific import build_scientific_arm
from nolane_ai.experiments.exp319_challenge import (
    StageCManifest,
    StageCPredictionOutput,
    build_prediction_shard,
    prediction_inputs_for_shard,
)
from nolane_ai.experiments.exp319_training import _diagnostic_generate, load_checkpoint_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one answer-blind EXP-319 Stage C prediction shard")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--a-receipt", required=True)
    parser.add_argument("--nrs-checkpoint", required=True)
    parser.add_argument("--nrs-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def _load_manifest(path: str) -> StageCManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["selection_roots"] = tuple(int(item) for item in raw["selection_roots"])
    raw["primary_arms"] = tuple(str(item) for item in raw["primary_arms"])
    raw["content_ids"] = tuple(str(item) for item in raw["content_ids"])
    return StageCManifest(**raw)


def _load_arm(arm_id: str, root: int, checkpoint: str, receipt: str, device: str):
    bundle = load_checkpoint_bundle(checkpoint, receipt)
    if bundle.receipt.stage != "B_TRAIN" or bundle.receipt.arm_id != arm_id or bundle.receipt.root != root:
        raise SystemExit(f"selected checkpoint provenance mismatch for {arm_id}/root-{root}")
    compiled = build_scientific_arm(arm_id, device=device)
    compiled.model.load_state_dict(bundle.model_state_dict)
    compiled.model.eval()
    return compiled


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _load_manifest(args.manifest)
    if manifest.root != args.root:
        raise SystemExit("manifest/root mismatch")
    models = {
        "A_FIXED": _load_arm("A_FIXED", args.root, args.a_checkpoint, args.a_receipt, args.device),
        "C_NRS_CORE": _load_arm("C_NRS_CORE", args.root, args.nrs_checkpoint, args.nrs_receipt, args.device),
    }
    outputs = []
    for item in prediction_inputs_for_shard(manifest, shard_index=args.shard_index):
        for arm_id, compiled in models.items():
            candidate, token_ids, stopped = _diagnostic_generate(compiled, prompt=item.model_input)
            outputs.append(StageCPredictionOutput(
                content_id=item.content_id,
                arm_id=arm_id,
                candidate_answer=candidate,
                generated_token_ids=tuple(token_ids),
                stopped_on_eos=stopped,
            ))
    shard = build_prediction_shard(manifest, shard_index=args.shard_index, outputs=outputs)
    _write_once(args.output, asdict(shard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
