from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nolane_ai.experiments.exp319_challenge import (
    StageBSelectionSeal,
    StageCManifest,
    StageCPredictionCommitment,
    StageCPredictionShard,
    merge_prediction_shards,
    score_stage_c_commitments,
    seal_stage_b_selection_authority,
)
from nolane_ai.experiments.exp319_evidence import StageASelectionSeal, build_root_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal one complete EXP-319 Stage C root")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--stage-a-selections", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--training-contract-digest", required=True)
    parser.add_argument("--scoring-contract-digest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_manifest(path: str) -> StageCManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["selection_roots"] = tuple(int(item) for item in raw["selection_roots"])
    raw["primary_arms"] = tuple(str(item) for item in raw["primary_arms"])
    raw["content_ids"] = tuple(str(item) for item in raw["content_ids"])
    return StageCManifest(**raw)


def _seal_from_payload(payload: dict) -> StageBSelectionSeal:
    raw = payload.get("seal", payload)
    return StageBSelectionSeal(
        arm_id=str(raw["arm_id"]), root=int(raw["root"]), selected_step=int(raw["selected_step"]),
        passes_floor=bool(raw["passes_floor"]), checkpoint_digest=str(raw["checkpoint_digest"]),
        selection_digest=str(raw["selection_digest"]),
    )


def _load_authority(path: str):
    target = Path(path)
    if target.is_dir():
        payloads = [json.loads(item.read_text(encoding="utf-8")) for item in sorted(target.glob("*.json"))]
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
        payloads = list(data["selections"]) if "selections" in data else [data]
    return seal_stage_b_selection_authority(_seal_from_payload(item) for item in payloads)


def _load_shard(path: str) -> StageCPredictionShard:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    commitments = []
    for item in raw["commitments"]:
        item = dict(item)
        item["generated_token_ids"] = tuple(int(token) for token in item["generated_token_ids"])
        commitments.append(StageCPredictionCommitment(**item))
    return StageCPredictionShard(
        schema=str(raw["schema"]), root=int(raw["root"]), shard_index=int(raw["shard_index"]),
        world_count=int(raw["world_count"]), challenge_manifest_digest=str(raw["challenge_manifest_digest"]),
        commitments=tuple(commitments), shard_digest=str(raw["shard_digest"]),
    )


def _load_stage_a(path: str) -> tuple[StageASelectionSeal, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(StageASelectionSeal(
        arm_id=str(item["arm_id"]), passes_floor=bool(item["passes_floor"]),
        selected_learning_rate=float(item["selected_learning_rate"]), selection_digest=str(item["selection_digest"]),
    ) for item in raw["selections"])


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _load_manifest(args.manifest)
    if manifest.root != args.root or manifest.source_digest != args.source_digest:
        raise SystemExit("root/source manifest mismatch")
    authority = _load_authority(args.selection_authority)
    shards = tuple(_load_shard(path) for path in args.shards)
    commitments = merge_prediction_shards(manifest, shards)
    scored = score_stage_c_commitments(manifest, commitments)
    evidence = build_root_evidence(
        root=args.root,
        source_digest=args.source_digest,
        training_contract_digest=args.training_contract_digest,
        scoring_contract_digest=args.scoring_contract_digest,
        stage_a_selections=_load_stage_a(args.stage_a_selections),
        stage_b_selection_authority=authority,
        stage_c_manifest=manifest,
        stage_c_commitments=commitments,
        stage_c_scored_rows=scored,
    )
    _write_once(args.output, asdict(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
