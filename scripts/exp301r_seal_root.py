from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nolane_ai.experiments.exp301_ceremony import (
    build_runtime_identity_for_root,
    materialize_root_challenge,
)
from nolane_ai.experiments.exp301_evaluation import PredictionCommitment
from nolane_ai.experiments.exp301_evidence import (
    _selection_from_raw,
    build_root_evidence_artifact,
    write_root_evidence_artifact,
)
from nolane_ai.experiments.exp301_runner import load_frozen_implementation_identity
from nolane_ai.experiments.exp301_scientific_executor import score_challenge_commitments
from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_IMPLEMENTATION_DIGEST,
    build_challenge_manifest,
    build_root_recovery_receipt,
    merge_prediction_shards,
    recovery_beacon,
    write_once_json,
)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"EXP-301R requires workflow-bound {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal one complete EXP-301R root")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--challenge-manifest", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-implementation-identity", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = int(_env("EXP301R_ROOT"))
    run_id = _env("EXP301R_RUN_ID")
    workflow_digest = _env("EXP301R_WORKFLOW_DIGEST")

    frozen = load_frozen_implementation_identity(args.frozen_implementation_identity)
    if frozen.frozen_implementation_digest != FROZEN_IMPLEMENTATION_DIGEST:
        raise SystemExit("EXP-301R frozen implementation digest mismatch")

    selection = _selection_from_raw(
        json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    )
    if selection.root != root:
        raise SystemExit("selection manifest root mismatch")

    challenge = materialize_root_challenge(
        root=root,
        beacon=recovery_beacon(run_id),
        frozen_implementation_digest=frozen.frozen_implementation_digest,
    )
    challenge_manifest_path = Path(args.challenge_manifest)
    challenge_manifest = json.loads(challenge_manifest_path.read_text(encoding="utf-8"))
    expected_manifest = build_challenge_manifest(
        challenge,
        run_id=run_id,
        workflow_digest=workflow_digest,
        selection_manifest_digest=selection.selection_manifest_digest,
    )
    if challenge_manifest != expected_manifest:
        raise SystemExit("challenge manifest does not reproduce from recovery identity")

    shard_paths = tuple(sorted(Path(args.shards_dir).glob(f"shard-r{root}-*.json")))
    raw_shards = tuple(json.loads(path.read_text(encoding="utf-8")) for path in shard_paths)
    merged_raw = merge_prediction_shards(raw_shards)
    commitments = tuple(PredictionCommitment(**item) for item in merged_raw)
    evaluation_rows = score_challenge_commitments(challenge, commitments)

    runtime_identity = build_runtime_identity_for_root(
        frozen_implementation_digest=frozen.frozen_implementation_digest,
        selection_manifest=selection,
        challenge=challenge,
        beacon=recovery_beacon(run_id),
    )
    artifact = build_root_evidence_artifact(
        frozen_implementation_digest=frozen.frozen_implementation_digest,
        selection_manifest=selection,
        challenge=challenge,
        runtime_identity=runtime_identity,
        challenge_beacon=recovery_beacon(run_id),
        commitments=commitments,
        evaluation_rows=evaluation_rows,
    )

    root_dir = Path(args.output_dir) / f"root-{root}"
    write_root_evidence_artifact(root_dir / "root-evidence.json", artifact)
    ordered_shards = tuple(sorted(raw_shards, key=lambda item: int(item["shard_index"])))
    receipt = build_root_recovery_receipt(
        root=root,
        run_id=run_id,
        workflow_digest=workflow_digest,
        selection_manifest_digest=selection.selection_manifest_digest,
        challenge_manifest_digest=challenge_manifest["manifest_digest"],
        root_evidence_artifact_digest=artifact.artifact_digest,
        root_run_identity=artifact.run_identity,
        shard_digests=tuple(item["shard_digest"] for item in ordered_shards),
    )
    write_once_json(root_dir / "root-recovery-receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
