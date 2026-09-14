from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nolane_ai.experiments.exp301_ceremony import materialize_root_challenge
from nolane_ai.experiments.exp301_evaluation import commit_prediction
from nolane_ai.experiments.exp301_evidence import _selection_from_raw
from nolane_ai.experiments.exp301_runner import load_frozen_implementation_identity
from nolane_ai.experiments.exp301_scientific import scientific_challenge_generate
from nolane_ai.experiments.exp301_scientific_executor import load_selected_models
from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_ARMS,
    FROZEN_EFFORTS,
    FROZEN_IMPLEMENTATION_DIGEST,
    build_prediction_shard,
    challenge_shard_bounds,
    recovery_beacon,
    write_once_json,
)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"EXP-301R requires workflow-bound {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one EXP-301R prediction-only challenge shard")
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--challenge-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-implementation-identity", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = int(_env("EXP301R_ROOT"))
    shard_index = int(_env("EXP301R_SHARD_INDEX"))
    run_id = _env("EXP301R_RUN_ID")
    workflow_digest = _env("EXP301R_WORKFLOW_DIGEST")
    device = _env("EXP301_DEVICE")
    if device != "cpu":
        raise SystemExit("EXP-301R recovery is frozen to CPU execution")

    frozen = load_frozen_implementation_identity(args.frozen_implementation_identity)
    if frozen.frozen_implementation_digest != FROZEN_IMPLEMENTATION_DIGEST:
        raise SystemExit("EXP-301R frozen implementation digest mismatch")

    selected_dir = Path(args.selected_dir)
    selection_path = selected_dir / f"root-{root}" / "selection-manifest.json"
    selection = _selection_from_raw(json.loads(selection_path.read_text(encoding="utf-8")))
    if selection.root != root:
        raise SystemExit("selection manifest root mismatch")

    challenge_manifest = json.loads(Path(args.challenge_manifest).read_text(encoding="utf-8"))
    challenge = materialize_root_challenge(
        root=root,
        beacon=recovery_beacon(run_id),
        frozen_implementation_digest=frozen.frozen_implementation_digest,
    )
    if challenge.challenge_nonce != challenge_manifest.get("challenge_nonce"):
        raise SystemExit("challenge nonce mismatch")
    if challenge.materialization_digest != challenge_manifest.get("challenge_materialization_digest"):
        raise SystemExit("challenge materialization digest mismatch")
    if tuple(world.content_id for world in challenge.worlds) != tuple(
        challenge_manifest.get("ordered_content_ids", ())
    ):
        raise SystemExit("challenge ordered content ids mismatch")

    selected_models = load_selected_models(
        selection,
        output_dir=selected_dir,
        device=device,
    )
    start, end = challenge_shard_bounds(shard_index)
    commitments = []
    for world in challenge.worlds[start:end]:
        for arm_id in FROZEN_ARMS:
            model = selected_models[arm_id]
            for effort in FROZEN_EFFORTS:
                generation = scientific_challenge_generate(model, prompt=world.model_input, effort=effort)
                commitments.append(
                    commit_prediction(
                        world,
                        arm_id=arm_id,
                        root=root,
                        effort_multiplier=effort,
                        candidate_answer=generation.candidate_answer,
                        generation_token_count=generation.generated_token_count,
                    )
                )

    shard = build_prediction_shard(
        challenge_manifest=challenge_manifest,
        shard_index=shard_index,
        commitments=tuple(commitments),
        run_id=run_id,
        workflow_digest=workflow_digest,
        selection_manifest_digest=selection.selection_manifest_digest,
    )
    output = Path(args.output_dir) / f"shard-r{root}-{shard_index:02d}.json"
    write_once_json(output, shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
