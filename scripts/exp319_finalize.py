from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nolane_ai.experiments.exp319_evidence import (
    RootEvidence,
    StageASelectionSeal,
    StageCArmSummary,
    reduce_cross_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the frozen EXP-319 cross-root diagnostic")
    parser.add_argument("--root-evidence", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_root(path: str) -> RootEvidence:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    stage_a = tuple(StageASelectionSeal(
        arm_id=str(item["arm_id"]),
        passes_floor=bool(item["passes_floor"]),
        selected_learning_rate=float(item["selected_learning_rate"]),
        selection_digest=str(item["selection_digest"]),
    ) for item in raw["stage_a_selections"])
    summaries = tuple(StageCArmSummary(
        arm_id=str(item["arm_id"]),
        answer_token_accuracy=float(item["answer_token_accuracy"]),
        family_exact=tuple((str(name), float(score)) for name, score in item["family_exact"]),
        family_balanced_exact_match=float(item["family_balanced_exact_match"]),
        nonzero_exact_families=int(item["nonzero_exact_families"]),
        eos_correctness=float(item["eos_correctness"]),
        invalid_output_rate=float(item["invalid_output_rate"]),
        passes_floor=bool(item["passes_floor"]),
    ) for item in raw.get("stage_c_summaries", []))
    return RootEvidence(
        schema=str(raw["schema"]), root=int(raw["root"]), provenance_valid=bool(raw["provenance_valid"]),
        source_digest=str(raw["source_digest"]), training_contract_digest=str(raw["training_contract_digest"]),
        scoring_contract_digest=str(raw["scoring_contract_digest"]), stage_c_floor_digest=str(raw["stage_c_floor_digest"]),
        stage_a_selections=stage_a, stage_b_selection_authority_digest=str(raw["stage_b_selection_authority_digest"]),
        stage_b_control_pass=bool(raw["stage_b_control_pass"]), stage_b_nrs_pass=bool(raw["stage_b_nrs_pass"]),
        stage_c_manifest_digest=raw.get("stage_c_manifest_digest"),
        stage_c_commitment_grid_digest=raw.get("stage_c_commitment_grid_digest"),
        stage_c_scoring_digest=raw.get("stage_c_scoring_digest"), stage_c_summaries=summaries,
        stage_c_control_pass=raw.get("stage_c_control_pass"), stage_c_nrs_pass=raw.get("stage_c_nrs_pass"),
        artifact_digest=str(raw["artifact_digest"]),
    )


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = reduce_cross_root(_load_root(path) for path in args.root_evidence)
    if any(evidence.authorizations.values()) or evidence.exp302_implementation_authorized or evidence.exp320_implementation_authorized or evidence.scale_authorized or evidence.authorized_30m or evidence.authorized_100m:
        raise SystemExit("EXP-319 reducer emitted forbidden authorization")
    _write_once(args.output, asdict(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
