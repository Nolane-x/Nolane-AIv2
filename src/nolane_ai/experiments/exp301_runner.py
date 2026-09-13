from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any

import torch

from .exp301_analysis import reduce_exp301
from .exp301_compute import COMPUTE_LEDGER_VERSION
from .exp301_evaluation import commit_prediction, score_committed_prediction
from .exp301_identity import (
    EXP301_PREREG_V2_DIGEST,
    FrozenImplementationIdentity,
    build_frozen_implementation_identity,
)
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss
from .exp301_worlds import materialize_world_set

EXP301_ROOTS = (0, 1, 2, 3)
EXP301_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EXP301_PRIMARY_EFFORTS = (1, 2, 4, 8)


def write_once_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def load_frozen_implementation_identity(path: str | Path) -> FrozenImplementationIdentity:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("frozen implementation identity must be a JSON object")

    expected_fields = {field.name for field in fields(FrozenImplementationIdentity)}
    actual_fields = set(raw)
    unexpected = sorted(actual_fields - expected_fields)
    missing = sorted(expected_fields - actual_fields)
    if unexpected:
        raise ValueError(f"frozen implementation identity has unexpected fields: {unexpected}")
    if missing:
        raise ValueError(f"frozen implementation identity is missing fields: {missing}")

    supplied_digest = raw["frozen_implementation_digest"]
    build_values = dict(raw)
    build_values.pop("schema")
    build_values.pop("frozen_implementation_digest")
    rebuilt = build_frozen_implementation_identity(**build_values)
    if rebuilt.schema != raw["schema"]:
        raise ValueError(
            f"frozen implementation identity schema mismatch: expected {rebuilt.schema}, got {raw['schema']}"
        )
    if rebuilt.frozen_implementation_digest != supplied_digest:
        raise ValueError(
            "frozen implementation identity digest mismatch: "
            f"expected {rebuilt.frozen_implementation_digest}, got {supplied_digest}"
        )
    return rebuilt


def validate_infrastructure_retry(prior: dict[str, Any], *, reason: str) -> dict[str, Any]:
    if prior.get("decision") != "INVALID_COURT":
        raise ValueError("infrastructure retry requires a prior invalid court")
    if prior.get("scientific_outcome_consumed") is True:
        raise ValueError("infrastructure retry forbidden after scientific outcome was consumed")
    identity = prior.get("run_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("prior run identity is required")
    if not reason.strip():
        raise ValueError("retry reason is required")
    return {
        "schema": "EXP301-INFRASTRUCTURE-RETRY-V1",
        "prior_run_identity": identity,
        "reason": reason,
        "scientific_outcome_consumed": False,
    }


def _run_training_probe() -> int:
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example("test-only training probe", "ok")
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long)
    logits = torch.nn.Parameter(torch.zeros(1, targets.shape[1], tokenizer.vocab_size))
    optimizer = torch.optim.AdamW([logits], lr=1e-4, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = compute_answer_only_loss(logits, targets=targets, answer_start=encoded.answer_start)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([logits], 1.0)
    optimizer.step()
    return 1


def run_test_only_court(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "test-only-court.json"
    if target.exists():
        raise FileExistsError(target)
    training_steps = _run_training_probe()
    worlds = materialize_world_set(split="development", roots=EXP301_ROOTS, per_family=2, test_only=True)
    rows = []
    for world in worlds:
        for effort in EXP301_PRIMARY_EFFORTS:
            for arm in EXP301_ARMS:
                commitment = commit_prediction(
                    world,
                    arm_id=arm,
                    root=world.root,
                    effort_multiplier=effort,
                    candidate_answer="",
                )
                rows.append(score_committed_prediction(commitment, world))
    analysis = reduce_exp301(rows, bootstrap_samples=20)
    receipt = {
        "schema": "EXP301-TEST-ONLY-COURT-V1",
        "scientific_evidence_eligible": False,
        "semantic_authority_promoted": False,
        "scientific_outcome_consumed": False,
        "decision": "UNVERIFIED_TEST_ONLY",
        "test_only_would_be_decision": analysis.decision,
        "training_probe_steps": training_steps,
        "evaluation_rows": len(rows),
        "roots": list(EXP301_ROOTS),
        "prereg_semantic_digest": EXP301_PREREG_V2_DIGEST,
        "compute_ledger_version": COMPUTE_LEDGER_VERSION,
    }
    write_once_json(target, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen EXP-301 execution path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--frozen-implementation-identity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.test_only:
        run_test_only_court(args.output_dir)
        return 0
    if not args.frozen_implementation_identity:
        raise SystemExit(
            "scientific execution requires --frozen-implementation-identity after pre-data freeze"
        )
    load_frozen_implementation_identity(args.frozen_implementation_identity)
    raise SystemExit(
        "scientific EXP-301 execution is fail-closed until the scientific trainer and challenge boundary are frozen"
    )


if __name__ == "__main__":
    raise SystemExit(main())
