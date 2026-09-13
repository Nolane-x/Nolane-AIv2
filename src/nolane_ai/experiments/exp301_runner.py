from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .exp301_analysis import reduce_exp301
from .exp301_compute import COMPUTE_LEDGER_VERSION
from .exp301_evaluation import commit_prediction, score_committed_prediction
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss
from .exp301_worlds import materialize_world_set

EXP301_PREREG_V2_DIGEST = "1660a728990290f1c605941d531be1d52fe82591c619a327d104e4b87c1e6bad"
EXP301_ROOTS = (0, 1, 2, 3)
EXP301_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EXP301_PRIMARY_EFFORTS = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class Exp301ExecutionIdentity:
    source_commit_sha: str
    source_tree_digest: str
    prereg_semantic_digest: str
    architecture_receipts_digest: str
    root: int
    train_generator_digest: str
    development_generator_digest: str
    challenge_generator_digest: str
    selected_hyperparameter_receipt_digest: str
    parameter_audit_digest: str
    compute_ledger_version: str
    scientific_evidence_eligible: bool
    run_identity: str


def _sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _identity_payload(identity: Exp301ExecutionIdentity) -> dict[str, Any]:
    payload = asdict(identity)
    payload.pop("run_identity", None)
    return payload


def canonical_execution_identity_digest(identity: Exp301ExecutionIdentity) -> str:
    return _sha256(_identity_payload(identity))


def _require_hex(label: str, value: str, length: int) -> None:
    if len(value) != length or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ValueError(f"{label} must be {length} lowercase hex characters")


def build_execution_identity(**values: Any) -> Exp301ExecutionIdentity:
    if values.get("prereg_semantic_digest") != EXP301_PREREG_V2_DIGEST:
        raise ValueError("prereg semantic digest must match frozen EXP-301 V2")
    if values.get("root") not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    if values.get("compute_ledger_version") != COMPUTE_LEDGER_VERSION:
        raise ValueError("compute ledger version mismatch")
    _require_hex("source_commit_sha", values["source_commit_sha"], 40)
    for field in (
        "source_tree_digest",
        "architecture_receipts_digest",
        "train_generator_digest",
        "development_generator_digest",
        "challenge_generator_digest",
        "selected_hyperparameter_receipt_digest",
        "parameter_audit_digest",
    ):
        _require_hex(field, values[field], 64)
    provisional = Exp301ExecutionIdentity(**values, run_identity="")
    return Exp301ExecutionIdentity(**values, run_identity=canonical_execution_identity_digest(provisional))


def write_once_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


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
    parser.add_argument("--execution-identity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.test_only:
        run_test_only_court(args.output_dir)
        return 0
    if not args.execution_identity:
        raise SystemExit("scientific execution requires --execution-identity after pre-data freeze")
    raise SystemExit("scientific EXP-301 execution is fail-closed until the pre-data freeze gate is complete")


if __name__ == "__main__":
    raise SystemExit(main())
