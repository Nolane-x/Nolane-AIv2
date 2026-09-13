from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_runner import (
    EXP301_PREREG_V2_DIGEST,
    Exp301ExecutionIdentity,
    build_execution_identity,
    build_parser,
    canonical_execution_identity_digest,
    run_test_only_court,
    validate_infrastructure_retry,
    write_once_json,
)


def _identity(**changes) -> Exp301ExecutionIdentity:
    values = dict(
        source_commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        prereg_semantic_digest=EXP301_PREREG_V2_DIGEST,
        architecture_receipts_digest="c" * 64,
        root=0,
        train_generator_digest="d" * 64,
        development_generator_digest="e" * 64,
        challenge_generator_digest="f" * 64,
        selected_hyperparameter_receipt_digest="1" * 64,
        parameter_audit_digest="2" * 64,
        compute_ledger_version="exp301-accounted-flops-v2",
        scientific_evidence_eligible=True,
    )
    values.update(changes)
    return build_execution_identity(**values)


def test_execution_identity_binds_every_frozen_input() -> None:
    identity = _identity()
    assert identity.run_identity == canonical_execution_identity_digest(identity)
    assert identity.root == 0
    assert identity.scientific_evidence_eligible is True

    changed = _identity(challenge_generator_digest="9" * 64)
    assert changed.run_identity != identity.run_identity


def test_execution_identity_rejects_wrong_prereg_or_root() -> None:
    with pytest.raises(ValueError, match="prereg"):
        _identity(prereg_semantic_digest="0" * 64)
    with pytest.raises(ValueError, match="root"):
        _identity(root=9)


def test_write_once_json_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    write_once_json(target, {"schema": "x", "value": 1})
    with pytest.raises(FileExistsError):
        write_once_json(target, {"schema": "x", "value": 2})


def test_test_only_court_exercises_pipeline_without_scientific_authority(tmp_path: Path) -> None:
    receipt = run_test_only_court(tmp_path / "court")

    assert receipt["schema"] == "EXP301-TEST-ONLY-COURT-V1"
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["semantic_authority_promoted"] is False
    assert receipt["decision"] == "UNVERIFIED_TEST_ONLY"
    assert receipt["test_only_would_be_decision"] in {
        "KILL_H_RD_01",
        "PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY",
        "INVALID_COURT",
    }
    assert receipt["training_probe_steps"] > 0
    assert receipt["evaluation_rows"] > 0
    assert receipt["roots"] == [0, 1, 2, 3]
    assert (tmp_path / "court" / "test-only-court.json").exists()

    with pytest.raises(FileExistsError):
        run_test_only_court(tmp_path / "court")


def test_infrastructure_retry_requires_invalid_unconsumed_attempt() -> None:
    original = {
        "decision": "INVALID_COURT",
        "scientific_outcome_consumed": False,
        "run_identity": "a" * 64,
    }
    receipt = validate_infrastructure_retry(original, reason="runner preemption")
    assert receipt["prior_run_identity"] == "a" * 64
    assert receipt["scientific_outcome_consumed"] is False

    with pytest.raises(ValueError, match="invalid"):
        validate_infrastructure_retry({**original, "decision": "KILL_H_RD_01"}, reason="x")
    with pytest.raises(ValueError, match="consumed"):
        validate_infrastructure_retry({**original, "scientific_outcome_consumed": True}, reason="x")


def test_cli_exposes_no_scientific_tuning_flags() -> None:
    help_text = build_parser().format_help()
    assert "--output-dir" in help_text
    assert "--test-only" in help_text
    assert "--execution-identity" in help_text
    for forbidden in ("--lr", "--loops", "--threshold", "--bootstrap-seed", "--task-weight", "--sample-count"):
        assert forbidden not in help_text
