from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments import exp301_runner as runner
from nolane_ai.experiments.exp301_compute import COMPUTE_LEDGER_VERSION
from nolane_ai.experiments.exp301_identity import (
    EXP301_PREREG_V2_DIGEST,
    FrozenImplementationIdentity,
    build_frozen_implementation_identity,
)


def _frozen() -> FrozenImplementationIdentity:
    return build_frozen_implementation_identity(
        source_commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        prereg_semantic_digest=EXP301_PREREG_V2_DIGEST,
        architecture_receipts_digest="c" * 64,
        scientific_execution_contract_digest="d" * 64,
        train_generator_digest="e" * 64,
        development_generator_digest="f" * 64,
        challenge_generator_digest="1" * 64,
        parameter_audit_digest="2" * 64,
        compute_ledger_version=COMPUTE_LEDGER_VERSION,
        analysis_digest="3" * 64,
        workflow_digest="4" * 64,
    )


def test_runner_has_no_legacy_root_specific_predata_identity_api() -> None:
    assert not hasattr(runner, "Exp301ExecutionIdentity")
    assert not hasattr(runner, "build_execution_identity")
    assert not hasattr(runner, "canonical_execution_identity_digest")


def test_frozen_identity_loader_revalidates_digest_and_schema(tmp_path: Path) -> None:
    frozen = _frozen()
    path = tmp_path / "frozen.json"
    runner.write_once_json(path, asdict(frozen))
    loaded = runner.load_frozen_implementation_identity(path)
    assert loaded == frozen

    payload = asdict(frozen)
    payload["frozen_implementation_digest"] = "0" * 64
    bad = tmp_path / "bad.json"
    runner.write_once_json(bad, payload)
    with pytest.raises(ValueError, match="digest"):
        runner.load_frozen_implementation_identity(bad)


def test_frozen_identity_loader_rejects_runtime_root_fields(tmp_path: Path) -> None:
    payload = asdict(_frozen())
    payload["root"] = 0
    path = tmp_path / "root-leak.json"
    runner.write_once_json(path, payload)
    with pytest.raises(ValueError, match="unexpected"):
        runner.load_frozen_implementation_identity(path)


def test_write_once_json_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    runner.write_once_json(target, {"schema": "x", "value": 1})
    with pytest.raises(FileExistsError):
        runner.write_once_json(target, {"schema": "x", "value": 2})


def test_test_only_court_exercises_pipeline_without_scientific_authority(tmp_path: Path) -> None:
    receipt = runner.run_test_only_court(tmp_path / "court")

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
        runner.run_test_only_court(tmp_path / "court")


def test_infrastructure_retry_requires_invalid_unconsumed_attempt() -> None:
    original = {
        "decision": "INVALID_COURT",
        "scientific_outcome_consumed": False,
        "run_identity": "a" * 64,
    }
    receipt = runner.validate_infrastructure_retry(original, reason="runner preemption")
    assert receipt["prior_run_identity"] == "a" * 64
    assert receipt["scientific_outcome_consumed"] is False

    with pytest.raises(ValueError, match="invalid"):
        runner.validate_infrastructure_retry({**original, "decision": "KILL_H_RD_01"}, reason="x")
    with pytest.raises(ValueError, match="consumed"):
        runner.validate_infrastructure_retry({**original, "scientific_outcome_consumed": True}, reason="x")


def test_cli_exposes_only_rootless_frozen_identity_and_no_scientific_tuning_flags() -> None:
    help_text = runner.build_parser().format_help()
    assert "--output-dir" in help_text
    assert "--test-only" in help_text
    assert "--frozen-implementation-identity" in help_text
    assert "--execution-identity" not in help_text
    for forbidden in ("--lr", "--loops", "--root", "--threshold", "--bootstrap-seed", "--task-weight", "--sample-count"):
        assert forbidden not in help_text


def test_scientific_cli_uses_only_frozen_identity_and_workflow_bound_root(monkeypatch, tmp_path: Path) -> None:
    frozen = _frozen()
    identity_path = tmp_path / "frozen.json"
    runner.write_once_json(identity_path, asdict(frozen))
    monkeypatch.setenv("EXP301_ROOT", "2")
    monkeypatch.setenv("EXP301_CHALLENGE_BEACON", "workflow-run-123")
    monkeypatch.setenv("EXP301_DEVICE", "cpu")
    captured = {}

    def fake_run_scientific_root(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runner, "run_scientific_root", fake_run_scientific_root, raising=False)
    assert runner.main(
        [
            "--output-dir",
            str(tmp_path / "scientific"),
            "--frozen-implementation-identity",
            str(identity_path),
        ]
    ) == 0
    assert captured == {
        "root": 2,
        "device": "cpu",
        "output_dir": tmp_path / "scientific",
        "frozen_implementation_identity": frozen,
        "challenge_beacon": "workflow-run-123",
    }
