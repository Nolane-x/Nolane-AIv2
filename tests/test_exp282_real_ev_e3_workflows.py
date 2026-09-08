from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CEREMONY = ROOT / ".github" / "workflows" / "exp282-real-ev-e3-ceremony.yml"
PERSISTENCE = ROOT / ".github" / "workflows" / "exp282-persist-real-ev-e3-evidence.yml"
BASELINE_SHA = "77383b0a9c2ce92ded66f07234891790652a037b"
PROTOCOL_SHA = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
ARM_PATH = ".github/ceremony/exp282-real-ev-e3-arm.json"

FORBIDDEN_PERSISTENCE_EXECUTION = (
    "run_exp282_paired_dev.py",
    "prepare_exp282_confirmatory_open.py",
    "seal_exp282_confirmatory_ceremony.py",
    "execute_exp282_confirmatory_ceremony.py",
    "build_exp282_confirmatory_analysis",
)


def _ceremony_text() -> str:
    assert CEREMONY.is_file(), f"missing real EXP-282 ceremony workflow: {CEREMONY}"
    return CEREMONY.read_text(encoding="utf-8")


def _persistence_text() -> str:
    if not PERSISTENCE.is_file():
        pytest.skip("persistence contract activates after first valid ceremony artifact exists")
    return PERSISTENCE.read_text(encoding="utf-8")


def test_real_ev_e3_ceremony_workflow_exists() -> None:
    assert CEREMONY.is_file()


def test_ceremony_is_armed_only_by_explicit_marker_push() -> None:
    text = _ceremony_text()
    assert "branches: [exp282-real-ev-e3-ceremony]" in text
    assert ARM_PATH in text
    assert "workflow_dispatch" not in text


def test_ceremony_proves_frozen_scientific_identity() -> None:
    text = _ceremony_text()
    assert BASELINE_SHA in text
    assert PROTOCOL_SHA in text
    for path in (
        "src",
        "scripts",
        "protocols/stage_a_v1.json",
        "protocols/stage_a_v1.sha256",
    ):
        assert path in text
    assert "source_tree_digest" in text
    assert "python scripts/verify_protocol.py" in text


def test_ceremony_freezes_exact_non_tiny_geometry() -> None:
    text = _ceremony_text()
    for token in (
        "--d-model 64",
        "--hidden-size 48",
        "--target-parameters 500000",
        "--train-replicates 16",
        "--eval-replicates 32",
        "--eval-start-replicate 50000",
        "--batch-size 8",
        "--timesteps 6",
        "--variables 4",
        "--visibility-rate 0.5",
        "--noise-std 0.35",
        "--lr 0.002",
        "--weight-decay 0.0",
    ):
        assert token in text
    assert "--tiny" not in text


def test_ceremony_waits_for_exact_head_ci_before_science() -> None:
    text = _ceremony_text()
    assert "GITHUB_SHA" in text
    assert "actions/workflows/ci.yml/runs" in text
    assert "head_sha=${GITHUB_SHA}" in text
    assert "event=pull_request" in text
    assert "conclusion" in text
    assert "success" in text
    assert "core (3.11)" in text
    assert "core (3.13)" in text
    assert "model-smoke" in text


def test_ceremony_has_pre_execution_replay_barrier() -> None:
    text = _ceremony_text()
    marker = "exp282-real-ev-e3-inference-started-${{ github.sha }}"
    outcome = "exp282-real-ev-e3-outcome-${{ github.sha }}"
    assert marker in text
    assert outcome in text
    marker_index = text.index(marker)
    execute_index = text.index("python scripts/execute_exp282_confirmatory_ceremony.py")
    assert marker_index < execute_index
    assert "actions/upload-artifact@v4" in text[marker_index:execute_index]


def test_ceremony_preserves_not_ready_as_valid_nonexecuted_outcome() -> None:
    text = _ceremony_text()
    assert "NOT_READY_VARIANCE_EXCEEDS_MAX_N" in text
    assert "CONFIRMATORY_OPEN_PREPARED" in text
    assert "confirmatory_data_consumed" in text
    assert "inference_started" in text


def test_ceremony_does_not_hard_code_scientific_outcome_or_challenge() -> None:
    text = _ceremony_text()
    assert "assert analysis['decision'] == 'PROMOTE_TO_NEXT_STAGE'" not in text
    assert 'assert analysis["decision"] == "PROMOTE_TO_NEXT_STAGE"' not in text
    assert "drand" not in text.lower()
    assert "public-beacon" not in text.lower()
    assert "challenge_materialized" in text
    assert "PROMOTE_TO_NEXT_STAGE" in text
    assert "HOLD_UNSTABLE" in text
    assert "KILL_SUBSYSTEM" in text


def test_persistence_is_pinned_and_never_reruns_science() -> None:
    text = _persistence_text()
    assert "actions/artifacts/" in text
    assert "actions/runs/" in text
    assert "SHA256SUMS" in text
    assert "evidence/exp282/2026-09-08-real-ev-e3-confirmatory-open" in text
    assert BASELINE_SHA in text
    for token in FORBIDDEN_PERSISTENCE_EXECUTION:
        assert token not in text


def test_persistence_proves_evidence_only_scope() -> None:
    text = _persistence_text()
    for path in (
        "src",
        "scripts",
        "protocols/stage_a_v1.json",
        "protocols/stage_a_v1.sha256",
    ):
        assert path in text
    assert "actions-artifact-metadata.json" in text
    assert "actions-run-metadata.json" in text
    assert "PROVENANCE.md" in text
