from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

# Load the already-green Task-6 fixture by exact sibling path. Pytest's import
# path differs between local and hosted execution, so avoid relying on tests/
# being an import package while still reusing the real checkpoint/seal setup.
_support_path = Path(__file__).with_name("test_exp277_reconstruction_court.py")
_support_spec = importlib.util.spec_from_file_location("exp277_reconstruction_test_support", _support_path)
assert _support_spec is not None and _support_spec.loader is not None
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
court_fixture = _support.court_fixture


def _execute(fixture, *, beacon=None, source_tree_digest=None, executor_code_digest=None):
    from nolane_ai.experiments.exp277_confirmatory_executor import execute_exp277_confirmatory_challenge

    return execute_exp277_confirmatory_challenge(
        reconstruction_authorization=fixture["reconstruction"],
        seal=fixture["seal"],
        beacon_receipt=beacon or fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        current_source_tree_digest=source_tree_digest or fixture["reconstruction"]["source_tree_digest"],
        executor_code_digest=executor_code_digest or "3" * 64,
    )


def test_exp277_raw_executor_consumes_exact_reserved_lineage_without_parameter_writes(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import validate_exp277_confirmatory_raw
    from nolane_ai.protocol.identity import file_sha256

    fixture = court_fixture
    if "reconstruction" not in fixture:
        from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

        fixture["reconstruction"] = build_exp277_reconstruction_authorization(
            seal=fixture["seal"],
            reconstruction_code_digest="4" * 64,
        )

    checkpoint_sha_before = file_sha256(fixture["checkpoint_path"])
    raw = _execute(fixture)
    assert raw["schema"] == "NLM-EXP-277-CONFIRMATORY-CHALLENGE-RAW-V1"
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["status"] == "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["test_only"] is False
    assert raw["scientific_evidence_eligible"] is True
    assert raw["confirmatory_data_consumed"] is True
    assert raw["challenge_materialized"] is True
    assert raw["seed_materialization_status"] == "EXECUTED"
    assert raw["decision_rule_executed"] is False
    assert raw["confirmatory_n"] == len(raw["reserved_replicate_ids"])
    assert [row["replicate"] for row in raw["per_replicate"]] == raw["reserved_replicate_ids"]
    assert all(row["paired"]["world_pairing_closed"] is True for row in raw["per_replicate"])
    assert all(
        row["checkpoint_functional_state_before"] == row["checkpoint_functional_state_after"]
        for row in raw["per_replicate"]
    )
    assert all(
        row["oracle_information_receipt"]["arcs_received_oracle_incidence"] is False
        for row in raw["per_replicate"]
    )
    assert file_sha256(fixture["checkpoint_path"]) == checkpoint_sha_before
    assert validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


def test_exp277_test_only_executor_cannot_create_scientific_evidence(court_fixture) -> None:
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    fixture = court_fixture
    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    beacon = build_test_beacon_receipt(
        source="synthetic-test-beacon",
        beacon_id="test-round-277",
        published_at_utc="2026-09-08T10:02:00Z",
        entropy_hex="cd" * 32,
        evidence_reference="test-only://exp277",
    )
    raw = _execute(fixture, beacon=beacon)
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["test_only"] is True
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["synthetic_challenge_data_consumed"] is True
    assert raw["decision"] == "UNVERIFIED"
    assert raw["decision_rule_executed"] is False


def test_exp277_executor_rejects_source_tree_or_executor_machinery_drift(court_fixture) -> None:
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    fixture = court_fixture
    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    with pytest.raises(ValueError, match="source tree"):
        _execute(fixture, source_tree_digest="9" * 64)
    with pytest.raises(ValueError, match="executor"):
        _execute(fixture, executor_code_digest="8" * 64)


def test_exp277_executor_rejects_checkpoint_tamper_before_challenge_execution(court_fixture) -> None:
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    fixture = court_fixture
    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    bad = deepcopy(fixture["checkpoint_receipt"])
    bad["scientific_identity_digest"] = "7" * 64
    from nolane_ai.experiments.exp277_confirmatory_executor import execute_exp277_confirmatory_challenge

    with pytest.raises(ValueError, match="checkpoint"):
        execute_exp277_confirmatory_challenge(
            reconstruction_authorization=fixture["reconstruction"],
            seal=fixture["seal"],
            beacon_receipt=fixture["beacon"],
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=bad,
            current_source_tree_digest=fixture["reconstruction"]["source_tree_digest"],
            executor_code_digest="3" * 64,
        )


def test_exp277_raw_validator_rejects_reordered_rows_after_rehash(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import _artifact_digest, validate_exp277_confirmatory_raw
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    fixture = court_fixture
    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    raw = _execute(fixture)
    raw["per_replicate"][0], raw["per_replicate"][1] = raw["per_replicate"][1], raw["per_replicate"][0]
    raw["artifact_digest"] = _artifact_digest(raw)
    errors = validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert any("replicate lineage" in error.lower() for error in errors)


def test_exp277_raw_validator_rejects_rehashed_semantic_row_tamper(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import _artifact_digest, validate_exp277_confirmatory_raw
    from nolane_ai.experiments.exp277_reconstruction_court import _row_digest, build_exp277_reconstruction_authorization

    fixture = court_fixture
    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    raw = _execute(fixture)
    raw["per_replicate"][0]["arcs_branch"]["verified_solution_rate"] = 0.123
    raw["per_replicate"][0]["row_digest"] = _row_digest(raw["per_replicate"][0])
    raw["artifact_digest"] = _artifact_digest(raw)
    errors = validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert any("solution" in error.lower() for error in errors)
