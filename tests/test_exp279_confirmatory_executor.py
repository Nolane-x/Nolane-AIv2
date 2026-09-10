from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


def _fixture(tmp_path):
    from tests.test_exp279_reconstruction_execution import _fixture as build_fixture

    return build_fixture(tmp_path)


def _execute(
    fixture,
    *,
    source_tree_digest=None,
    executor_code_digest=None,
    checkpoint_receipt=None,
):
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        execute_exp279_confirmatory_challenge,
    )

    reconstruction = fixture["reconstruction"]
    return execute_exp279_confirmatory_challenge(
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=checkpoint_receipt or fixture["checkpoint_receipt"],
        current_source_tree_digest=(
            source_tree_digest or reconstruction["source_tree_digest"]
        ),
        executor_code_digest=executor_code_digest or "3" * 64,
    )


def test_exp279_test_only_raw_executor_preserves_unverified_boundary_and_reserved_lineage(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        validate_exp279_confirmatory_raw,
    )

    fixture = _fixture(tmp_path)
    raw = _execute(fixture)
    reconstruction = fixture["reconstruction"]
    reserved = reconstruction["reserved_replicate_ids"]
    schedule = reconstruction["stratum_schedule"]

    assert raw["schema"] == "NLM-EXP-279-CONFIRMATORY-CHALLENGE-RAW-V1"
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["test_only"] is True
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["synthetic_challenge_data_consumed"] is True
    assert raw["challenge_materialized"] is True
    assert raw["seed_materialization_status"] == "TEST_ONLY_EXECUTED"
    assert raw["decision_rule_executed"] is False
    assert raw["confirmatory_n"] == len(reserved)
    assert raw["reserved_replicate_ids"] == reserved
    assert [row["replicate"] for row in raw["per_replicate"]] == reserved
    assert [row["stratum"] for row in raw["per_replicate"]] == [
        schedule[index % len(schedule)] for index in range(len(reserved))
    ]
    assert validate_exp279_confirmatory_raw(
        raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


def test_exp279_executor_integrity_drift_is_invalid_before_challenge_materialization(tmp_path) -> None:
    fixture = _fixture(tmp_path)

    source_invalid = _execute(fixture, source_tree_digest="9" * 64)
    assert source_invalid["status"] == "INVALID_RUN"
    assert source_invalid["decision"] == "INVALID_RUN"
    assert source_invalid["scientific_evidence_eligible"] is False
    assert source_invalid["confirmatory_data_consumed"] is False
    assert source_invalid["challenge_materialized"] is False
    assert source_invalid["per_replicate"] == []
    assert any("source" in error.lower() for error in source_invalid["integrity_errors"])

    executor_invalid = _execute(fixture, executor_code_digest="8" * 64)
    assert executor_invalid["status"] == "INVALID_RUN"
    assert executor_invalid["decision"] == "INVALID_RUN"
    assert executor_invalid["challenge_materialized"] is False
    assert executor_invalid["per_replicate"] == []
    assert any("executor" in error.lower() for error in executor_invalid["integrity_errors"])


def test_exp279_executor_checkpoint_tamper_is_invalid_before_challenge_materialization(tmp_path) -> None:
    from nolane_ai.experiments.exp279_checkpoint import _receipt_digest

    fixture = _fixture(tmp_path)
    bad = deepcopy(fixture["checkpoint_receipt"])
    bad["scientific_identity_digest"] = "7" * 64
    bad["receipt_digest"] = _receipt_digest(bad)

    invalid = _execute(fixture, checkpoint_receipt=bad)
    assert invalid["status"] == "INVALID_RUN"
    assert invalid["decision"] == "INVALID_RUN"
    assert invalid["confirmatory_data_consumed"] is False
    assert invalid["challenge_materialized"] is False
    assert invalid["per_replicate"] == []
    assert any("checkpoint" in error.lower() for error in invalid["integrity_errors"])


def test_exp279_executor_generates_rows_independently_from_reconstruction_builder(tmp_path, monkeypatch) -> None:
    import nolane_ai.experiments.exp279_confirmatory_executor as executor

    fixture = _fixture(tmp_path)

    def forbidden_reconstruction_builder(**_kwargs):
        raise AssertionError("executor must not generate evidence via reconstruction")

    monkeypatch.setattr(
        executor,
        "reconstruct_exp279_expected_row",
        forbidden_reconstruction_builder,
        raising=False,
    )
    raw = _execute(fixture)
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["per_replicate"]


def test_exp279_raw_validator_rejects_rehashed_reorder_and_semantic_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        _artifact_digest,
        validate_exp279_confirmatory_raw,
    )
    from nolane_ai.experiments.exp279_reconstruction_court import _row_digest

    fixture = _fixture(tmp_path)
    raw = _execute(fixture)

    reordered = deepcopy(raw)
    reordered["per_replicate"][0], reordered["per_replicate"][1] = (
        reordered["per_replicate"][1],
        reordered["per_replicate"][0],
    )
    reordered["artifact_digest"] = _artifact_digest(reordered)
    errors = validate_exp279_confirmatory_raw(
        reordered,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert any("replicate" in error.lower() or "lineage" in error.lower() for error in errors)

    tampered = deepcopy(raw)
    tampered["per_replicate"][0]["hybrid"]["verified_solution_rate"] = 0.123456
    tampered["per_replicate"][0]["row_digest"] = _row_digest(
        tampered["per_replicate"][0]
    )
    tampered["artifact_digest"] = _artifact_digest(tampered)
    errors = validate_exp279_confirmatory_raw(
        tampered,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert any("solution" in error.lower() or "metric" in error.lower() for error in errors)
