from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

_support_path = Path(__file__).with_name("test_exp277_reconstruction_court.py")
_support_spec = importlib.util.spec_from_file_location("exp277_reconstruction_test_support", _support_path)
assert _support_spec is not None and _support_spec.loader is not None
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
court_fixture = _support.court_fixture


def _ensure_reconstruction(fixture) -> None:
    if "reconstruction" in fixture:
        return
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )


def _execute(
    fixture,
    *,
    beacon=None,
    source_tree_digest=None,
    executor_code_digest=None,
    checkpoint_receipt=None,
):
    from nolane_ai.experiments.exp277_confirmatory_executor import execute_exp277_confirmatory_challenge

    _ensure_reconstruction(fixture)
    return execute_exp277_confirmatory_challenge(
        reconstruction_authorization=fixture["reconstruction"],
        seal=fixture["seal"],
        beacon_receipt=beacon or fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=checkpoint_receipt or fixture["checkpoint_receipt"],
        current_source_tree_digest=source_tree_digest or fixture["reconstruction"]["source_tree_digest"],
        executor_code_digest=executor_code_digest or "3" * 64,
    )


def test_exp277_raw_executor_consumes_exact_reserved_lineage_without_parameter_writes(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import validate_exp277_confirmatory_raw
    from nolane_ai.protocol.identity import file_sha256

    fixture = court_fixture
    checkpoint_sha_before = file_sha256(fixture["checkpoint_path"])
    raw = _execute(fixture)
    assert raw["schema"] == "NLM-EXP-277-CONFIRMATORY-CHALLENGE-RAW-V1"
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["status"] == "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["test_only"] is False
    assert raw["scientific_evidence_eligible"] is True
    assert raw["confirmatory_data_consumed"] is True
    assert raw["synthetic_challenge_data_consumed"] is False
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


def test_exp277_raw_executor_records_exact_arm_inputs_and_analytical_costs(court_fixture) -> None:
    fixture = court_fixture
    raw = _execute(fixture)
    expected_flops = fixture["reconstruction"]["arm_accounted_flops_per_episode"]
    for row in raw["per_replicate"]:
        assert row["arm_input_receipt"] == {
            "same_surface_events": True,
            "same_variable_states": True,
            "arcs_received_oracle_incidence": False,
            "oracle_cbrf_received_oracle_incidence": True,
            "evaluator_targets_withheld_from_arms": True,
        }
        for arm in ("arcs_branch", "oracle_cbrf"):
            assert row[arm]["accounted_flops_per_episode"] == expected_flops[arm]
            receipt = row[arm]["analytical_cost_receipt"]
            assert receipt == {
                "accounting_semantics": "analytical scalar arithmetic FLOPs for frozen neural geometry; not hardware-profiler FLOPs",
                "accounted_flops_per_episode": expected_flops[arm],
                "hardware_profiler_flops_claimed": False,
            }


def test_exp277_test_only_executor_cannot_create_scientific_evidence(court_fixture) -> None:
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt

    fixture = court_fixture
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
    assert raw["seed_materialization_status"] == "TEST_ONLY_EXECUTED"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["decision_rule_executed"] is False


def test_exp277_executor_source_tree_or_executor_machinery_drift_is_invalid_run(court_fixture) -> None:
    fixture = court_fixture
    source_invalid = _execute(fixture, source_tree_digest="9" * 64)
    assert source_invalid["status"] == "INVALID_RUN"
    assert source_invalid["decision"] == "INVALID_RUN"
    assert source_invalid["confirmatory_data_consumed"] is False
    assert source_invalid["challenge_materialized"] is False
    assert source_invalid["per_replicate"] == []
    assert any("source tree" in error.lower() for error in source_invalid["integrity_errors"])

    executor_invalid = _execute(fixture, executor_code_digest="8" * 64)
    assert executor_invalid["status"] == "INVALID_RUN"
    assert executor_invalid["decision"] == "INVALID_RUN"
    assert executor_invalid["per_replicate"] == []
    assert any("executor" in error.lower() for error in executor_invalid["integrity_errors"])


def test_exp277_executor_checkpoint_tamper_is_invalid_run_before_challenge_execution(court_fixture) -> None:
    fixture = court_fixture
    bad = deepcopy(fixture["checkpoint_receipt"])
    bad["scientific_identity_digest"] = "7" * 64
    invalid = _execute(fixture, checkpoint_receipt=bad)
    assert invalid["status"] == "INVALID_RUN"
    assert invalid["decision"] == "INVALID_RUN"
    assert invalid["confirmatory_data_consumed"] is False
    assert invalid["challenge_materialized"] is False
    assert invalid["per_replicate"] == []
    assert any("checkpoint" in error.lower() for error in invalid["integrity_errors"])


def test_exp277_raw_validator_rejects_reordered_rows_after_rehash(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import _artifact_digest, validate_exp277_confirmatory_raw

    fixture = court_fixture
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
    from nolane_ai.experiments.exp277_reconstruction_court import _row_digest

    fixture = court_fixture
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
