from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("torch")

from tests.exp289_confirmatory_fixtures import CODE_DIGEST, task5_inputs
from tests.test_exp289_confirmatory_authorization import _authorize
from tests.test_exp289_confirmatory_ceremony import _seal


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _beacon(seal: dict) -> dict:
    from nolane_ai.experiments.exp289_beacon import build_test_beacon_receipt

    published = (
        _parse_utc(seal["seal_created_at_utc"]) + timedelta(seconds=1)
    ).isoformat().replace("+00:00", "Z")
    return build_test_beacon_receipt(
        source="TEST-ONLY EXP-289 Gate-B beacon",
        beacon_id="exp289-test-gate-b-1",
        published_at_utc=published,
        entropy_hex="ab" * 32,
        evidence_reference="test-only://exp289/gate-b",
    )


def _executor_api():
    try:
        from nolane_ai.experiments.exp289_confirmatory_executor import (
            execute_exp289_confirmatory_challenge,
            validate_exp289_confirmatory_raw,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-289 raw challenge executor/reconstruction firewall is missing")
    return execute_exp289_confirmatory_challenge, validate_exp289_confirmatory_raw


def _analysis_api():
    try:
        from nolane_ai.experiments.exp289_confirmatory_analysis import (
            bootstrap_exp289_relative_rder,
            build_exp289_confirmatory_analysis,
            decide_exp289_confirmatory_outcome,
            validate_exp289_confirmatory_analysis,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-289 Gate-B analysis firewall is missing")
    return (
        bootstrap_exp289_relative_rder,
        build_exp289_confirmatory_analysis,
        decide_exp289_confirmatory_outcome,
        validate_exp289_confirmatory_analysis,
    )


def test_exp289_reserved_replicate_ids_are_transitively_frozen_before_beacon(task5_inputs: dict) -> None:
    reserved = task5_inputs["prep"]["confirmatory_lineage"]["reserved_replicate_ids"]
    authorization = _authorize(task5_inputs)
    assert authorization["reserved_replicate_ids"] == reserved
    assert len(reserved) == authorization["confirmatory_n"]
    seal = _seal(task5_inputs, authorization)
    assert seal["reserved_replicate_ids"] == reserved
    assert seal["execution_authorization"]["reserved_replicate_ids"] == reserved


@pytest.fixture(scope="module")
def task6_run(task5_inputs: dict) -> dict:
    execute, validate_raw = _executor_api()
    authorization = _authorize(task5_inputs)
    seal = _seal(task5_inputs, authorization)
    beacon = _beacon(seal)
    raw = execute(
        seal=seal,
        beacon_receipt=beacon,
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
        current_source_tree_digest=CODE_DIGEST,
        executor_code_digest=CODE_DIGEST,
    )
    assert validate_raw(
        raw,
        seal=seal,
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
    ) == []
    return {"authorization": authorization, "seal": seal, "beacon": beacon, "raw": raw}


def test_exp289_test_only_executor_materializes_only_reserved_raw_rows_without_promotion(task5_inputs: dict, task6_run: dict) -> None:
    raw = task6_run["raw"]
    seal = task6_run["seal"]
    assert raw["schema"] == "NLM-EXP-289-CONFIRMATORY-CHALLENGE-RAW-V1"
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["test_only"] is True
    assert raw["scientific_evidence_eligible"] is False
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["confirmatory_data_consumed"] is False
    assert raw["synthetic_challenge_data_consumed"] is True
    assert raw["challenge_materialized"] is True
    assert raw["seed_materialization_status"] == "TEST_ONLY_EXECUTED"
    assert raw["decision_rule_executed"] is False
    assert raw["confirmatory_n"] == seal["confirmatory_n"] == 32
    assert raw["reserved_replicate_ids"] == seal["reserved_replicate_ids"]
    assert [row["replicate"] for row in raw["per_replicate"]] == seal["reserved_replicate_ids"]
    assert len(raw["per_replicate"]) == 32
    assert raw["seal_digest"] == seal["seal_digest"]
    assert raw["checkpoint_scientific_identity_digest"] == task5_inputs["checkpoint_receipt"]["scientific_identity_digest"]
    assert all(isinstance(row["challenge_seed"], int) and row["challenge_seed"] >= 0 for row in raw["per_replicate"])
    assert all(row["challenge_batch_digest"] for row in raw["per_replicate"])
    assert all(row["row_digest"] for row in raw["per_replicate"])


def test_exp289_raw_validator_reconstructs_and_rejects_rehashed_semantic_tamper(task5_inputs: dict, task6_run: dict) -> None:
    from nolane_ai.experiments.exp289_confirmatory_executor import _artifact_digest, _row_digest

    _, validate_raw = _executor_api()
    forged = deepcopy(task6_run["raw"])
    forged["per_replicate"][0]["no_nogood"]["repeat_dead_end_rate"] = 0.125
    forged["per_replicate"][0]["row_digest"] = _row_digest(forged["per_replicate"][0])
    forged["artifact_digest"] = _artifact_digest(forged)
    errors = validate_raw(
        forged,
        seal=task6_run["seal"],
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
    )
    assert errors
    assert any("reconstruct" in item.lower() or "semantic" in item.lower() for item in errors)


def test_exp289_relative_rder_bootstrap_is_deterministic_and_forbids_epsilon_rescue() -> None:
    bootstrap, _, _, _ = _analysis_api()
    first = bootstrap(
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        seed_material="exp289-bootstrap-contract",
        samples=500,
        alpha=0.05,
    )
    second = bootstrap(
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        seed_material="exp289-bootstrap-contract",
        samples=500,
        alpha=0.05,
    )
    assert first == second
    assert first["observed_relative_reduction"] == pytest.approx(0.5)
    assert first["one_sided_lower"] == pytest.approx(0.5)
    assert first["one_sided_upper"] == pytest.approx(0.5)
    assert first["invalid_denominator_resamples"] == 0
    assert first["denominator_policy"] == "strictly_positive_no_epsilon"
    with pytest.raises(ValueError, match="denominator|epsilon"):
        bootstrap(
            [0.0, 1.0],
            [0.0, 0.5],
            seed_material="exp289-invalid-denominator",
            samples=100,
            alpha=0.05,
        )


def test_exp289_gate_b_decision_rule_locks_mesi_and_safety_guards() -> None:
    _, _, decide, _ = _analysis_api()
    assert decide(primary_lower=0.25, primary_upper=0.40, overprune_rate=0.005, solution_lower=-0.01, solution_upper=0.02, denominator_valid=True) == "PROMOTE_TO_NEXT_STAGE"
    assert decide(primary_lower=0.10, primary_upper=0.249, overprune_rate=0.0, solution_lower=0.0, solution_upper=0.01, denominator_valid=True) == "KILL_SUBSYSTEM"
    assert decide(primary_lower=0.40, primary_upper=0.60, overprune_rate=0.0051, solution_lower=0.0, solution_upper=0.01, denominator_valid=True) == "KILL_SUBSYSTEM"
    assert decide(primary_lower=0.40, primary_upper=0.60, overprune_rate=0.0, solution_lower=-0.03, solution_upper=-0.011, denominator_valid=True) == "KILL_SUBSYSTEM"
    assert decide(primary_lower=0.20, primary_upper=0.30, overprune_rate=0.0, solution_lower=-0.005, solution_upper=0.01, denominator_valid=True) == "HOLD_UNSTABLE"
    assert decide(primary_lower=0.40, primary_upper=0.60, overprune_rate=0.0, solution_lower=0.0, solution_upper=0.01, denominator_valid=False) == "HOLD_UNSTABLE"


def test_exp289_test_only_analysis_executes_would_be_rule_but_stays_ev_e2(task5_inputs: dict, task6_run: dict) -> None:
    _, build, _, validate = _analysis_api()
    analysis = build(
        raw_artifact=task6_run["raw"],
        seal=task6_run["seal"],
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
        analysis_code_digest=CODE_DIGEST,
    )
    assert analysis["schema"] == "NLM-EXP-289-CONFIRMATORY-ANALYSIS-V1"
    assert analysis["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert analysis["test_only"] is True
    assert analysis["scientific_evidence_eligible"] is False
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["confirmatory_data_consumed"] is False
    assert analysis["synthetic_challenge_data_consumed"] is True
    assert analysis["decision_rule_executed"] is False
    assert analysis["test_only_decision_executed"] is True
    assert analysis["test_only_would_be_decision"] in {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}
    primary = analysis["primary_endpoint"]
    assert primary["metric"] == "repeat_dead_end_rate"
    assert primary["effect_type"] == "paired_relative_rder_reduction"
    assert primary["mesi_relative_reduction"] == 0.25
    assert primary["alpha"] == 0.05
    assert primary["samples"] == 10_000
    assert primary["denominator_policy"] == "strictly_positive_no_epsilon"
    assert analysis["protected_overprune"]["ceiling"] == 0.005
    assert analysis["protected_solution_rate"]["floor_difference"] == -0.01
    assert validate(
        analysis,
        raw_artifact=task6_run["raw"],
        seal=task6_run["seal"],
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
    ) == []


def test_exp289_analysis_rejects_rehashed_raw_semantic_tamper(task5_inputs: dict, task6_run: dict) -> None:
    from nolane_ai.experiments.exp289_confirmatory_executor import _artifact_digest, _row_digest

    _, build, _, _ = _analysis_api()
    forged = deepcopy(task6_run["raw"])
    forged["per_replicate"][0]["local_nogood"]["verified_solution_rate"] = 0.25
    forged["per_replicate"][0]["row_digest"] = _row_digest(forged["per_replicate"][0])
    forged["artifact_digest"] = _artifact_digest(forged)
    with pytest.raises(ValueError, match="raw|reconstruct|semantic"):
        build(
            raw_artifact=forged,
            seal=task6_run["seal"],
            checkpoint_path=task5_inputs["checkpoint_path"],
            checkpoint_receipt=task5_inputs["checkpoint_receipt"],
            analysis_code_digest=CODE_DIGEST,
        )
