from __future__ import annotations

import pytest

from tests.test_exp286_confirmatory_authorization import GEOMETRY_DIGEST, _prepared


def _contract_api():
    try:
        from nolane_ai.experiments.exp286_challenge_worlds import (
            challenge_contract,
            challenge_contract_digest,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 post-freeze challenge contract authority is missing")
    return challenge_contract, challenge_contract_digest


def test_exp286_challenge_contract_matches_frozen_scientific_semantics() -> None:
    challenge_contract, challenge_contract_digest = _contract_api()
    contract = challenge_contract()

    assert contract["schema"] == "NLM-EXP-286-POST-FREEZE-CHALLENGE-CONTRACT-V1"
    assert contract["experiment_id"] == "EXP-286"
    assert contract["lane"] == "POST_FREEZE_CHALLENGE"
    assert contract["arms"] == ["chronological_failure", "oracle_conflict_core"]
    assert contract["primary_endpoint"] == "accounted_reasoning_flops_to_verified_solution"
    assert contract["direction"] == "lower"
    assert contract["paired_design"] is True
    assert contract["candidate_oracle_conflict_labels_visible"] is False
    assert contract["control_oracle_conflict_labels_visible"] is True
    assert contract["oracle_labeling_flops_charged_online"] is True
    assert contract["seed_authority"] == "future_public_beacon_after_freeze"
    assert len(challenge_contract_digest()) == 64


def test_exp286_authorization_binds_challenge_contract_before_beacon() -> None:
    _, challenge_contract_digest = _contract_api()
    from nolane_ai.experiments.exp286_confirmatory_authorization import (
        authorize_exp286_confirmatory_execution,
        validate_exp286_confirmatory_execution_authorization,
    )

    execution, prep = _prepared(authoritative=True)
    authorization = authorize_exp286_confirmatory_execution(
        execution_artifact=execution,
        prep_artifact=prep,
        expected_geometry_digest=GEOMETRY_DIGEST,
        execution_code_digest="a" * 64,
    )

    assert authorization["lineage"]["challenge_contract_digest"] == challenge_contract_digest()
    assert authorization["preflight"]["challenge_contract_bound"] is True
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["challenge_materialized"] is False
    assert "beacon" not in authorization
    assert "challenge_seed" not in authorization
    assert "seeds" not in authorization
    assert validate_exp286_confirmatory_execution_authorization(authorization) == []


def test_exp286_authorization_validator_rejects_missing_challenge_contract_binding() -> None:
    from nolane_ai.experiments.exp286_confirmatory_authorization import (
        authorize_exp286_confirmatory_execution,
        validate_exp286_confirmatory_execution_authorization,
    )

    execution, prep = _prepared(authoritative=True)
    authorization = authorize_exp286_confirmatory_execution(
        execution_artifact=execution,
        prep_artifact=prep,
        expected_geometry_digest=GEOMETRY_DIGEST,
        execution_code_digest="a" * 64,
    )
    authorization["lineage"].pop("challenge_contract_digest", None)
    authorization["preflight"].pop("challenge_contract_bound", None)
    authorization["authorization_digest"] = ""

    errors = validate_exp286_confirmatory_execution_authorization(authorization)
    assert any("challenge_contract_digest" in item for item in errors)
    assert any("challenge_contract_bound" in item for item in errors)
