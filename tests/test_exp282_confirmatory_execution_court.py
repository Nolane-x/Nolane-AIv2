import pytest


def _prepared_prep():
    from nolane_ai.experiments.exp282_confirmatory_prep import _prep_digest

    payload = {
        "schema": "NLM-EXP-282-CONFIRMATORY-OPEN-PREP-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "CONFIRMATORY_OPEN_PREPARED",
        "scope": "exp282-confirmatory-open-preparation-only",
        "confirmatory_data_consumed": False,
        "frozen_analysis": {
            "primary_endpoint": "grounded_decision_accuracy",
            "primary_direction": "higher",
            "mesi_absolute_gain": 0.03,
            "power_target": 0.90,
            "familywise_alpha": 0.05,
            "inference_tail": "one_sided_lower_bound",
            "analysis_method": "paired accuracy difference with bootstrap CI plus calibration guard",
            "multiplicity_family": "BELIEF_STATE",
            "brier_guard": "explicit_belief <= recurrent_hidden + 0.02",
            "compute_guard": "difference <= 0.05 relative unless included in primary cost normalization",
        },
        "pilot_summary": {
            "n": 32,
            "mean_accuracy_gain": 0.03,
            "paired_sd": 0.01,
            "paired_effect_digest": "e" * 64,
            "replicate_start": 1000,
            "replicate_end": 1031,
            "replicate_ids": list(range(1000, 1032)),
            "training_replicate_ids": [0, 1, 2],
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
        },
        "sample_size_freeze": {
            "method": "paired-normal-approximation-from-pilot-sd-v1",
            "unclamped_required_n": 1,
            "confirmatory_n": 32,
            "min_n": 32,
            "max_n": 128,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "planning_constants": {
                "alpha_tail": "one_sided_lower_bound",
                "z_alpha": 1.6448536269514715,
                "z_power": 1.2815515655446008,
                "formula": "ceil(((z_alpha + z_power) * paired_sd / mesi)^2), then apply frozen min_n without max_n truncation",
            },
        },
        "confirmatory_lineage": {
            "lane": "CONFIRMATORY_OPEN_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "reserved_replicate_ids": list(range(1032, 1064)),
            "seed_materialization_status": "NOT_EXECUTED",
        },
        "lineage": {
            "protocol_digest": "p" * 64,
            "execution_artifact_digest": "a" * 64,
            "arm_registry_digest": "r" * 64,
            "analysis_code_digest": "d" * 64,
            "paired_checkpoint_sha256": "k" * 64,
        },
        "remaining_blockers": [
            "confirmatory-open execution has not consumed any reserved confirmatory data",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "prep_digest": "",
    }
    payload["prep_digest"] = _prep_digest(payload)
    return payload


def test_execution_court_authorizes_only_without_executing_or_materializing_seeds():
    from nolane_ai.experiments.exp282_confirmatory_execution_court import (
        authorize_exp282_confirmatory_execution,
        validate_exp282_confirmatory_execution_authorization,
    )

    auth = authorize_exp282_confirmatory_execution(
        prep_artifact=_prepared_prep(),
        execution_code_digest="x" * 64,
    )
    assert auth["schema"] == "NLM-EXP-282-CONFIRMATORY-EXECUTION-AUTH-V1"
    assert auth["evidence_level"] == "EV-E2"
    assert auth["decision"] == "UNVERIFIED"
    assert auth["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert auth["confirmatory_data_consumed"] is False
    assert auth["seed_materialization_status"] == "NOT_EXECUTED"
    assert auth["confirmatory_n"] == 32
    assert auth["reserved_replicate_ids"] == list(range(1032, 1064))
    assert "seeds" not in auth
    assert auth["preflight"]["all_checks_passed"] is True
    assert auth["lineage"]["paired_checkpoint_sha256"] == "k" * 64
    assert validate_exp282_confirmatory_execution_authorization(auth) == []


def test_execution_court_hard_blocks_not_ready_prep():
    from nolane_ai.experiments.exp282_confirmatory_execution_court import authorize_exp282_confirmatory_execution
    from nolane_ai.experiments.exp282_confirmatory_prep import _prep_digest

    prep = _prepared_prep()
    prep["status"] = "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
    prep["pilot_summary"]["paired_sd"] = 0.116
    prep["sample_size_freeze"]["unclamped_required_n"] = 129
    prep["sample_size_freeze"]["confirmatory_n"] = None
    prep["confirmatory_lineage"]["reserved_replicate_ids"] = []
    prep["prep_digest"] = _prep_digest(prep)
    with pytest.raises(ValueError, match="not CONFIRMATORY_OPEN_PREPARED"):
        authorize_exp282_confirmatory_execution(prep_artifact=prep, execution_code_digest="x" * 64)


def test_execution_court_rejects_semantically_invalid_prep_even_if_rehashed():
    from nolane_ai.experiments.exp282_confirmatory_execution_court import authorize_exp282_confirmatory_execution
    from nolane_ai.experiments.exp282_confirmatory_prep import _prep_digest

    prep = _prepared_prep()
    prep["sample_size_freeze"]["confirmatory_n"] = 33
    prep["prep_digest"] = _prep_digest(prep)
    with pytest.raises(ValueError, match="invalid confirmatory prep"):
        authorize_exp282_confirmatory_execution(prep_artifact=prep, execution_code_digest="x" * 64)


def test_execution_authorization_validator_rejects_manual_execution_claim_and_tamper():
    from nolane_ai.experiments.exp282_confirmatory_execution_court import (
        _authorization_digest,
        authorize_exp282_confirmatory_execution,
        validate_exp282_confirmatory_execution_authorization,
    )

    auth = authorize_exp282_confirmatory_execution(
        prep_artifact=_prepared_prep(),
        execution_code_digest="x" * 64,
    )
    auth["confirmatory_data_consumed"] = True
    auth["seed_materialization_status"] = "EXECUTED"
    auth["status"] = "EXECUTED"
    auth["authorization_digest"] = _authorization_digest(auth)
    errors = validate_exp282_confirmatory_execution_authorization(auth)
    assert "execution authorization cannot consume confirmatory data" in errors
    assert "execution authorization cannot materialize confirmatory seeds" in errors
    assert "execution authorization status drift" in errors
