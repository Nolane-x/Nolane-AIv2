import copy

import pytest

torch = pytest.importorskip("torch")


def _paired_execution(tmp_path):
    from nolane_ai.experiments.exp282_paired_runner import run_exp282_paired_development

    return run_exp282_paired_development(
        root_seed="reconstruct",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=2,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=2,
        visibility_rate=0.5,
        noise_std=0.25,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
        checkpoint_path=tmp_path / "paired.pt",
    )


def _authorization(execution):
    from nolane_ai.protocol.evidence import canonical_sha256

    payload = {
        "schema": "NLM-EXP-282-CONFIRMATORY-EXECUTION-AUTH-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "AUTHORIZED_NOT_EXECUTED",
        "scope": "exp282-confirmatory-open-execution-authorization-only",
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "confirmatory_n": 32,
        "reserved_replicate_ids": list(range(200, 232)),
        "frozen_analysis_digest": "f" * 64,
        "sample_size_freeze_digest": "s" * 64,
        "lineage": {
            "prep_digest": "q" * 64,
            "protocol_digest": execution["protocol_digest"],
            "analysis_code_digest": "a" * 64,
            "execution_code_digest": "x" * 64,
            "paired_checkpoint_sha256": execution["checkpoint"]["checkpoint_sha256"],
            "paired_execution_contract_digest": execution["checkpoint"]["execution_contract_digest"],
        },
        "preflight": {
            "prep_semantically_valid": True,
            "prep_status_prepared": True,
            "confirmatory_ids_reserved": True,
            "development_lineages_disjoint": True,
            "confirmatory_data_unconsumed": True,
            "seeds_unmaterialized": True,
            "all_checks_passed": True,
        },
        "remaining_blockers": [
            "confirmatory observations have not been executed",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "authorization_digest": "",
    }
    payload["authorization_digest"] = canonical_sha256({k: v for k, v in payload.items() if k != "authorization_digest"})
    return payload


def test_reconstruction_court_binds_exact_checkpoint_and_execution_contract(tmp_path):
    from nolane_ai.experiments.exp282_reconstruction_court import (
        authorize_exp282_confirmatory_reconstruction,
        validate_exp282_confirmatory_reconstruction,
    )

    execution = _paired_execution(tmp_path)
    auth = _authorization(execution)
    result = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=auth,
    )

    assert result["schema"] == "NLM-EXP-282-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
    assert result["evidence_level"] == "EV-E2"
    assert result["decision"] == "UNVERIFIED"
    assert result["status"] == "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
    assert result["confirmatory_data_consumed"] is False
    assert result["seed_materialization_status"] == "NOT_EXECUTED"
    assert "seeds" not in result
    assert result["confirmatory_n"] == 32
    assert result["reserved_replicate_ids"] == list(range(200, 232))
    assert result["checkpoint_sha256"] == execution["checkpoint"]["checkpoint_sha256"]
    assert result["execution_contract"] == execution["checkpoint"]["execution_contract"]
    assert result["execution_contract_digest"] == execution["checkpoint"]["execution_contract_digest"]
    assert result["lineage"]["execution_authorization_digest"] == auth["authorization_digest"]
    assert validate_exp282_confirmatory_reconstruction(result) == []


def test_reconstruction_court_rejects_checkpoint_or_contract_mismatch(tmp_path):
    from nolane_ai.experiments.exp282_reconstruction_court import authorize_exp282_confirmatory_reconstruction

    execution = _paired_execution(tmp_path)
    auth = _authorization(execution)
    auth["lineage"]["paired_checkpoint_sha256"] = "0" * 64
    from nolane_ai.protocol.evidence import canonical_sha256
    auth["authorization_digest"] = canonical_sha256({k: v for k, v in auth.items() if k != "authorization_digest"})
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        authorize_exp282_confirmatory_reconstruction(
            execution_artifact=execution,
            execution_authorization=auth,
        )

    execution = _paired_execution(tmp_path)
    auth = _authorization(execution)
    execution["checkpoint"]["execution_contract"]["arm_geometry"]["hidden_size"] += 1
    with pytest.raises(ValueError, match="paired development artifact"):
        authorize_exp282_confirmatory_reconstruction(
            execution_artifact=execution,
            execution_authorization=auth,
        )


def test_reconstruction_validator_rejects_rehashed_semantic_tamper(tmp_path):
    from nolane_ai.experiments.exp282_reconstruction_court import (
        _reconstruction_digest,
        authorize_exp282_confirmatory_reconstruction,
        validate_exp282_confirmatory_reconstruction,
    )

    execution = _paired_execution(tmp_path)
    result = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=_authorization(execution),
    )
    result["execution_contract"]["world_geometry"]["noise_std"] = 99.0
    result["reconstruction_digest"] = _reconstruction_digest(result)
    errors = validate_exp282_confirmatory_reconstruction(result)
    assert "execution contract digest mismatch" in errors


def test_reconstruction_validator_rejects_reserved_lineage_tamper_even_after_rehash(tmp_path):
    from nolane_ai.experiments.exp282_reconstruction_court import (
        _reconstruction_digest,
        authorize_exp282_confirmatory_reconstruction,
        validate_exp282_confirmatory_reconstruction,
    )

    execution = _paired_execution(tmp_path)
    result = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=_authorization(execution),
    )
    result["reserved_replicate_ids"].pop()
    result["reconstruction_digest"] = _reconstruction_digest(result)
    errors = validate_exp282_confirmatory_reconstruction(result)
    assert "reconstruction reserved replicate count mismatch" in errors
