from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_DIGEST = (ROOT / "protocols" / "exp286_development_geometry_v1.sha256").read_text(encoding="utf-8").strip()


def _protocol() -> dict:
    return json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))


def _frozen_experiment() -> dict:
    return next(item for item in _protocol()["experiments"] if item["experiment_id"] == "EXP-286")


def _prepared(*, authoritative: bool) -> tuple[dict, dict]:
    from nolane_ai.experiments.exp286_confirmatory_prep import build_exp286_confirmatory_prep
    from nolane_ai.experiments.exp286_paired_runner import _artifact_digest, run_exp286_paired_development
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from tests.test_neural_arm_registry import _audit, _protocol_subset

    execution = run_exp286_paired_development(
        root_seed="exp286-authorization-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=32,
        eval_start_replicate=500,
        batch_size=2,
        timesteps=3,
        variables=5,
        decoys=2,
        max_search_steps=8,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    if authoritative:
        execution["development_geometry_authority"] = {
            "schema": "NLM-EXP-286-DEVELOPMENT-GEOMETRY-V1",
            "authority_scope": "DEVELOPMENT_PILOT_ONLY",
            "manifest_digest": GEOMETRY_DIGEST,
            "confirmatory_authority": False,
        }
        execution["artifact_digest"] = _artifact_digest(execution)

    registry = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest=execution["protocol_digest"],
        model_audit=_audit(),
        exp286_pair_audit=execution["resource_match"]["pair_audit"],
        exp286_execution_artifact=execution,
    )
    prep = build_exp286_confirmatory_prep(
        experiment=_frozen_experiment(),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest="a" * 64,
        familywise_alpha=_protocol()["global_sample_size_plan"]["familywise_alpha"],
    )
    assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"
    return execution, prep


def test_exp286_authorization_rejects_smoke_gate_a_even_when_prep_is_ready() -> None:
    from nolane_ai.experiments.exp286_confirmatory_authorization import authorize_exp286_confirmatory_execution

    execution, prep = _prepared(authoritative=False)
    with pytest.raises(ValueError, match="authoritative DEVELOPMENT geometry"):
        authorize_exp286_confirmatory_execution(
            execution_artifact=execution,
            prep_artifact=prep,
            expected_geometry_digest=GEOMETRY_DIGEST,
            execution_code_digest="a" * 64,
        )


def test_exp286_authorization_rejects_execution_code_lineage_drift() -> None:
    from nolane_ai.experiments.exp286_confirmatory_authorization import authorize_exp286_confirmatory_execution

    execution, prep = _prepared(authoritative=True)
    with pytest.raises(ValueError, match="authorization code lineage mismatch"):
        authorize_exp286_confirmatory_execution(
            execution_artifact=execution,
            prep_artifact=prep,
            expected_geometry_digest=GEOMETRY_DIGEST,
            execution_code_digest="b" * 64,
        )


def test_exp286_authorization_closes_geometry_and_prep_lineage_without_materializing_challenge() -> None:
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

    assert authorization["schema"] == "NLM-EXP-286-CONFIRMATORY-EXECUTION-AUTH-V1"
    assert authorization["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert authorization["evidence_level"] == "EV-E2"
    assert authorization["decision"] == "UNVERIFIED"
    assert authorization["confirmatory_data_consumed"] is False
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["challenge_materialized"] is False
    assert authorization["confirmatory_n"] == 32
    assert authorization["lineage"]["development_geometry_digest"] == GEOMETRY_DIGEST
    assert authorization["lineage"]["development_execution_digest"] == execution["artifact_digest"]
    assert authorization["lineage"]["prep_digest"] == prep["prep_digest"]
    assert authorization["preflight"]["authoritative_geometry_bound"] is True
    assert authorization["preflight"]["all_checks_passed"] is True
    assert "seeds" not in authorization
    assert validate_exp286_confirmatory_execution_authorization(authorization) == []
