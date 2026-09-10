from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
CODE_DIGEST = "a" * 64
FREEZE_SHA = "1" * 40
FREEZE_TIME = "2026-09-10T05:30:00Z"


def _api():
    try:
        from nolane_ai.experiments.exp286_confirmatory_ceremony import (
            seal_exp286_confirmatory_gate_a,
            validate_exp286_confirmatory_gate_a_seal,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 immutable Gate-A seal is missing")
    return seal_exp286_confirmatory_gate_a, validate_exp286_confirmatory_gate_a_seal


def _inputs() -> tuple[dict, dict, dict, str, str]:
    from nolane_ai.experiments.exp286_confirmatory_authorization import (
        authorize_exp286_confirmatory_execution,
    )
    from nolane_ai.experiments.exp286_confirmatory_prep import build_exp286_confirmatory_prep
    from nolane_ai.experiments.exp286_paired_runner import (
        _artifact_digest,
        run_exp286_paired_development,
    )
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from tests.test_neural_arm_registry import _audit, _protocol_subset

    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    protocol_digest = (ROOT / "protocols" / "stage_a_v1.sha256").read_text(encoding="utf-8").strip()
    geometry_digest = (ROOT / "protocols" / "exp286_development_geometry_v1.sha256").read_text(encoding="utf-8").strip()
    frozen_experiment = next(
        item for item in protocol["experiments"] if item["experiment_id"] == "EXP-286"
    )

    execution = run_exp286_paired_development(
        root_seed="exp286-gate-a-seal-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=32,
        eval_start_replicate=900,
        batch_size=2,
        timesteps=3,
        variables=5,
        decoys=2,
        max_search_steps=8,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest=protocol_digest,
        code_digest=CODE_DIGEST,
    )
    execution["development_geometry_authority"] = {
        "schema": "NLM-EXP-286-DEVELOPMENT-GEOMETRY-V1",
        "authority_scope": "DEVELOPMENT_PILOT_ONLY",
        "manifest_digest": geometry_digest,
        "confirmatory_authority": False,
    }
    execution["artifact_digest"] = _artifact_digest(execution)

    registry = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest=protocol_digest,
        model_audit=_audit(),
        exp286_pair_audit=execution["resource_match"]["pair_audit"],
        exp286_execution_artifact=execution,
    )
    prep = build_exp286_confirmatory_prep(
        experiment=frozen_experiment,
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=CODE_DIGEST,
        familywise_alpha=protocol["global_sample_size_plan"]["familywise_alpha"],
    )
    assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"

    authorization = authorize_exp286_confirmatory_execution(
        execution_artifact=execution,
        prep_artifact=prep,
        expected_geometry_digest=geometry_digest,
        execution_code_digest=CODE_DIGEST,
    )
    return execution, prep, authorization, protocol_digest, geometry_digest


@pytest.fixture(scope="module")
def sealed_inputs() -> tuple[dict, dict, dict, str, str]:
    return _inputs()


def test_exp286_gate_a_seal_binds_all_pre_beacon_authority_without_promoting_evidence(
    sealed_inputs: tuple[dict, dict, dict, str, str],
) -> None:
    seal_fn, validate = _api()
    execution, prep, authorization, protocol_digest, geometry_digest = sealed_inputs

    seal = seal_fn(
        protocol_digest=protocol_digest,
        development_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        expected_geometry_digest=geometry_digest,
        code_tree_digest=CODE_DIGEST,
        freeze_commit_sha=FREEZE_SHA,
        freeze_commit_timestamp_utc=FREEZE_TIME,
    )

    assert seal["schema"] == "NLM-EXP-286-CONFIRMATORY-GATE-A-SEAL-V1"
    assert seal["status"] == "CONFIRMATORY_GATE_A_SEALED"
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_ready"] is True
    assert seal["confirmatory_ready_scope"] == "FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["seed_materialization_status"] == "NOT_EXECUTED"
    assert seal["challenge_materialized"] is False
    assert seal["decision_rule_executed"] is False
    assert seal["freeze_commit_sha"] == FREEZE_SHA
    assert seal["freeze_commit_timestamp_utc"] == FREEZE_TIME
    assert seal["code_tree_digest"] == CODE_DIGEST
    assert seal["challenge_contract_digest"] == authorization["lineage"]["challenge_contract_digest"]
    assert seal["confirmatory_n"] == authorization["confirmatory_n"]
    assert seal["lineage"]["protocol_digest"] == protocol_digest
    assert seal["lineage"]["development_geometry_digest"] == geometry_digest
    assert seal["lineage"]["development_execution_digest"] == execution["artifact_digest"]
    assert seal["lineage"]["prep_digest"] == prep["prep_digest"]
    assert seal["lineage"]["execution_authorization_digest"] == authorization["authorization_digest"]
    assert "beacon" not in repr(seal).lower()
    assert "challenge_seed" not in repr(seal).lower()
    assert validate(seal) == []


def test_exp286_gate_a_seal_rejects_code_tree_drift(
    sealed_inputs: tuple[dict, dict, dict, str, str],
) -> None:
    seal_fn, _ = _api()
    execution, prep, authorization, protocol_digest, geometry_digest = sealed_inputs

    with pytest.raises(ValueError, match="code-tree closure"):
        seal_fn(
            protocol_digest=protocol_digest,
            development_execution_artifact=execution,
            prep_artifact=prep,
            execution_authorization=authorization,
            expected_geometry_digest=geometry_digest,
            code_tree_digest="b" * 64,
            freeze_commit_sha=FREEZE_SHA,
            freeze_commit_timestamp_utc=FREEZE_TIME,
        )


def test_exp286_gate_a_seal_validator_rejects_freeze_and_lineage_tampering(
    sealed_inputs: tuple[dict, dict, dict, str, str],
) -> None:
    seal_fn, validate = _api()
    execution, prep, authorization, protocol_digest, geometry_digest = sealed_inputs
    seal = seal_fn(
        protocol_digest=protocol_digest,
        development_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        expected_geometry_digest=geometry_digest,
        code_tree_digest=CODE_DIGEST,
        freeze_commit_sha=FREEZE_SHA,
        freeze_commit_timestamp_utc=FREEZE_TIME,
    )

    changed = deepcopy(seal)
    changed["freeze_commit_sha"] = "not-a-sha"
    assert any("freeze commit SHA" in item for item in validate(changed))

    changed = deepcopy(seal)
    changed["lineage"]["development_geometry_digest"] = "0" * 64
    assert any("geometry" in item.lower() or "digest" in item.lower() for item in validate(changed))


def test_exp286_gate_a_seal_changes_binding_when_freeze_lineage_changes(
    sealed_inputs: tuple[dict, dict, dict, str, str],
) -> None:
    seal_fn, _ = _api()
    execution, prep, authorization, protocol_digest, geometry_digest = sealed_inputs
    common = dict(
        protocol_digest=protocol_digest,
        development_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        expected_geometry_digest=geometry_digest,
        code_tree_digest=CODE_DIGEST,
        freeze_commit_timestamp_utc=FREEZE_TIME,
    )
    first = seal_fn(freeze_commit_sha="1" * 40, **common)
    second = seal_fn(freeze_commit_sha="2" * 40, **common)
    assert first["pre_beacon_binding_digest"] != second["pre_beacon_binding_digest"]
    assert first["seal_digest"] != second["seal_digest"]
