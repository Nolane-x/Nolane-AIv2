from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from tests.test_neural_arm_registry import _audit, _protocol_subset


def _exp277_pair_and_execution():
    from nolane_ai.experiments.exp277_paired_runner import run_exp277_paired_development
    from nolane_ai.experiments.matched_cbrf_arms import audit_matched_exp277_arm_pair, build_matched_exp277_arm_pair

    arcs, oracle = build_matched_exp277_arm_pair(d_model=8, hidden_size=6, target_parameters=5_000)
    pair_audit = audit_matched_exp277_arm_pair(arcs, oracle, timesteps=3, variables=4, constraints=2)
    execution = run_exp277_paired_development(
        root_seed="registry-exp277",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=3,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    return pair_audit, execution


def test_registry_accepts_exp277_pair_audit_but_keeps_confirmatory_gate_closed() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry

    pair_audit, _ = _exp277_pair_and_execution()
    artifact = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest="p" * 64,
        model_audit=_audit(),
        exp277_pair_audit=pair_audit,
    )
    exp277 = artifact["experiments"]["EXP-277"]
    assert exp277["development_match_status"] == "PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED"
    assert exp277["resource_match_evidence"]["parameter_match"] is True
    assert exp277["resource_match_evidence"]["functional_parameter_match"] is True
    assert exp277["resource_match_evidence"]["oracle_information_separation"] is True
    assert exp277["resource_match_evidence"]["compute_budget_closed"] is True
    assert exp277["arms"]["arcs_branch"]["implementation_status"] == "IMPLEMENTED"
    assert exp277["arms"]["oracle_cbrf"]["implementation_status"] == "IMPLEMENTED"
    assert exp277["arms"]["arcs_branch"]["implementation_id"] == "exp277_matched_arcs_dev_v1"
    assert exp277["arms"]["oracle_cbrf"]["implementation_id"] == "exp277_matched_oracle_cbrf_dev_v1"
    assert exp277["match_court"] == "BLOCKED"
    assert exp277["blockers"] == [
        "EXP-277: matched arms are not yet integrated into paired structure-dense evaluator lineage"
    ]
    assert validate_neural_arm_registry(artifact) == []


def test_registry_accepts_exp277_paired_execution_without_false_promotion() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry

    pair_audit, execution = _exp277_pair_and_execution()
    artifact = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest="p" * 64,
        model_audit=_audit(),
        exp277_pair_audit=pair_audit,
        exp277_execution_artifact=execution,
    )
    exp277 = artifact["experiments"]["EXP-277"]
    assert exp277["development_match_status"] == "PAIRED_STRUCTURE_DENSE_DEV_READY"
    assert exp277["paired_execution_evidence"]["artifact_digest"] == execution["artifact_digest"]
    assert exp277["paired_execution_evidence"]["code_digest"] == execution["code_digest"]
    assert exp277["paired_execution_evidence"]["training_replicates"] == 2
    assert exp277["paired_execution_evidence"]["evaluation_replicates"] == 3
    assert exp277["paired_execution_evidence"]["oracle_relative_utility_gain"] == pytest.approx(
        execution["evaluation"]["aggregate"]["oracle_relative_utility_gain"]
    )
    assert exp277["match_court"] == "BLOCKED"
    assert exp277["blockers"] == [
        "EXP-277: confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution remain open"
    ]
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert validate_neural_arm_registry(artifact) == []


def test_registry_rejects_invalid_exp277_pair_audit_and_unpaired_execution() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    pair_audit, execution = _exp277_pair_and_execution()
    bad = deepcopy(pair_audit)
    bad["compute_budget_closed"] = False
    with pytest.raises(ValueError, match="EXP-277 matched pair audit"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp277_pair_audit=bad,
        )
    with pytest.raises(ValueError, match="requires matched pair audit"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp277_execution_artifact=execution,
        )


def test_registry_rejects_tampered_or_wrong_protocol_exp277_execution() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    pair_audit, execution = _exp277_pair_and_execution()
    tampered = deepcopy(execution)
    tampered["evaluation"]["per_replicate"].pop()
    with pytest.raises(ValueError, match="EXP-277 paired execution artifact"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp277_pair_audit=pair_audit,
            exp277_execution_artifact=tampered,
        )
    wrong_protocol = deepcopy(execution)
    wrong_protocol["protocol_digest"] = "q" * 64
    from nolane_ai.experiments.exp277_paired_runner import _artifact_digest
    wrong_protocol["artifact_digest"] = _artifact_digest(wrong_protocol)
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp277_pair_audit=pair_audit,
            exp277_execution_artifact=wrong_protocol,
        )
