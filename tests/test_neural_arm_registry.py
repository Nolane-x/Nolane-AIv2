import copy

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel


def _protocol_subset():
    return {
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "status": "FROZEN_V1",
        "experiments": [
            {
                "experiment_id": "EXP-277",
                "arms": [
                    {"id": "arcs_branch", "description": "V0.15 ARCS recurrent-depth + branch bank + verifier court"},
                    {"id": "oracle_cbrf", "description": "same substrate with ground-truth constraint/factor representation"},
                ],
                "resource_match": {"parameter_budget": "matched active parameter count", "inference_budget": "same max accounted FLOPs per episode", "world_pairing": "same world lineage and replicate index"},
            },
            {
                "experiment_id": "EXP-279",
                "arms": [
                    {"id": "propagation_only", "description": "constraint propagation without branch search"},
                    {"id": "branch_only", "description": "ARCS branch search without CBRF propagation"},
                    {"id": "hybrid", "description": "propagation followed by branch search when residual uncertainty remains"},
                ],
                "resource_match": {"parameter_budget": "reclaimed parameters assigned to simpler rivals", "inference_budget": "equal max accounted FLOPs", "structure_fit": "predeclared strata"},
            },
            {
                "experiment_id": "EXP-282",
                "arms": [
                    {"id": "recurrent_hidden", "description": "matched recurrent state without explicit belief representation"},
                    {"id": "explicit_belief", "description": "explicit calibrated belief state over hidden world variables"},
                ],
                "resource_match": {"parameter_budget": "equal state/controller parameters", "observation_history": "identical", "compute_budget": "matched"},
            },
        ],
    }


def _audit():
    return audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))


def test_registry_matches_frozen_arm_ids_and_refuses_false_readiness():
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry
    artifact = build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p" * 64, model_audit=_audit())
    assert artifact["schema"] == "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["pilot_total_parameters"] == 16_000_000
    assert set(artifact["experiments"]) == {"EXP-277", "EXP-279", "EXP-282"}
    assert all(item["match_court"] == "BLOCKED" for item in artifact["experiments"].values())
    assert artifact["experiments"]["EXP-282"]["arms"]["explicit_belief"]["implementation_status"] == "DEVELOPMENT_COMPONENT_PRESENT"
    assert artifact["experiments"]["EXP-282"]["arms"]["recurrent_hidden"]["implementation_status"] == "BLOCKED"
    assert validate_neural_arm_registry(artifact) == []


def test_registry_detects_protocol_arm_drift():
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    protocol = _protocol_subset()
    protocol["experiments"][0]["arms"][0]["id"] = "arcs_proxy"
    with pytest.raises(ValueError, match="protocol arm drift"):
        build_neural_arm_registry(protocol=protocol, protocol_digest="p", model_audit=_audit())


def test_registry_records_resource_mismatch_and_region_functional_counts():
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    artifact = build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p", model_audit=_audit())
    exp282 = artifact["experiments"]["EXP-282"]
    assert exp282["resource_match_contract"]["parameter_budget"] == "equal state/controller parameters"
    counts = artifact["region_functional_parameters"]
    assert counts["recurrent_deliberation_core"] == 395_265
    assert counts["constraint_belief_fabric"] == 198_403
    assert any("matched" in blocker.lower() for blocker in exp282["blockers"])


def test_registry_validator_rejects_manual_promotion_or_digest_tamper():
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry
    artifact = build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p", model_audit=_audit())
    artifact["experiments"]["EXP-282"]["match_court"] = "CONFIRMATORY_READY"
    artifact["decision"] = "PROMOTE_TO_NEXT_STAGE"
    errors = validate_neural_arm_registry(artifact)
    assert "neural arm registry cannot promote a scientific claim" in errors
    assert "EXP-282 cannot be confirmatory-ready while blockers remain" in errors
    assert "neural arm registry digest mismatch" in errors


def test_registry_accepts_exp282_matched_pair_evidence_without_false_confirmatory_readiness():
    from nolane_ai.experiments.matched_belief_arms import audit_matched_belief_arm_pair, build_matched_belief_arm_pair
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry
    recurrent, explicit = build_matched_belief_arm_pair(d_model=16, hidden_size=12, target_parameters=10_000)
    pair_audit = audit_matched_belief_arm_pair(recurrent, explicit, timesteps=5, variables=4)
    artifact = build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p" * 64, model_audit=_audit(), exp282_pair_audit=pair_audit)
    exp282 = artifact["experiments"]["EXP-282"]
    assert exp282["development_match_status"] == "PARAMETER_AND_COMPUTE_MATCH_CLOSED"
    assert exp282["resource_match_evidence"]["parameter_match"] is True
    assert exp282["resource_match_evidence"]["observation_history_match"] is True
    assert exp282["resource_match_evidence"]["accounted_flop_match"] is True
    assert exp282["resource_match_evidence"]["relative_accounted_flop_difference"] == pytest.approx(0.0)
    assert exp282["arms"]["recurrent_hidden"]["implementation_status"] == "IMPLEMENTED"
    assert exp282["arms"]["explicit_belief"]["implementation_status"] == "IMPLEMENTED"
    assert exp282["match_court"] == "BLOCKED"
    assert exp282["blockers"] == ["EXP-282: matched arms are not yet integrated into paired partial-observability checkpoint/evaluator lineage"]
    assert validate_neural_arm_registry(artifact) == []


def test_registry_rejects_exp282_pair_audit_that_does_not_close_compute_match():
    from nolane_ai.experiments.matched_belief_arms import audit_matched_belief_arm_pair, build_matched_belief_arm_pair
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    recurrent, explicit = build_matched_belief_arm_pair(d_model=16, hidden_size=12, target_parameters=10_000)
    pair_audit = audit_matched_belief_arm_pair(recurrent, explicit, timesteps=3, variables=2)
    pair_audit["full_accounted_flop_match"] = False
    with pytest.raises(ValueError, match="EXP-282 matched pair audit"):
        build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p", model_audit=_audit(), exp282_pair_audit=pair_audit)


def _exp282_pair_and_execution():
    from nolane_ai.experiments.exp282_paired_runner import run_exp282_paired_development
    from nolane_ai.experiments.matched_belief_arms import audit_matched_belief_arm_pair, build_matched_belief_arm_pair
    recurrent, explicit = build_matched_belief_arm_pair(d_model=8, hidden_size=6, target_parameters=5_000)
    pair_audit = audit_matched_belief_arm_pair(recurrent, explicit, timesteps=4, variables=3)
    execution = run_exp282_paired_development(root_seed="registry-execution", d_model=8, hidden_size=6, target_parameters=5_000, train_replicates=2, eval_replicates=3, eval_start_replicate=100, batch_size=2, timesteps=4, variables=3, visibility_rate=0.5, noise_std=0.3, lr=1e-3, weight_decay=0.0, protocol_digest="p" * 64, code_digest="c" * 64)
    return pair_audit, execution


def test_registry_accepts_paired_exp282_execution_but_keeps_confirmatory_gate_closed():
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry
    pair_audit, execution = _exp282_pair_and_execution()
    artifact = build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p" * 64, model_audit=_audit(), exp282_pair_audit=pair_audit, exp282_execution_artifact=execution)
    exp282 = artifact["experiments"]["EXP-282"]
    assert exp282["development_match_status"] == "PAIRED_PARTIAL_OBSERVABILITY_DEV_READY"
    assert exp282["paired_execution_evidence"]["artifact_digest"] == execution["artifact_digest"]
    assert exp282["paired_execution_evidence"]["training_replicates"] == 2
    assert exp282["paired_execution_evidence"]["evaluation_replicates"] == 3
    assert exp282["match_court"] == "BLOCKED"
    assert exp282["blockers"] == ["EXP-282: confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution remain open"]
    assert validate_neural_arm_registry(artifact) == []


def test_registry_rejects_tampered_exp282_execution_artifact():
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    pair_audit, execution = _exp282_pair_and_execution()
    execution["evaluation"]["per_replicate"].pop()
    with pytest.raises(ValueError, match="EXP-282 paired execution artifact"):
        build_neural_arm_registry(protocol=_protocol_subset(), protocol_digest="p", model_audit=_audit(), exp282_pair_audit=pair_audit, exp282_execution_artifact=execution)
