from __future__ import annotations

import importlib

import pytest

pytest.importorskip("torch")


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_paired_runner")


def _run():
    m = _module()
    return m.run_exp298_root(
        root_seed="20260913-exp298-test-root",
        canonical_index=0,
        eval_replicates=1,
        eval_start_replicate=60000,
        d_model=16,
        hidden_size=16,
        target_parameters=50000,
        max_exact_probes=4096,
        protocol_digest="a" * 64,
        geometry_digest="b" * 64,
        code_digest="c" * 64,
        repository_head="d" * 40,
    )


def test_root_runner_publishes_three_domain_reconstructable_evidence():
    m = _module()
    payload = _run()
    assert payload["schema"] == "NLM-EXP-298-CROSS-DOMAIN-FIDELITY-ROOT-V1"
    assert payload["experiment_id"] == "EXP-298"
    assert payload["canonical_index"] == 0
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "CROSS_DOMAIN_FIDELITY_TRANSFER_ESTABLISHED"
    assert payload["successor_design_authorized"] is False
    assert payload["authorization_scope"] == "NONE"
    assert payload["evaluation_candidate_count"] == 48
    assert payload["heldout_domain_count"] == 3
    assert payload["model_state_unchanged_during_evaluation"] is True
    assert set(payload["evaluation"]["domains"]) == {
        "code_invariant",
        "causal_diagnosis",
        "grounded_language_ambiguity",
    }
    assert len(payload["evaluation"]["raw_candidates"]) == 48
    for domain, metrics in payload["evaluation"]["domains"].items():
        assert metrics["candidate_count"] == 16, domain
        assert metrics["compile_only_control"]["faithful_count"] == 8
        assert metrics["compile_only_control"]["wrong_count"] == 8
        assert metrics["cross_domain_fidelity_fabric"]["faithful_count"] == 8
        assert metrics["cross_domain_fidelity_fabric"]["wrong_count"] == 8
        assert metrics["balanced_accuracy_gain"] >= 0.10
        assert metrics["cross_domain_fidelity_fabric"]["wrong_formalization_authority_rate"] <= 0.05
        assert metrics["cross_domain_fidelity_fabric"]["faithful_formalization_rejection_rate"] <= 0.10
    assert payload["evaluation"]["worst_domain_fidelity_gain"] >= 0.10
    assert m.validate_exp298_root(payload) == []


def test_root_rows_publish_leakage_and_cost_boundaries():
    payload = _run()
    for row in payload["evaluation"]["raw_candidates"]:
        assert row["arm_input_receipt"] == {
            "candidate_set_frozen_before_arms": True,
            "byte_identical_candidate_order": True,
            "source_digest_match": True,
            "candidate_digest_match": True,
            "evaluator_truth_in_causal_path": False,
            "trap_family_in_causal_path": False,
        }
        control = row["arms"]["compile_only_control"]
        fidelity = row["arms"]["cross_domain_fidelity_fabric"]
        assert control["compile_validation_operations"] == 1
        assert control["semantic_probe_evaluations"] == 0
        assert control["semantic_domain_operations"] == 0
        assert control["hardware_profiler_flops_claimed"] is False
        assert fidelity["compile_validation_operations"] == 1
        assert fidelity["semantic_probe_evaluations"] >= 1
        assert fidelity["semantic_domain_operations"] >= 2
        assert fidelity["hardware_profiler_flops_claimed"] is False
        assert fidelity["total_accounted_cost_proxy"] >= fidelity["neural_accounted_flops"]


def test_root_publishes_all_frozen_non_claims_false():
    payload = _run()
    for key in (
        "scientific_evidence_eligible",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
        "unrestricted_semantic_authority_claimed",
        "open_language_understanding_claimed",
        "causal_discovery_claimed",
        "general_code_reasoning_claimed",
        "exp290_authority_inherited",
        "exp291_296_authority_inherited",
    ):
        assert payload[key] is False, key
