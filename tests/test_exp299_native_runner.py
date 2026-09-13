from __future__ import annotations

from copy import deepcopy

from nolane_ai.experiments.exp299_native_runner import (
    FALSE_FLAGS,
    run_exp299_root,
    validate_exp299_root,
)


HEX64_A = "11" * 32
HEX64_B = "22" * 32
HEX64_C = "33" * 32
HEX40 = "44" * 20


def _run() -> dict[str, object]:
    return run_exp299_root(
        root_seed="exp299-test-root",
        canonical_index=0,
        fit_replicates=1,
        fit_start_replicate=0,
        eval_replicates=1,
        eval_start_replicate=10_000,
        d_model=16,
        hidden_size=16,
        target_parameters=20_000,
        batch_size=16,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        authority_threshold=0.5,
        max_exact_probes=4096,
        protocol_digest=HEX64_A,
        geometry_digest=HEX64_B,
        code_digest=HEX64_C,
        repository_head=HEX40,
    )


def test_tiny_root_is_deterministic_leakage_closed_and_self_validating() -> None:
    root = _run()
    assert root["schema"] == "NLM-EXP-299-NATIVE-FIDELITY-ROOT-V1"
    assert root["experiment_id"] == "EXP-299"
    assert root["evidence_level"] == "EV-E2"
    assert root["canonical_index"] == 0
    assert root["config"]["domains"] == [
        "code_invariant",
        "causal_diagnosis",
        "grounded_language_ambiguity",
    ]
    assert root["config"]["fit_replicates"] == 1
    assert root["config"]["eval_replicates"] == 1
    assert root["fit"]["candidate_count"] == 48
    assert root["evaluation"]["candidate_count"] == 48
    assert root["fit"]["inconclusive_count"] == 0
    assert root["evaluation"]["inconclusive_count"] == 0
    assert set(root["fit"]["candidate_instance_ids"]).isdisjoint(
        set(root["evaluation"]["candidate_instance_ids"])
    )
    assert root["partition_audit"]["fit_eval_disjoint"] is True
    assert root["resource_match"]["pair_audit"]["initialization_match"] is True
    assert root["resource_match"]["pair_audit"]["neural_accounted_flops_match"] is True
    assert root["fit"]["auxiliary_loss_weights"] == {
        "binary_supervision_control": 0.0,
        "court_teacher_then_native": 0.25,
    }
    assert root["model_state_digest_before_evaluation"] == root[
        "model_state_digest_after_evaluation"
    ]
    assert root["model_state_unchanged_during_evaluation"] is True
    assert root["heldout_teacher_scaffold_consumed"] is False
    for flag in FALSE_FLAGS:
        assert root[flag] is False

    for row in root["evaluation"]["raw_candidates"]:
        assert row["prediction_committed_before_court"] is True
        assert set(row["model_causal_input"]) == {
            "structural_pair_digest",
            "executable",
        }
        assert "is_faithful" not in row["model_causal_input"]
        assert "stratum" not in row["model_causal_input"]
        assert "court_receipt" not in row["model_causal_input"]
        for arm in row["arms"].values():
            assert arm["native_inference"] is True
            assert arm["teacher_scaffold_consumed"] is False

    assert validate_exp299_root(root) == []


def test_same_tiny_root_reconstructs_identically() -> None:
    first = _run()
    second = _run()
    assert first["artifact_digest"] == second["artifact_digest"]
    assert first["fit"]["partition_digest"] == second["fit"]["partition_digest"]
    assert first["evaluation"]["raw_digest"] == second["evaluation"]["raw_digest"]


def test_root_validator_fails_closed_on_raw_metric_tamper() -> None:
    root = _run()
    tampered = deepcopy(root)
    tampered["evaluation"]["raw_candidates"][0]["arms"][
        "court_teacher_then_native"
    ]["authority_granted"] = not tampered["evaluation"]["raw_candidates"][0][
        "arms"
    ]["court_teacher_then_native"]["authority_granted"]
    errors = validate_exp299_root(tampered)
    assert errors
    assert any(
        "artifact digest" in error or "aggregate" in error or "raw" in error
        for error in errors
    )


def test_root_validator_rejects_evidence_boundary_escalation() -> None:
    root = _run()
    tampered = deepcopy(root)
    tampered["scientific_evidence_eligible"] = True
    errors = validate_exp299_root(tampered)
    assert any("forbidden flag" in error for error in errors)
