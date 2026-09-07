from copy import deepcopy

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_paired_runner import (
    run_exp297_paired_development,
    validate_exp297_execution,
)
from nolane_ai.protocol.evidence import canonical_sha256


def _tiny(**overrides):
    kwargs = dict(
        root_seed="exp297-test",
        eval_replicates=2,
        eval_start_replicate=100,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    kwargs.update(overrides)
    return run_exp297_paired_development(**kwargs)


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    payload["artifact_digest"] = canonical_sha256(clean)


def test_exp297_runner_reconstructs_endpoints_from_raw_candidate_decisions():
    artifact = _tiny()
    assert artifact["schema"] == "NLM-EXP-297-PAIRED-DEV-EVAL-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_seed_materialized",
        "challenge_materialized",
        "decision_rule_executed",
        "hidden_trap_family_consumed",
        "semantic_authority_promoted",
    ):
        assert artifact[flag] is False
    assert validate_exp297_execution(artifact) == []

    aggregate = artifact["evaluation"]["aggregate"]
    assert aggregate["compile_only"]["semantic_fidelity_balanced_accuracy"] == 0.5
    assert aggregate["compile_only"]["wrong_formalization_authority_rate"] == 1.0
    assert aggregate["fidelity_court"]["semantic_fidelity_balanced_accuracy"] == 1.0
    assert aggregate["fidelity_court"]["wrong_formalization_authority_rate"] == 0.0
    assert aggregate["fidelity_court"]["faithful_formalization_rejection_rate"] == 0.0


def test_exp297_validator_rejects_rehashed_truth_witness_authority_cost_and_order_tampering():
    artifact = _tiny(eval_replicates=1)
    mutations = []

    changed = deepcopy(artifact)
    changed["evaluation"]["raw_candidates"][0]["is_faithful"] = not changed["evaluation"]["raw_candidates"][0]["is_faithful"]
    mutations.append(changed)

    changed = deepcopy(artifact)
    changed["evaluation"]["raw_candidates"][1]["court_receipt"]["decision"] = "court_accept"
    mutations.append(changed)

    changed = deepcopy(artifact)
    changed["evaluation"]["raw_candidates"][1]["arms"]["fidelity_court"]["authority_granted"] = True
    mutations.append(changed)

    changed = deepcopy(artifact)
    changed["evaluation"]["raw_candidates"][1]["arms"]["fidelity_court"]["semantic_verification_operations"] = 0
    mutations.append(changed)

    changed = deepcopy(artifact)
    rows = changed["evaluation"]["raw_candidates"]
    rows[0], rows[1] = rows[1], rows[0]
    mutations.append(changed)

    for changed in mutations:
        _rehash(changed)
        errors = validate_exp297_execution(changed)
        assert errors


def test_exp297_inconclusive_verification_fails_closed():
    artifact = _tiny(eval_replicates=1, max_exact_assignments=8)
    rows = artifact["evaluation"]["raw_candidates"]
    faithful_rows = [row for row in rows if row["is_faithful"]]
    assert faithful_rows
    assert all(row["court_receipt"]["decision"] == "court_inconclusive" for row in faithful_rows)
    assert all(not row["arms"]["fidelity_court"]["authority_granted"] for row in faithful_rows)
    assert artifact["evaluation"]["aggregate"]["fidelity_court"]["faithful_formalization_rejection_rate"] == 1.0
    assert validate_exp297_execution(artifact) == []
