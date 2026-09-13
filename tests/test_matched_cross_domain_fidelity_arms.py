from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralFidelityCourt
from nolane_ai.experiments.exp298_code_worlds import CodeInvariantSemantics, generate_exp298_code_world


def _module():
    import importlib

    return importlib.import_module("nolane_ai.experiments.matched_cross_domain_fidelity_arms")


def _receipt(*, faithful: bool):
    batch = generate_exp298_code_world(29831)
    case = next(case for case in batch.candidates if case.is_faithful is faithful)
    receipt = BehavioralFidelityCourt(max_exact_probes=4096).adjudicate(
        batch.source,
        case.candidate,
        CodeInvariantSemantics(batch.source),
    )
    return batch, case, receipt


def test_pair_is_exactly_matched_at_frozen_geometry():
    m = _module()
    torch.manual_seed(298)
    control, fidelity = m.build_matched_cross_domain_fidelity_arms(
        d_model=64,
        hidden_size=64,
        target_parameters=500000,
    )
    audit = m.audit_matched_cross_domain_fidelity_arms(control, fidelity)
    assert audit["schema"] == "NLM-EXP-298-MATCHED-CROSS-DOMAIN-FIDELITY-ARMS-DEV-V1"
    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["active_functional_parameter_match"] is True
    assert audit["optimizer_visible_parameter_match"] is True
    assert audit["initialization_match"] is True
    assert audit["neural_accounted_flops_match"] is True
    assert audit["receipt_access_is_only_causal_intervention"] is True
    assert audit["arms"]["compile_only_control"]["total_parameters"] == 500000
    assert audit["arms"]["cross_domain_fidelity_fabric"]["total_parameters"] == 500000
    assert audit["label_information_consumed"] is False
    assert audit["trap_family_consumed"] is False


def test_control_cannot_consume_behavioral_receipt_and_fidelity_requires_it():
    m = _module()
    control, fidelity = m.build_matched_cross_domain_fidelity_arms(
        d_model=64, hidden_size=64, target_parameters=500000
    )
    batch, case, receipt = _receipt(faithful=True)
    with pytest.raises(ValueError, match="control arm must not consume"):
        control.decide(
            source_digest=batch.source_digest,
            candidate_digest=case.candidate_digest,
            executable=True,
            court_receipt=receipt,
        )
    with pytest.raises(ValueError, match="fidelity arm requires"):
        fidelity.decide(
            source_digest=batch.source_digest,
            candidate_digest=case.candidate_digest,
            executable=True,
        )


def test_authority_semantics_are_fail_closed_and_neural_cost_is_matched():
    m = _module()
    control, fidelity = m.build_matched_cross_domain_fidelity_arms(
        d_model=64, hidden_size=64, target_parameters=500000
    )
    faithful_batch, faithful_case, accept_receipt = _receipt(faithful=True)
    wrong_batch, wrong_case, reject_receipt = _receipt(faithful=False)

    control_decision = control.decide(
        source_digest=faithful_batch.source_digest,
        candidate_digest=faithful_case.candidate_digest,
        executable=True,
    )
    accept_decision = fidelity.decide(
        source_digest=faithful_batch.source_digest,
        candidate_digest=faithful_case.candidate_digest,
        executable=True,
        court_receipt=accept_receipt,
    )
    reject_decision = fidelity.decide(
        source_digest=wrong_batch.source_digest,
        candidate_digest=wrong_case.candidate_digest,
        executable=True,
        court_receipt=reject_receipt,
    )

    assert control_decision.authority_granted is True
    assert accept_decision.authority_granted is True
    assert reject_decision.authority_granted is False
    assert control_decision.receipt_semantics == "canonical_null_cross_domain_receipt"
    assert accept_decision.receipt_semantics == "observed_public_behavioral_fidelity_receipt"
    assert control_decision.neural_accounted_flops == accept_decision.neural_accounted_flops
    for decision in (control_decision, accept_decision, reject_decision):
        assert 0.0 <= decision.fidelity_score <= 1.0
        assert 0.0 <= decision.authority_score <= 1.0
        assert 0.0 <= decision.verifier_score <= 1.0
