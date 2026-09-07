import torch

from nolane_ai.experiments.exp297_fidelity_worlds import generate_fidelity_world
from nolane_ai.experiments.matched_fidelity_arms import (
    audit_matched_fidelity_arms,
    build_matched_fidelity_arms,
)
from nolane_ai.reasoning.fidelity import FidelityCourt


def test_exp297_arms_are_exactly_matched_and_identically_initialized():
    compile_only, fidelity = build_matched_fidelity_arms(
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
    )
    audit = audit_matched_fidelity_arms(compile_only, fidelity)

    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["active_functional_parameter_match"] is True
    assert audit["optimizer_visible_parameter_match"] is True
    assert audit["initialization_match"] is True
    assert audit["neural_accounted_flops_match"] is True
    assert audit["compute_ledger"]["compile_only"]["neural_accounted_flops"] == audit["compute_ledger"]["fidelity_court"]["neural_accounted_flops"]


def test_compile_only_uses_null_receipt_but_same_neural_path():
    batch = generate_fidelity_world(297101)
    case = batch.candidates[1]
    compile_only, fidelity = build_matched_fidelity_arms(
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
    )
    receipt = FidelityCourt(max_exact_assignments=4096).adjudicate(batch.source, case.candidate)

    compile_decision = compile_only.decide(batch.source, case.candidate, compile_valid=True)
    fidelity_decision = fidelity.decide(
        batch.source,
        case.candidate,
        compile_valid=True,
        court_receipt=receipt,
    )

    assert compile_decision.receipt_semantics == "canonical_null_fidelity_receipt"
    assert fidelity_decision.receipt_semantics == "observed_public_fidelity_receipt"
    assert compile_decision.authority_granted is True
    assert fidelity_decision.authority_granted is (receipt.decision == "court_accept")
    assert torch.isfinite(torch.tensor(compile_decision.fidelity_score))
    assert torch.isfinite(torch.tensor(fidelity_decision.fidelity_score))
