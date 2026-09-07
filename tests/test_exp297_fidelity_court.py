from nolane_ai.reasoning.cps import CanonicalProblemState, TableConstraint, Variable
from nolane_ai.reasoning.fidelity import FidelityCourt, canonical_problem_digest
from nolane_ai.reasoning.worlds import make_fidelity_pair


def test_bidirectional_court_emits_directional_witness():
    source, faithful, wrong = make_fidelity_pair(seed=21)
    court = FidelityCourt(max_exact_assignments=4096)

    faithful_receipt = court.adjudicate(source, faithful)
    wrong_receipt = court.adjudicate(source, wrong)

    assert faithful_receipt.decision == "court_accept"
    assert faithful_receipt.witness is None
    assert wrong_receipt.decision == "court_reject"
    assert wrong_receipt.witness is not None
    assert wrong_receipt.witness.direction in {"source_to_candidate", "candidate_to_source"}
    assert wrong_receipt.witness.source_accepts != wrong_receipt.witness.candidate_accepts
    assert wrong_receipt.semantic_verification_operations > 0
    assert canonical_problem_digest(source) != canonical_problem_digest(wrong)


def test_exact_ceiling_is_inconclusive_not_accept():
    variables = tuple(Variable(f"x{i}", (0, 1)) for i in range(5))
    relation = TableConstraint("always", ("x0",), ((0,), (1,)))
    source = CanonicalProblemState(variables, (relation,))
    candidate = CanonicalProblemState(variables, (TableConstraint("same", ("x0",), ((0,), (1,))),))

    receipt = FidelityCourt(max_exact_assignments=16).adjudicate(source, candidate)

    assert receipt.decision == "court_inconclusive"
    assert receipt.termination_reason == "assignment_space_exceeds_exact_ceiling"
    assert not FidelityCourt(max_exact_assignments=16).accept(source, candidate)
