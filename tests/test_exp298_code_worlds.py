from __future__ import annotations

import importlib

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralFidelityCourt


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_code_worlds")


def test_code_world_is_deterministic_balanced_and_frozen():
    m = _module()
    left = m.generate_exp298_code_world(29801)
    right = m.generate_exp298_code_world(29801)
    assert left == right
    assert left.seed == 29801
    assert len(left.candidates) == 16
    assert sum(case.is_faithful for case in left.candidates) == 8
    assert sum(not case.is_faithful for case in left.candidates) == 8
    assert {case.stratum for case in left.candidates} == set(m.EXPECTED_STRATA)
    assert m.EXPECTED_STRATA == (
        "faithful_equivalent",
        "off_by_one_boundary",
        "stale_variable_reference",
        "branch_omission",
        "variable_binding_swap",
        "predicate_polarity_flip",
        "invariant_strengthen",
        "invariant_weaken",
        "incorrect_post_state_reference",
    )


def test_code_candidates_hide_evaluator_taxonomy_from_arm_view():
    m = _module()
    batch = m.generate_exp298_code_world(29802)
    for case in batch.candidates:
        view = case.arm_view()
        assert set(view) == {"candidate_id", "candidate_digest", "candidate"}
        assert "is_faithful" not in view
        assert "stratum" not in view


def test_exact_code_semantics_accepts_faithful_and_rejects_every_wrong_candidate():
    m = _module()
    batch = m.generate_exp298_code_world(29803)
    semantics = m.CodeInvariantSemantics(batch.source)
    court = BehavioralFidelityCourt(max_exact_probes=4096)
    decisions = []
    for case in batch.candidates:
        receipt = court.adjudicate(batch.source, case.candidate, semantics)
        decisions.append((case.is_faithful, case.stratum, receipt.decision))
        if case.is_faithful:
            assert receipt.decision == "court_accept"
            assert receipt.complete_probe_space is True
        else:
            assert receipt.decision == "court_reject", case.stratum
            assert receipt.witness is not None
    assert all(decision == "court_reject" for faithful, _, decision in decisions if not faithful)


def test_code_probe_space_is_finite_canonical_and_executable():
    m = _module()
    batch = m.generate_exp298_code_world(29804)
    semantics = m.CodeInvariantSemantics(batch.source)
    probes = semantics.probes()
    assert 1 <= len(probes) <= 4096
    assert tuple(probe.probe_id for probe in probes) == tuple(sorted(probe.probe_id for probe in probes))
    for problem in (batch.source, *(case.candidate for case in batch.candidates)):
        for probe in probes:
            outcome = semantics.evaluate(problem, probe)
            assert isinstance(outcome.accepted, bool)
            assert isinstance(outcome.value, tuple)
