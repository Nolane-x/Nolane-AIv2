from __future__ import annotations

import importlib

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralFidelityCourt


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_language_worlds")


def test_language_world_is_deterministic_balanced_and_frozen():
    m = _module()
    left = m.generate_exp298_language_world(29821)
    right = m.generate_exp298_language_world(29821)
    assert left == right
    assert len(left.candidates) == 16
    assert sum(case.is_faithful for case in left.candidates) == 8
    assert sum(not case.is_faithful for case in left.candidates) == 8
    assert {case.stratum for case in left.candidates} == set(m.EXPECTED_STRATA)
    assert m.EXPECTED_STRATA == (
        "faithful_equivalent",
        "quantifier_scope_swap",
        "negation_scope_flip",
        "relation_direction_swap",
        "referent_binding_swap",
        "attachment_swap",
        "conjunction_disjunction_flip",
        "existential_universal_flip",
    )


def test_language_candidates_hide_evaluator_taxonomy_from_arm_view():
    m = _module()
    batch = m.generate_exp298_language_world(29822)
    for case in batch.candidates:
        view = case.arm_view()
        assert set(view) == {"candidate_id", "candidate_digest", "candidate"}
        assert "is_faithful" not in view
        assert "stratum" not in view


def test_grounded_language_probe_space_is_finite_and_canonical():
    m = _module()
    batch = m.generate_exp298_language_world(29823)
    semantics = m.GroundedLanguageSemantics(batch.source)
    probes = semantics.probes()
    assert 8 <= len(probes) <= 4096
    assert tuple(probe.probe_id for probe in probes) == tuple(sorted(probe.probe_id for probe in probes))
    assert len({probe.payload for probe in probes}) == len(probes)


def test_every_frozen_language_trap_has_an_exact_grounded_witness():
    m = _module()
    batch = m.generate_exp298_language_world(29824)
    semantics = m.GroundedLanguageSemantics(batch.source)
    court = BehavioralFidelityCourt(max_exact_probes=4096)
    observed_wrong = set()
    for case in batch.candidates:
        receipt = court.adjudicate(batch.source, case.candidate, semantics)
        if case.is_faithful:
            assert receipt.decision == "court_accept"
            assert receipt.complete_probe_space is True
        else:
            observed_wrong.add(case.stratum)
            assert receipt.decision == "court_reject", case.stratum
            assert receipt.witness is not None
            assert isinstance(receipt.witness["source_outcome"]["value"], bool)
            assert isinstance(receipt.witness["candidate_outcome"]["value"], bool)
    assert observed_wrong == set(m.EXPECTED_STRATA[1:])


def test_language_domain_is_controlled_not_free_form():
    m = _module()
    batch = m.generate_exp298_language_world(29825)
    assert batch.source.domain_kind == "finite_grounded_micro_world"
    assert batch.source.open_language is False
    for case in batch.candidates:
        assert case.candidate.open_language is False
        assert case.candidate.domain_kind == "finite_grounded_micro_world"
