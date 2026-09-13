from __future__ import annotations

import importlib

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralFidelityCourt


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_causal_worlds")


def test_causal_world_is_deterministic_balanced_and_frozen():
    m = _module()
    left = m.generate_exp298_causal_world(29811)
    right = m.generate_exp298_causal_world(29811)
    assert left == right
    assert len(left.candidates) == 16
    assert sum(case.is_faithful for case in left.candidates) == 8
    assert sum(not case.is_faithful for case in left.candidates) == 8
    assert {case.stratum for case in left.candidates} == set(m.EXPECTED_STRATA)
    assert m.EXPECTED_STRATA == (
        "faithful_equivalent",
        "edge_reversal",
        "mediator_omission",
        "wrong_intervention_target",
        "effect_polarity_flip",
        "exogenous_binding_swap",
        "observationally_equivalent_interventionally_wrong",
        "spurious_direct_edge",
    )


def test_causal_candidates_hide_evaluator_taxonomy_from_arm_view():
    m = _module()
    batch = m.generate_exp298_causal_world(29812)
    for case in batch.candidates:
        view = case.arm_view()
        assert set(view) == {"candidate_id", "candidate_digest", "candidate"}
        assert "is_faithful" not in view
        assert "stratum" not in view


def test_intervention_only_trap_matches_observation_but_diverges_under_do():
    m = _module()
    batch = m.generate_exp298_causal_world(29813)
    trap = next(
        case for case in batch.candidates
        if case.stratum == "observationally_equivalent_interventionally_wrong"
        and not case.is_faithful
    )
    semantics = m.CausalDiagnosisSemantics(batch.source)
    observational = [probe for probe in semantics.probes() if probe.payload[-2:] == ("none", -1)]
    interventions = [probe for probe in semantics.probes() if probe.payload[-2:] != ("none", -1)]
    assert observational and interventions
    for probe in observational:
        assert semantics.evaluate(batch.source, probe) == semantics.evaluate(trap.candidate, probe)
    assert any(
        semantics.evaluate(batch.source, probe) != semantics.evaluate(trap.candidate, probe)
        for probe in interventions
    )


def test_exact_causal_semantics_accepts_faithful_and_rejects_every_wrong_candidate():
    m = _module()
    batch = m.generate_exp298_causal_world(29814)
    semantics = m.CausalDiagnosisSemantics(batch.source)
    court = BehavioralFidelityCourt(max_exact_probes=4096)
    for case in batch.candidates:
        receipt = court.adjudicate(batch.source, case.candidate, semantics)
        if case.is_faithful:
            assert receipt.decision == "court_accept"
            assert receipt.complete_probe_space is True
        else:
            assert receipt.decision == "court_reject", case.stratum
            assert receipt.witness is not None


def test_causal_probe_space_is_finite_canonical_and_has_interventions():
    m = _module()
    batch = m.generate_exp298_causal_world(29815)
    semantics = m.CausalDiagnosisSemantics(batch.source)
    probes = semantics.probes()
    assert 1 <= len(probes) <= 4096
    assert tuple(probe.probe_id for probe in probes) == tuple(sorted(probe.probe_id for probe in probes))
    assert any(probe.payload[-2:] == ("none", -1) for probe in probes)
    assert any(probe.payload[-2:] == ("A", 0) for probe in probes)
    assert any(probe.payload[-2:] == ("A", 1) for probe in probes)
    assert any(probe.payload[-2:] == ("B", 0) for probe in probes)
    assert any(probe.payload[-2:] == ("B", 1) for probe in probes)
