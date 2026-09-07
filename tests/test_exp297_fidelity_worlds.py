from nolane_ai.experiments.exp297_fidelity_worlds import EXPECTED_STRATA, generate_fidelity_world
from nolane_ai.reasoning.fidelity import FidelityCourt, compile_valid


def test_exp297_generator_is_deterministic_balanced_and_multistratum():
    first = generate_fidelity_world(297001)
    second = generate_fidelity_world(297001)

    assert first == second
    observed_order = tuple(dict.fromkeys(case.stratum for case in first.candidates))
    assert observed_order == EXPECTED_STRATA
    assert set(EXPECTED_STRATA) == {
        "faithful_equivalent",
        "relation_shift",
        "constraint_drop",
        "constraint_strengthen",
        "constraint_weaken",
        "variable_binding_swap",
        "domain_mapping_error",
        "negation_or_relation_flip",
    }
    for stratum in EXPECTED_STRATA:
        cases = [case for case in first.candidates if case.stratum == stratum]
        assert {case.is_faithful for case in cases} == {True, False}
    assert all(compile_valid(case.candidate) for case in first.candidates)
    assert len({case.candidate_digest for case in first.candidates}) == len(first.candidates)


def test_construction_provenance_matches_exact_semantics_across_seeded_traps():
    court = FidelityCourt(max_exact_assignments=4096)
    for seed in (1, 2, 3, 8, 21, 297001):
        batch = generate_fidelity_world(seed)
        for case in batch.candidates:
            assert court.accept(batch.source, case.candidate) is case.is_faithful, (
                seed,
                case.stratum,
                case.candidate_id,
            )


def test_arm_view_excludes_evaluator_truth_and_trap_family():
    batch = generate_fidelity_world(297002)
    for case in batch.candidates:
        view = case.arm_view()
        rendered = repr(view)
        assert "is_faithful" not in rendered
        assert "stratum" not in rendered
        assert "trap" not in rendered.lower()
        assert view["candidate_digest"] == case.candidate_digest
