from nolane_ai.experiments.exp297_fidelity_worlds import EXPECTED_STRATA, generate_fidelity_world
from nolane_ai.reasoning.fidelity import compile_valid


def test_exp297_generator_is_deterministic_balanced_and_multistratum():
    first = generate_fidelity_world(297001)
    second = generate_fidelity_world(297001)

    assert first == second
    assert tuple(case.stratum for case in first.candidates) == EXPECTED_STRATA
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
    assert any(case.is_faithful for case in first.candidates)
    assert any(not case.is_faithful for case in first.candidates)
    assert all(compile_valid(case.candidate) for case in first.candidates)
    assert len({case.candidate_digest for case in first.candidates}) == len(first.candidates)


def test_arm_view_excludes_evaluator_truth_and_trap_family():
    batch = generate_fidelity_world(297002)
    for case in batch.candidates:
        view = case.arm_view()
        rendered = repr(view)
        assert "is_faithful" not in rendered
        assert "stratum" not in rendered
        assert "trap" not in rendered.lower()
        assert view["candidate_digest"] == case.candidate_digest
