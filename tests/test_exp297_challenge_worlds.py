from nolane_ai.experiments.exp297_challenge_worlds import (
    challenge_contract,
    challenge_contract_digest,
    generate_exp297_challenge_world,
)
from nolane_ai.experiments.exp297_fidelity_worlds import EXPECTED_STRATA, generate_fidelity_world
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.reasoning.fidelity import FidelityCourt, compile_valid


def test_exp297_challenge_worlds_are_deterministic_balanced_and_semantically_correct():
    court = FidelityCourt(max_exact_assignments=4096)
    for seed in (1, 2, 3, 11, 97, 297001, 999983):
        first = generate_exp297_challenge_world(seed)
        second = generate_exp297_challenge_world(seed)
        assert first == second
        assert len(first.candidates) == 16
        assert {case.stratum for case in first.candidates} == set(EXPECTED_STRATA)
        for stratum in EXPECTED_STRATA:
            cases = [case for case in first.candidates if case.stratum == stratum]
            assert {case.is_faithful for case in cases} == {True, False}
        assert all(compile_valid(case.candidate) for case in first.candidates)
        for case in first.candidates:
            assert court.accept(first.source, case.candidate) is case.is_faithful


def test_exp297_challenge_arm_view_has_no_polarity_or_trap_shortcuts():
    forbidden = {
        "faithful",
        "wrong",
        "relation_shift",
        "constraint_drop",
        "constraint_strengthen",
        "constraint_weaken",
        "variable_binding_swap",
        "domain_mapping_error",
        "negation_or_relation_flip",
        "binding_swap",
        "relation_flip",
        "_pos",
        "_neg",
    }
    batch = generate_exp297_challenge_world(297777)
    for case in batch.candidates:
        view = case.arm_view()
        rendered = repr(view).lower()
        assert case.candidate_id.startswith("case-")
        assert "stratum" not in rendered
        assert "is_faithful" not in rendered
        assert "trap" not in rendered
        assert all(token not in rendered for token in forbidden)
        assert all(
            constraint.name == f"compiled_c{index}"
            for index, constraint in enumerate(case.candidate.constraints)
        )
        assert case.candidate.world_id.startswith("exp297-candidate-")


def test_exp297_challenge_candidate_position_does_not_encode_class():
    batches = [generate_exp297_challenge_world(seed) for seed in range(1, 65)]
    for position in range(16):
        assert {batch.candidates[position].is_faithful for batch in batches} == {True, False}


def test_exp297_challenge_geometry_is_not_the_development_generator_for_same_seed():
    for seed in (7, 29, 311, 297001):
        challenge = generate_exp297_challenge_world(seed)
        development = generate_fidelity_world(seed)
        challenge_digests = tuple(case.candidate_digest for case in challenge.candidates)
        development_digests = tuple(case.candidate_digest for case in development.candidates)
        assert challenge.source_digest != development.source_digest or challenge_digests != development_digests


def test_exp297_challenge_contract_is_canonical_and_freezes_hidden_geometry_axes():
    contract = challenge_contract()
    assert contract["schema"] == "NLM-EXP-297-HIDDEN-SEMANTIC-CHALLENGE-CONTRACT-V1"
    assert contract["candidate_count_per_replicate"] == 16
    assert contract["strata"] == list(EXPECTED_STRATA)
    assert contract["domain_size_range"] == [3, 5]
    assert set(contract["seed_varied_axes"]) == {
        "domain_size",
        "relation_offset",
        "relation_orientation",
        "allowed_row_order",
        "neutral_variable_permutation",
        "transformation_parameter",
        "candidate_order",
    }
    assert contract["truth_in_arm_view"] is False
    assert contract["trap_family_in_arm_view"] is False
    assert challenge_contract_digest() == canonical_sha256(contract)
