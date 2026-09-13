from __future__ import annotations

import inspect

import pytest

from nolane_ai.experiments.exp298_causal_worlds import generate_exp298_causal_world
from nolane_ai.experiments.exp298_code_worlds import generate_exp298_code_world
from nolane_ai.experiments.exp298_language_worlds import generate_exp298_language_world
from nolane_ai.experiments.exp299_structural_encoding import (
    FEATURE_WIDTH,
    PAIR_FEATURE_WIDTH,
    encode_exp299_pair,
    encode_exp299_problem,
    structural_encoder_contract,
)


DOMAIN_CASES = (
    ("code_invariant", generate_exp298_code_world),
    ("causal_diagnosis", generate_exp298_causal_world),
    ("grounded_language_ambiguity", generate_exp298_language_world),
)


def test_feature_geometry_is_frozen() -> None:
    assert FEATURE_WIDTH == 128
    assert PAIR_FEATURE_WIDTH == 512


def test_problem_encoding_is_deterministic_and_finite() -> None:
    for domain_id, generator in DOMAIN_CASES:
        batch = generator(20260913)
        first = encode_exp299_problem(domain_id, batch.source)
        second = encode_exp299_problem(domain_id, batch.source)
        assert first == second
        assert len(first) == FEATURE_WIDTH
        assert all(isinstance(value, float) for value in first)
        assert all(value == value and abs(value) != float("inf") for value in first)


def test_faithful_candidates_have_zero_structural_delta() -> None:
    for domain_id, generator in DOMAIN_CASES:
        batch = generator(1701)
        faithful = next(case for case in batch.candidates if case.is_faithful)
        pair = encode_exp299_pair(domain_id, batch.source, faithful.candidate)
        assert len(pair) == PAIR_FEATURE_WIDTH
        delta = pair[2 * FEATURE_WIDTH : 3 * FEATURE_WIDTH]
        abs_delta = pair[3 * FEATURE_WIDTH :]
        assert all(value == 0.0 for value in delta)
        assert all(value == 0.0 for value in abs_delta)


def test_every_wrong_candidate_changes_public_structural_representation() -> None:
    for domain_id, generator in DOMAIN_CASES:
        batch = generator(99127)
        source = encode_exp299_problem(domain_id, batch.source)
        wrong = [case for case in batch.candidates if not case.is_faithful]
        assert wrong
        for case in wrong:
            candidate = encode_exp299_problem(domain_id, case.candidate)
            assert candidate != source, (domain_id, case.stratum)
            pair = encode_exp299_pair(domain_id, batch.source, case.candidate)
            abs_delta = pair[3 * FEATURE_WIDTH :]
            assert any(value > 0.0 for value in abs_delta), (domain_id, case.stratum)


def test_encoder_api_has_no_evaluator_only_arguments() -> None:
    forbidden = {"is_faithful", "stratum", "receipt", "witness", "court_decision", "truth"}
    for function in (encode_exp299_problem, encode_exp299_pair):
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & forbidden)


def test_contract_explicitly_excludes_evaluator_fields() -> None:
    contract = structural_encoder_contract()
    assert contract["schema"] == "NLM-EXP-299-STRUCTURAL-ENCODER-V1"
    assert contract["feature_width"] == 128
    assert contract["pair_feature_width"] == 512
    assert contract["overflow_policy"] == "fail_closed"
    excluded = set(contract["excluded_evaluator_fields"])
    assert {
        "is_faithful",
        "stratum",
        "court_decision",
        "court_witness",
        "evaluation_truth",
    } <= excluded


def test_unknown_domain_fails_closed() -> None:
    batch = generate_exp298_code_world(11)
    with pytest.raises(ValueError, match="unknown EXP-299 structural domain"):
        encode_exp299_problem("not_a_domain", batch.source)


def test_wrong_domain_object_fails_closed() -> None:
    code = generate_exp298_code_world(7)
    with pytest.raises(TypeError, match="CausalDiagnosisProblem"):
        encode_exp299_problem("causal_diagnosis", code.source)


def test_all_v1_generated_language_asts_fit_without_truncation() -> None:
    # Vary candidate order/identity roots while exercising every frozen AST family.
    for seed in range(16):
        batch = generate_exp298_language_world(seed)
        encode_exp299_problem("grounded_language_ambiguity", batch.source)
        for case in batch.candidates:
            encoded = encode_exp299_problem(
                "grounded_language_ambiguity", case.candidate
            )
            assert len(encoded) == FEATURE_WIDTH
