from __future__ import annotations

from itertools import product
from typing import Any

from nolane_ai.experiments.exp298_causal_worlds import (
    CausalDiagnosisProblem,
    EquationSpec,
)
from nolane_ai.experiments.exp298_code_worlds import CodeInvariantProblem
from nolane_ai.experiments.exp298_language_worlds import (
    GroundedLanguageProblem,
    LanguageExpr,
)

FEATURE_WIDTH = 128
PAIR_FEATURE_WIDTH = 4 * FEATURE_WIDTH
SCHEMA = "NLM-EXP-299-STRUCTURAL-ENCODER-V1"

_CAUSAL_SYMBOLS = {
    "U": 1,
    "V": 2,
    "A": 3,
    "B": 4,
    "C": 5,
}
_CAUSAL_OPS = ("identity", "not", "xor", "and", "or")
_LANGUAGE_OPS = {
    "pred": 1,
    "rel": 2,
    "not": 3,
    "and": 4,
    "or": 5,
    "exists": 6,
    "forall": 7,
}
_LANGUAGE_SYMBOLS = {
    "P": 1,
    "Q": 2,
    "S": 3,
    "R": 4,
    "x": 5,
    "y": 6,
    "a": 7,
    "b": 8,
}


def _pad(values: list[float], *, domain_id: str) -> tuple[float, ...]:
    if len(values) > FEATURE_WIDTH:
        raise ValueError(
            f"EXP-299 structural encoding overflow for {domain_id}: "
            f"{len(values)} > {FEATURE_WIDTH}"
        )
    values.extend([0.0] * (FEATURE_WIDTH - len(values)))
    return tuple(float(value) for value in values)


def _encode_code(problem: CodeInvariantProblem) -> tuple[float, ...]:
    values: list[float] = []
    for domain in problem.domains:
        values.append(float(len(domain)) / 4.0)

    # The pre-state is represented by canonical row position.  V1 domains are
    # finite and every transition table is canonicalized by pre-state.
    rows = sorted(problem.transitions, key=lambda item: item[0])
    for _, post in rows:
        values.extend(
            (
                float(post[0]) / 2.0,
                float(post[1]) / 2.0,
                float(post[2]),
            )
        )

    invariant = set(problem.invariant_states)
    for state in product(*problem.domains):
        values.append(float(tuple(int(v) for v in state) in invariant))
    return _pad(values, domain_id="code_invariant")


def _causal_symbol(name: str) -> float:
    if name not in _CAUSAL_SYMBOLS:
        raise ValueError(f"unsupported EXP-299 causal symbol: {name}")
    return float(_CAUSAL_SYMBOLS[name]) / 8.0


def _encode_equation(target: str, spec: EquationSpec) -> list[float]:
    if spec.op not in _CAUSAL_OPS:
        raise ValueError(f"unsupported EXP-299 causal operator: {spec.op}")
    if len(spec.args) > 2:
        raise ValueError("EXP-299 causal equation arity exceeds V1 bound")
    values = [_causal_symbol(target)]
    values.extend(float(spec.op == op) for op in _CAUSAL_OPS)
    values.append(float(len(spec.args)) / 2.0)
    for index in range(2):
        values.append(_causal_symbol(spec.args[index]) if index < len(spec.args) else 0.0)
    return values


def _encode_causal(problem: CausalDiagnosisProblem) -> tuple[float, ...]:
    values: list[float] = [
        float(len(problem.exogenous_names)) / 4.0,
        float(len(problem.endogenous_names)) / 4.0,
    ]
    values.extend(_causal_symbol(name) for name in problem.exogenous_names)
    values.extend(_causal_symbol(name) for name in problem.endogenous_names)
    for target, spec in problem.equations:
        values.extend(_encode_equation(target, spec))
    values.append(float(len(problem.intervention_alias)) / 4.0)
    for source, target in problem.intervention_alias:
        values.extend((_causal_symbol(source), _causal_symbol(target)))
    values.append(_causal_symbol(problem.diagnosis_target))
    return _pad(values, domain_id="causal_diagnosis")


def _language_symbol(value: str) -> float:
    if value not in _LANGUAGE_SYMBOLS:
        raise ValueError(f"unsupported EXP-299 language symbol: {value}")
    return float(_LANGUAGE_SYMBOLS[value]) / 16.0


def _append_expr_tokens(expr: LanguageExpr, values: list[float]) -> None:
    if expr.op not in _LANGUAGE_OPS:
        raise ValueError(f"unsupported EXP-299 language operator: {expr.op}")
    values.append(float(_LANGUAGE_OPS[expr.op]) / 8.0)
    values.append(float(len(expr.args)) / 8.0)
    for argument in expr.args:
        if isinstance(argument, LanguageExpr):
            # Explicit nested marker separates a child AST from a scalar arg.
            values.append(1.0)
            _append_expr_tokens(argument, values)
        elif isinstance(argument, str):
            values.append(0.5)
            values.append(_language_symbol(argument))
        else:
            raise TypeError(
                "EXP-299 grounded-language AST contains unsupported argument type"
            )


def _encode_language(problem: GroundedLanguageProblem) -> tuple[float, ...]:
    if problem.domain_kind != "finite_grounded_micro_world":
        raise ValueError("EXP-299 V1 only supports finite_grounded_micro_world")
    values: list[float] = [
        float(len(problem.entities)) / 4.0,
        float(bool(problem.open_language)),
        1.0,  # frozen finite-grounded domain-kind marker
    ]
    for entity in problem.entities:
        values.append(_language_symbol(entity))
    _append_expr_tokens(problem.expression, values)
    return _pad(values, domain_id="grounded_language_ambiguity")


def encode_exp299_problem(domain_id: str, problem: object) -> tuple[float, ...]:
    if domain_id == "code_invariant":
        if not isinstance(problem, CodeInvariantProblem):
            raise TypeError("code_invariant requires CodeInvariantProblem")
        return _encode_code(problem)
    if domain_id == "causal_diagnosis":
        if not isinstance(problem, CausalDiagnosisProblem):
            raise TypeError("causal_diagnosis requires CausalDiagnosisProblem")
        return _encode_causal(problem)
    if domain_id == "grounded_language_ambiguity":
        if not isinstance(problem, GroundedLanguageProblem):
            raise TypeError(
                "grounded_language_ambiguity requires GroundedLanguageProblem"
            )
        return _encode_language(problem)
    raise ValueError(f"unknown EXP-299 structural domain: {domain_id}")


def encode_exp299_pair(
    domain_id: str,
    source: object,
    candidate: object,
) -> tuple[float, ...]:
    source_features = encode_exp299_problem(domain_id, source)
    candidate_features = encode_exp299_problem(domain_id, candidate)
    delta = tuple(
        candidate_value - source_value
        for source_value, candidate_value in zip(source_features, candidate_features)
    )
    abs_delta = tuple(abs(value) for value in delta)
    result = source_features + candidate_features + delta + abs_delta
    if len(result) != PAIR_FEATURE_WIDTH:
        raise RuntimeError("EXP-299 pair feature geometry mismatch")
    return result


def structural_encoder_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "feature_width": FEATURE_WIDTH,
        "pair_feature_width": PAIR_FEATURE_WIDTH,
        "overflow_policy": "fail_closed",
        "domains": [
            "code_invariant",
            "causal_diagnosis",
            "grounded_language_ambiguity",
        ],
        "pair_channels": [
            "source_features",
            "candidate_features",
            "candidate_minus_source",
            "absolute_difference",
        ],
        "excluded_evaluator_fields": [
            "is_faithful",
            "stratum",
            "court_decision",
            "court_witness",
            "evaluation_truth",
        ],
        "evaluator_information_consumed": False,
    }
