from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random
from typing import Any

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralOutcome, BehavioralProbe
from nolane_ai.protocol.evidence import canonical_sha256

EXPECTED_STRATA = (
    "faithful_equivalent",
    "quantifier_scope_swap",
    "negation_scope_flip",
    "relation_direction_swap",
    "referent_binding_swap",
    "attachment_swap",
    "conjunction_disjunction_flip",
    "existential_universal_flip",
)


@dataclass(frozen=True, slots=True)
class LanguageExpr:
    op: str
    args: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class GroundedLanguageProblem:
    entities: tuple[str, str]
    expression: LanguageExpr
    domain_kind: str = "finite_grounded_micro_world"
    open_language: bool = False


@dataclass(frozen=True, slots=True)
class GroundedLanguageCandidateCase:
    candidate_id: str
    stratum: str
    is_faithful: bool
    candidate: GroundedLanguageProblem
    candidate_digest: str

    def arm_view(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "candidate": self.candidate,
        }


@dataclass(frozen=True, slots=True)
class GroundedLanguageWorldBatch:
    seed: int
    source: GroundedLanguageProblem
    source_digest: str
    candidates: tuple[GroundedLanguageCandidateCase, ...]


@dataclass(frozen=True, slots=True)
class _Context:
    unary: tuple[tuple[str, tuple[str, ...]], ...]
    relation: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


def pred(name: str, variable: str) -> LanguageExpr:
    return LanguageExpr("pred", (name, variable))


def rel(name: str, left: str, right: str) -> LanguageExpr:
    return LanguageExpr("rel", (name, left, right))


def neg(child: LanguageExpr) -> LanguageExpr:
    return LanguageExpr("not", (child,))


def land(*children: LanguageExpr) -> LanguageExpr:
    return LanguageExpr("and", tuple(children))


def lor(*children: LanguageExpr) -> LanguageExpr:
    return LanguageExpr("or", tuple(children))


def exists(variable: str, child: LanguageExpr) -> LanguageExpr:
    return LanguageExpr("exists", (variable, child))


def forall(variable: str, child: LanguageExpr) -> LanguageExpr:
    return LanguageExpr("forall", (variable, child))


def _expr_payload(value: Any) -> Any:
    if isinstance(value, LanguageExpr):
        return {"op": value.op, "args": [_expr_payload(item) for item in value.args]}
    return value


def canonical_language_problem_digest(problem: GroundedLanguageProblem) -> str:
    return canonical_sha256(
        {
            "entities": list(problem.entities),
            "expression": _expr_payload(problem.expression),
            "domain_kind": problem.domain_kind,
            "open_language": problem.open_language,
        }
    )


def _context_from_mask(mask: int, entities: tuple[str, str]) -> _Context:
    a, b = entities
    facts = (
        ("P", a),
        ("P", b),
        ("Q", a),
        ("Q", b),
        ("S", a),
        ("S", b),
        ("R", (a, b)),
        ("R", (b, a)),
    )
    unary: dict[str, set[str]] = {"P": set(), "Q": set(), "S": set()}
    relation: set[tuple[str, str]] = set()
    for bit, fact in enumerate(facts):
        if not (mask & (1 << bit)):
            continue
        name, value = fact
        if name == "R":
            relation.add(value)
        else:
            unary[name].add(value)
    return _Context(
        unary=tuple((name, tuple(sorted(values))) for name, values in sorted(unary.items())),
        relation=(("R", tuple(sorted(relation))),),
    )


def _eval(
    expr: LanguageExpr,
    *,
    entities: tuple[str, str],
    context: _Context,
    env: dict[str, str],
) -> bool:
    unary = {name: set(values) for name, values in context.unary}
    relations = {name: set(values) for name, values in context.relation}
    if expr.op == "pred":
        name, variable = expr.args
        return env[str(variable)] in unary[str(name)]
    if expr.op == "rel":
        name, left, right = expr.args
        return (env[str(left)], env[str(right)]) in relations[str(name)]
    if expr.op == "not":
        return not _eval(expr.args[0], entities=entities, context=context, env=env)
    if expr.op == "and":
        return all(_eval(child, entities=entities, context=context, env=env) for child in expr.args)
    if expr.op == "or":
        return any(_eval(child, entities=entities, context=context, env=env) for child in expr.args)
    if expr.op in {"exists", "forall"}:
        variable, child = expr.args
        results = []
        for entity in entities:
            nested = dict(env)
            nested[str(variable)] = entity
            results.append(_eval(child, entities=entities, context=context, env=nested))
        return any(results) if expr.op == "exists" else all(results)
    raise ValueError(f"unsupported grounded-language operator: {expr.op}")


class GroundedLanguageSemantics:
    domain_id = "grounded_language_ambiguity"

    def __init__(self, source: GroundedLanguageProblem) -> None:
        self._entities = source.entities
        self._probes = tuple(
            BehavioralProbe(probe_id=f"context-{mask:03d}", payload=(mask,))
            for mask in range(256)
        )

    def probes(self) -> tuple[BehavioralProbe, ...]:
        return self._probes

    def evaluate(
        self, problem: GroundedLanguageProblem, probe: BehavioralProbe
    ) -> BehavioralOutcome:
        mask = int(probe.payload[0])
        context = _context_from_mask(mask, problem.entities)
        truth = _eval(
            problem.expression,
            entities=problem.entities,
            context=context,
            env={},
        )
        return BehavioralOutcome(accepted=True, value=bool(truth))

    def canonical_digest(self, problem: GroundedLanguageProblem) -> str:
        return canonical_language_problem_digest(problem)


def _source_expression() -> LanguageExpr:
    inner = land(
        pred("Q", "y"),
        neg(pred("S", "y")),
        rel("R", "x", "y"),
    )
    return exists("x", land(pred("P", "x"), exists("y", inner)))


def _wrong_expression(stratum: str) -> LanguageExpr:
    source_inner = land(
        pred("Q", "y"),
        neg(pred("S", "y")),
        rel("R", "x", "y"),
    )
    if stratum == "quantifier_scope_swap":
        return exists("x", land(pred("P", "x"), forall("y", source_inner)))
    if stratum == "negation_scope_flip":
        changed = land(
            pred("Q", "y"),
            neg(land(pred("S", "y"), rel("R", "x", "y"))),
        )
        return exists("x", land(pred("P", "x"), exists("y", changed)))
    if stratum == "relation_direction_swap":
        changed = land(
            pred("Q", "y"),
            neg(pred("S", "y")),
            rel("R", "y", "x"),
        )
        return exists("x", land(pred("P", "x"), exists("y", changed)))
    if stratum == "referent_binding_swap":
        changed = land(
            pred("Q", "x"),
            neg(pred("S", "y")),
            rel("R", "x", "y"),
        )
        return exists("x", land(pred("P", "x"), exists("y", changed)))
    if stratum == "attachment_swap":
        changed = lor(
            land(pred("Q", "y"), neg(pred("S", "y"))),
            rel("R", "x", "y"),
        )
        return exists("x", land(pred("P", "x"), exists("y", changed)))
    if stratum == "conjunction_disjunction_flip":
        return exists("x", lor(pred("P", "x"), exists("y", source_inner)))
    if stratum == "existential_universal_flip":
        return forall("x", land(pred("P", "x"), exists("y", source_inner)))
    raise ValueError(f"unknown language stratum: {stratum}")


def generate_exp298_language_world(seed: int) -> GroundedLanguageWorldBatch:
    source = GroundedLanguageProblem(
        entities=("a", "b"),
        expression=_source_expression(),
    )
    source_digest = canonical_language_problem_digest(source)
    drafts: list[tuple[str, bool, GroundedLanguageProblem, str]] = []

    for _ in range(8):
        drafts.append(("faithful_equivalent", True, source, source_digest))

    wrong_strata = EXPECTED_STRATA[1:] + ("attachment_swap",)
    for stratum in wrong_strata:
        candidate = GroundedLanguageProblem(
            entities=source.entities,
            expression=_wrong_expression(stratum),
        )
        drafts.append(
            (stratum, False, candidate, canonical_language_problem_digest(candidate))
        )

    order_rng = random.Random(seed ^ 0x2981A6)
    order_rng.shuffle(drafts)
    cases: list[GroundedLanguageCandidateCase] = []
    for slot, (stratum, is_faithful, candidate, digest) in enumerate(drafts):
        nonce = order_rng.getrandbits(128)
        opaque = sha256(f"language|{seed}|{slot}|{nonce}".encode("ascii")).hexdigest()[:20]
        cases.append(
            GroundedLanguageCandidateCase(
                candidate_id=f"case-{opaque}",
                stratum=stratum,
                is_faithful=is_faithful,
                candidate=candidate,
                candidate_digest=digest,
            )
        )

    return GroundedLanguageWorldBatch(
        seed=seed,
        source=source,
        source_digest=source_digest,
        candidates=tuple(cases),
    )
