from __future__ import annotations

from dataclasses import dataclass
import random

from nolane_ai.reasoning.cps import CanonicalProblemState, TableConstraint, Variable
from nolane_ai.reasoning.fidelity import canonical_problem_digest

EXPECTED_STRATA = (
    "faithful_equivalent",
    "relation_shift",
    "constraint_drop",
    "constraint_strengthen",
    "constraint_weaken",
    "variable_binding_swap",
    "domain_mapping_error",
    "negation_or_relation_flip",
)


@dataclass(frozen=True, slots=True)
class FidelityCandidateCase:
    candidate_id: str
    stratum: str
    is_faithful: bool
    candidate: CanonicalProblemState
    candidate_digest: str

    def arm_view(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "candidate": self.candidate,
        }


@dataclass(frozen=True, slots=True)
class FidelityWorldBatch:
    seed: int
    source: CanonicalProblemState
    source_digest: str
    candidates: tuple[FidelityCandidateCase, ...]


def _problem(
    variables: tuple[Variable, ...],
    constraints: tuple[TableConstraint, ...],
    world_id: str,
) -> CanonicalProblemState:
    return CanonicalProblemState(variables, constraints, world_id=world_id)


def _base(seed: int) -> tuple[tuple[Variable, ...], tuple[TableConstraint, ...]]:
    rng = random.Random(seed)
    domain = (0, 1, 2)
    variables = (Variable("x", domain), Variable("y", domain), Variable("z", domain))
    # Use a deliberately non-symmetric source relation so swapping bindings is
    # guaranteed to alter semantics instead of becoming neutral under equality.
    shift = 1 + rng.randrange(2)
    relation = tuple((x, (x + shift) % 3) for x in domain)
    relation2 = tuple((y, (y + 1) % 3) for y in domain)
    constraints = (
        TableConstraint("xy_relation", ("x", "y"), relation),
        TableConstraint("yz_relation", ("y", "z"), relation2),
    )
    return variables, constraints


def _renamed(
    constraints: tuple[TableConstraint, ...], suffix: str
) -> tuple[TableConstraint, ...]:
    return tuple(
        TableConstraint(f"{constraint.name}_{suffix}", constraint.scope, constraint.allowed)
        for constraint in constraints
    )


def _wrong_for(
    stratum: str, constraints: tuple[TableConstraint, ...]
) -> tuple[TableConstraint, ...]:
    first, second = constraints
    domain = (0, 1, 2)
    if stratum == "faithful_equivalent":
        rows = tuple(row for row in first.allowed if row != first.allowed[0])
        return (TableConstraint("faithful_equivalent_negative", first.scope, rows), second)
    if stratum == "relation_shift":
        rows = tuple((x, (y + 1) % 3) for x, y in first.allowed)
        return (TableConstraint("relation_shift", first.scope, rows), second)
    if stratum == "constraint_drop":
        return (TableConstraint("constraint_drop_keep", first.scope, first.allowed),)
    if stratum == "constraint_strengthen":
        return (TableConstraint("constraint_strengthen", first.scope, first.allowed[:-1]), second)
    if stratum == "constraint_weaken":
        extras = tuple(
            (x, y)
            for x in domain
            for y in domain
            if (x, y) not in first.allowed
        )
        return (
            TableConstraint("constraint_weaken", first.scope, first.allowed + extras[:1]),
            second,
        )
    if stratum == "variable_binding_swap":
        return (TableConstraint("binding_swap", ("y", "x"), first.allowed), second)
    if stratum == "domain_mapping_error":
        mapped = tuple(((x + 1) % 3, y) for x, y in first.allowed)
        return (TableConstraint("domain_mapping_error", first.scope, mapped), second)
    if stratum == "negation_or_relation_flip":
        complement = tuple(
            (x, y)
            for x in domain
            for y in domain
            if (x, y) not in first.allowed
        )
        return (TableConstraint("relation_flip", first.scope, complement), second)
    raise ValueError(f"unknown fidelity stratum: {stratum}")


def generate_fidelity_world(seed: int) -> FidelityWorldBatch:
    variables, constraints = _base(seed)
    source = _problem(variables, constraints, f"exp297-source-{seed}")
    cases: list[FidelityCandidateCase] = []
    for index, stratum in enumerate(EXPECTED_STRATA):
        faithful = _problem(
            variables,
            _renamed(constraints, f"{stratum}_pos"),
            f"exp297-{seed}-{index}-pos",
        )
        wrong = _problem(
            variables,
            _wrong_for(stratum, constraints),
            f"exp297-{seed}-{index}-neg",
        )
        for polarity, is_faithful, candidate in (
            ("faithful", True, faithful),
            ("wrong", False, wrong),
        ):
            cases.append(
                FidelityCandidateCase(
                    candidate_id=f"{seed}:{index}:{polarity}",
                    stratum=stratum,
                    is_faithful=is_faithful,
                    candidate=candidate,
                    candidate_digest=canonical_problem_digest(candidate),
                )
            )
    return FidelityWorldBatch(
        seed=seed,
        source=source,
        source_digest=canonical_problem_digest(source),
        candidates=tuple(cases),
    )
