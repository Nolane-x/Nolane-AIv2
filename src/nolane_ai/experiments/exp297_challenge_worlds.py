from __future__ import annotations

from hashlib import sha256
import random
from typing import Iterable

from nolane_ai.experiments.exp297_fidelity_worlds import (
    EXPECTED_STRATA,
    FidelityCandidateCase,
    FidelityWorldBatch,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.reasoning.cps import CanonicalProblemState, TableConstraint, Variable
from nolane_ai.reasoning.fidelity import canonical_problem_digest

CONTRACT_SCHEMA = "NLM-EXP-297-HIDDEN-SEMANTIC-CHALLENGE-CONTRACT-V1"


def challenge_contract() -> dict[str, object]:
    return {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": "EXP-297",
        "lane": "POST_FREEZE_CHALLENGE",
        "candidate_count_per_replicate": 16,
        "strata": list(EXPECTED_STRATA),
        "faithful_per_stratum": 1,
        "wrong_per_stratum": 1,
        "domain_size_range": [3, 5],
        "seed_varied_axes": [
            "domain_size",
            "relation_offset",
            "relation_orientation",
            "allowed_row_order",
            "neutral_variable_permutation",
            "transformation_parameter",
            "candidate_order",
        ],
        "truth_in_arm_view": False,
        "trap_family_in_arm_view": False,
        "candidate_ids": "opaque_sha256_prefix",
        "constraint_labels": "neutral_compiled_c_index",
        "semantic_provenance_check": "exact_bidirectional_fidelity_court",
    }


def challenge_contract_digest() -> str:
    return canonical_sha256(challenge_contract())


def _safe_offsets(domain_size: int) -> list[int]:
    return [
        offset
        for offset in range(1, domain_size)
        if (2 * offset) % domain_size != 0
    ]


def _cyclic_relation(
    domain_size: int,
    *,
    offset: int,
    reverse_orientation: bool,
    rng: random.Random,
) -> tuple[tuple[int, int], ...]:
    rows = [
        ((value + offset) % domain_size, value)
        if reverse_orientation
        else (value, (value + offset) % domain_size)
        for value in range(domain_size)
    ]
    rng.shuffle(rows)
    return tuple(rows)


def _neutralize(
    constraints: Iterable[TableConstraint],
    *,
    rng: random.Random,
) -> tuple[TableConstraint, ...]:
    result: list[TableConstraint] = []
    for index, constraint in enumerate(constraints):
        rows = list(constraint.allowed)
        rng.shuffle(rows)
        result.append(
            TableConstraint(
                f"compiled_c{index}",
                constraint.scope,
                tuple(rows),
            )
        )
    return tuple(result)


def _source_geometry(
    seed: int,
) -> tuple[
    random.Random,
    tuple[Variable, ...],
    tuple[TableConstraint, ...],
    int,
    int,
]:
    rng = random.Random(seed ^ 0x297C0F17)
    domain_size = 3 + rng.randrange(3)
    domain = tuple(range(domain_size))

    names = ["v0", "v1", "v2"]
    rng.shuffle(names)
    variables = tuple(Variable(name, domain) for name in names)

    safe_offsets = _safe_offsets(domain_size)
    offset0 = rng.choice(safe_offsets)
    offset1 = rng.choice(safe_offsets)
    relation0 = _cyclic_relation(
        domain_size,
        offset=offset0,
        reverse_orientation=bool(rng.randrange(2)),
        rng=rng,
    )
    relation1 = _cyclic_relation(
        domain_size,
        offset=offset1,
        reverse_orientation=bool(rng.randrange(2)),
        rng=rng,
    )
    constraints = (
        TableConstraint("source_c0", (names[0], names[1]), relation0),
        TableConstraint("source_c1", (names[1], names[2]), relation1),
    )
    transformation_parameter = rng.randrange(1, domain_size)
    return rng, variables, constraints, domain_size, transformation_parameter


def _wrong_for(
    stratum: str,
    constraints: tuple[TableConstraint, ...],
    *,
    domain_size: int,
    transformation_parameter: int,
    rng: random.Random,
) -> tuple[TableConstraint, ...]:
    first, second = constraints
    if stratum == "faithful_equivalent":
        drop_index = rng.randrange(len(first.allowed))
        rows = tuple(
            row for index, row in enumerate(first.allowed) if index != drop_index
        )
        return (TableConstraint("internal_c0", first.scope, rows), second)
    if stratum == "relation_shift":
        rows = tuple(
            (left, (right + transformation_parameter) % domain_size)
            for left, right in first.allowed
        )
        return (TableConstraint("internal_c0", first.scope, rows), second)
    if stratum == "constraint_drop":
        return (TableConstraint("internal_c0", first.scope, first.allowed),)
    if stratum == "constraint_strengthen":
        drop_index = rng.randrange(len(first.allowed))
        rows = tuple(
            row for index, row in enumerate(first.allowed) if index != drop_index
        )
        return (TableConstraint("internal_c0", first.scope, rows), second)
    if stratum == "constraint_weaken":
        allowed = set(first.allowed)
        extras = [
            (left, right)
            for left in range(domain_size)
            for right in range(domain_size)
            if (left, right) not in allowed
        ]
        rng.shuffle(extras)
        return (
            TableConstraint(
                "internal_c0",
                first.scope,
                first.allowed + (extras[0],),
            ),
            second,
        )
    if stratum == "variable_binding_swap":
        return (
            TableConstraint(
                "internal_c0",
                (first.scope[1], first.scope[0]),
                first.allowed,
            ),
            second,
        )
    if stratum == "domain_mapping_error":
        rows = tuple(
            ((left + transformation_parameter) % domain_size, right)
            for left, right in first.allowed
        )
        return (TableConstraint("internal_c0", first.scope, rows), second)
    if stratum == "negation_or_relation_flip":
        allowed = set(first.allowed)
        complement = tuple(
            (left, right)
            for left in range(domain_size)
            for right in range(domain_size)
            if (left, right) not in allowed
        )
        return (TableConstraint("internal_c0", first.scope, complement), second)
    raise ValueError(f"unknown EXP-297 challenge stratum: {stratum}")


def generate_exp297_challenge_world(seed: int) -> FidelityWorldBatch:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("EXP-297 challenge seed must be a non-negative integer")

    rng, variables, source_constraints, domain_size, transform = _source_geometry(seed)
    source = CanonicalProblemState(
        variables,
        source_constraints,
        world_id=f"exp297-source-{seed}",
    )

    drafts: list[tuple[str, bool, CanonicalProblemState, str]] = []
    for stratum in EXPECTED_STRATA:
        faithful_constraints = _neutralize(source_constraints, rng=rng)
        wrong_constraints = _neutralize(
            _wrong_for(
                stratum,
                source_constraints,
                domain_size=domain_size,
                transformation_parameter=transform,
                rng=rng,
            ),
            rng=rng,
        )
        faithful = CanonicalProblemState(
            variables,
            faithful_constraints,
            world_id=f"exp297-candidate-{seed}",
        )
        wrong = CanonicalProblemState(
            variables,
            wrong_constraints,
            world_id=f"exp297-candidate-{seed}",
        )
        for is_faithful, candidate in ((True, faithful), (False, wrong)):
            drafts.append(
                (
                    stratum,
                    is_faithful,
                    candidate,
                    canonical_problem_digest(candidate),
                )
            )

    rng.shuffle(drafts)
    candidates: list[FidelityCandidateCase] = []
    for slot, (stratum, is_faithful, candidate, candidate_digest) in enumerate(drafts):
        nonce = rng.getrandbits(128)
        opaque = sha256(
            f"challenge|{seed}|{slot}|{nonce}".encode("ascii")
        ).hexdigest()[:20]
        candidates.append(
            FidelityCandidateCase(
                candidate_id=f"case-{opaque}",
                stratum=stratum,
                is_faithful=is_faithful,
                candidate=candidate,
                candidate_digest=candidate_digest,
            )
        )

    return FidelityWorldBatch(
        seed=seed,
        source=source,
        source_digest=canonical_problem_digest(source),
        candidates=tuple(candidates),
    )
