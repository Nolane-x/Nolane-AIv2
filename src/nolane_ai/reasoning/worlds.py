from __future__ import annotations

import random

from .cps import CanonicalProblemState, TableConstraint, Variable


def make_structure_dense_world(seed: int, *, variables: int = 7, domain_size: int = 3) -> CanonicalProblemState:
    if variables < 2 or domain_size < 2:
        raise ValueError("structure-dense world requires variables>=2 and domain_size>=2")
    rng = random.Random(seed)
    names = tuple(f"x{i}" for i in range(variables))
    domain = tuple(range(domain_size))
    anchor_value = rng.randrange(domain_size)
    all_equal = tuple(tuple(value for _ in names) for value in domain)
    return CanonicalProblemState(
        variables=tuple(Variable(name, domain) for name in names),
        constraints=(
            TableConstraint("all_equal", names, all_equal),
            TableConstraint("anchor", (names[-1],), ((anchor_value,),)),
        ),
        world_id=f"dense-{seed}",
    )


def make_conflict_world(seed: int, *, decoys: int = 5) -> CanonicalProblemState:
    rng = random.Random(seed)
    domain = (0, 1)
    decoy_vars = [Variable(f"d{i}", domain) for i in range(decoys)]
    if rng.random() < 0.5:
        decoy_vars.reverse()
    variables = tuple(decoy_vars + [Variable("core_x", domain), Variable("core_y", domain)])
    equality = ((0, 0), (1, 1))
    inequality = ((0, 1), (1, 0))
    return CanonicalProblemState(
        variables=variables,
        constraints=(
            TableConstraint("core_equal", ("core_x", "core_y"), equality),
            TableConstraint("core_unequal", ("core_x", "core_y"), inequality),
        ),
        oracle_conflict_variables=("core_x", "core_y"),
        world_id=f"conflict-{seed}",
    )


def make_fidelity_pair(seed: int) -> tuple[CanonicalProblemState, CanonicalProblemState, CanonicalProblemState]:
    rng = random.Random(seed)
    domain = (0, 1, 2)
    variables = (Variable("x", domain), Variable("y", domain))
    shift = rng.randrange(3)
    faithful_rows = tuple((x, (x + shift) % 3) for x in domain)
    wrong_rows = tuple((x, (x + shift + 1) % 3) for x in domain)
    original = CanonicalProblemState(variables, (TableConstraint("relation", ("x", "y"), faithful_rows),), world_id=f"fid-{seed}")
    faithful = CanonicalProblemState(variables, (TableConstraint("compiled_relation", ("x", "y"), faithful_rows),), world_id=f"fid-{seed}-faithful")
    wrong = CanonicalProblemState(variables, (TableConstraint("compiled_relation", ("x", "y"), wrong_rows),), world_id=f"fid-{seed}-wrong")
    return original, faithful, wrong


def make_satisfiable_backtracking_world(seed: int) -> CanonicalProblemState:
    """Small satisfiable world with an early chronological dead-end.

    The seed changes irrelevant naming/order metadata without changing the intended
    mechanism: branch value 0 for y is a dead-end, while x=0,y=1 is valid.
    """
    rng = random.Random(seed)
    domain = (0, 1)
    decoy_name = f"d{rng.randrange(1_000_000)}"
    variables = (Variable(decoy_name, domain), Variable("x", domain), Variable("y", domain))
    inequality = ((0, 1), (1, 0))
    return CanonicalProblemState(
        variables=variables,
        constraints=(
            TableConstraint("xy_neq", ("x", "y"), inequality),
            TableConstraint("y_anchor", ("y",), ((1,),)),
        ),
        world_id=f"sat-backtrack-{seed}",
    )
