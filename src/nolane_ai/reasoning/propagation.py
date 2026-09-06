from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .cps import CanonicalProblemState


@dataclass(slots=True)
class PropagationStats:
    constraint_checks: int = 0
    revisions: int = 0
    values_pruned: int = 0
    passes: int = 0


@dataclass(slots=True)
class PropagationResult:
    domains: dict[str, tuple[int, ...]]
    consistent: bool
    conflict_constraints: tuple[str, ...]
    stats: PropagationStats


def propagate(
    problem: CanonicalProblemState,
    assignment: Mapping[str, int] | None = None,
) -> PropagationResult:
    domains: dict[str, tuple[int, ...]] = problem.domains
    if assignment:
        for name, value in assignment.items():
            if name not in domains or value not in domains[name]:
                return PropagationResult(domains, False, (), PropagationStats())
            domains[name] = (value,)

    stats = PropagationStats()
    changed = True
    while changed:
        changed = False
        stats.passes += 1
        for constraint in problem.constraints:
            for variable in constraint.scope:
                current = domains[variable]
                kept: list[int] = []
                for value in current:
                    supported, checks = constraint.has_support(variable, value, domains)
                    stats.constraint_checks += checks
                    if supported:
                        kept.append(value)
                if len(kept) != len(current):
                    stats.revisions += 1
                    stats.values_pruned += len(current) - len(kept)
                    domains[variable] = tuple(kept)
                    changed = True
                    if not kept:
                        conflict = tuple(
                            c.name for c in problem.constraints if variable in c.scope
                        )
                        return PropagationResult(domains, False, conflict, stats)
    return PropagationResult(domains, True, (), stats)
