from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .cps import CanonicalProblemState
from .propagation import propagate


@dataclass(slots=True)
class SearchStats:
    nodes: int = 0
    backtracks: int = 0
    constraint_checks: int = 0
    propagation_checks: int = 0
    nogood_checks: int = 0
    nogood_hits: int = 0
    nogood_stores: int = 0
    dead_end_signatures: list[tuple[tuple[str, int], ...]] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    solution: dict[str, int] | None
    verified: bool
    stats: SearchStats

    @property
    def accounted_operations(self) -> int:
        return (
            self.stats.nodes
            + self.stats.constraint_checks
            + self.stats.propagation_checks
            + self.stats.nogood_checks
        )


class EpisodeNogoodStore:
    def __init__(self) -> None:
        self._nogoods: set[frozenset[tuple[str, int]]] = set()

    def matches(self, assignment: Mapping[str, int]) -> bool:
        current = frozenset(assignment.items())
        return any(nogood.issubset(current) for nogood in self._nogoods)

    def add(self, assignment: Mapping[str, int]) -> None:
        self._nogoods.add(frozenset(assignment.items()))

    def __len__(self) -> int:
        return len(self._nogoods)


def _constraint_consistent(problem: CanonicalProblemState, assignment: Mapping[str, int], stats: SearchStats) -> bool:
    for constraint in problem.constraints:
        if all(name in assignment for name in constraint.scope):
            stats.constraint_checks += 1
            if not constraint.is_satisfied(assignment):
                return False
    return True


def _ordered_variables(problem: CanonicalProblemState, priority_variables: tuple[str, ...]) -> tuple[str, ...]:
    priority = tuple(dict.fromkeys(priority_variables))
    return priority + tuple(name for name in problem.variable_names if name not in priority)


def solve_branch(
    problem: CanonicalProblemState,
    *,
    priority_variables: tuple[str, ...] = (),
    nogood_store: EpisodeNogoodStore | None = None,
) -> SearchResult:
    stats = SearchStats()
    order = _ordered_variables(problem, priority_variables)
    domains = problem.domains

    def dfs(depth: int, assignment: dict[str, int]) -> dict[str, int] | None:
        stats.nodes += 1
        if nogood_store is not None:
            stats.nogood_checks += 1
            if nogood_store.matches(assignment):
                stats.nogood_hits += 1
                return None
        if not _constraint_consistent(problem, assignment, stats):
            stats.backtracks += 1
            stats.dead_end_signatures.append(tuple(sorted(assignment.items())))
            if nogood_store is not None:
                nogood_store.add(assignment)
                stats.nogood_stores += 1
            return None
        if depth == len(order):
            return dict(assignment) if problem.is_solution(assignment) else None

        name = order[depth]
        for value in domains[name]:
            assignment[name] = value
            solved = dfs(depth + 1, assignment)
            if solved is not None:
                return solved
            assignment.pop(name, None)

        stats.backtracks += 1
        stats.dead_end_signatures.append(tuple(sorted(assignment.items())))
        if nogood_store is not None:
            nogood_store.add(assignment)
            stats.nogood_stores += 1
        return None

    solution = dfs(0, {})
    return SearchResult(solution=solution, verified=problem.is_solution(solution) if solution is not None else True, stats=stats)


def solve_hybrid(
    problem: CanonicalProblemState,
    *,
    nogood_store: EpisodeNogoodStore | None = None,
) -> SearchResult:
    stats = SearchStats()
    base_order = problem.variable_names

    def dfs(assignment: dict[str, int]) -> dict[str, int] | None:
        stats.nodes += 1
        if nogood_store is not None:
            stats.nogood_checks += 1
            if nogood_store.matches(assignment):
                stats.nogood_hits += 1
                return None
        propagated = propagate(problem, assignment)
        stats.propagation_checks += propagated.stats.constraint_checks
        if not propagated.consistent:
            stats.backtracks += 1
            stats.dead_end_signatures.append(tuple(sorted(assignment.items())))
            if nogood_store is not None:
                nogood_store.add(assignment)
                stats.nogood_stores += 1
            return None

        inferred = dict(assignment)
        for name, values in propagated.domains.items():
            if len(values) == 1:
                inferred[name] = values[0]
        if len(inferred) == len(base_order):
            if problem.is_solution(inferred):
                return inferred
            stats.backtracks += 1
            return None

        candidates = [name for name in base_order if name not in inferred]
        name = min(candidates, key=lambda n: len(propagated.domains[n]))
        for value in propagated.domains[name]:
            branch = dict(inferred)
            branch[name] = value
            solved = dfs(branch)
            if solved is not None:
                return solved
        stats.backtracks += 1
        stats.dead_end_signatures.append(tuple(sorted(assignment.items())))
        if nogood_store is not None:
            nogood_store.add(assignment)
            stats.nogood_stores += 1
        return None

    solution = dfs({})
    return SearchResult(solution=solution, verified=problem.is_solution(solution) if solution is not None else True, stats=stats)
