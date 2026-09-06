from __future__ import annotations

from .cps import CanonicalProblemState


def compile_valid(problem: CanonicalProblemState) -> bool:
    # Construction already enforces syntactic/domain validity.
    return bool(problem.variables) and all(constraint.scope for constraint in problem.constraints)


def semantic_signature(problem: CanonicalProblemState, *, max_assignments: int) -> frozenset[tuple[tuple[str, int], ...]]:
    solutions: set[tuple[tuple[str, int], ...]] = set()
    for assignment in problem.enumerate_assignments(limit=max_assignments):
        if problem.is_solution(assignment):
            solutions.add(tuple(sorted(assignment.items())))
    return frozenset(solutions)


class FidelityCourt:
    def __init__(self, *, max_exact_assignments: int = 4096) -> None:
        if max_exact_assignments <= 0:
            raise ValueError("max_exact_assignments must be positive")
        self.max_exact_assignments = max_exact_assignments

    def accept(self, source: CanonicalProblemState, candidate: CanonicalProblemState) -> bool:
        if source.variable_names != candidate.variable_names:
            return False
        if source.domains != candidate.domains:
            return False
        return semantic_signature(source, max_assignments=self.max_exact_assignments) == semantic_signature(
            candidate, max_assignments=self.max_exact_assignments
        )
