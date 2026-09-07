from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

from .cps import CanonicalProblemState

FidelityDecision = Literal["court_accept", "court_reject", "court_inconclusive"]
FidelityDirection = Literal["source_to_candidate", "candidate_to_source", "structural_mismatch"]


def compile_valid(problem: CanonicalProblemState) -> bool:
    # Construction already enforces syntactic/domain validity.
    return bool(problem.variables) and all(constraint.scope for constraint in problem.constraints)


def canonical_problem_payload(problem: CanonicalProblemState) -> dict[str, object]:
    return {
        "variables": [
            {"name": variable.name, "domain": list(variable.domain)}
            for variable in problem.variables
        ],
        "constraints": [
            {
                "name": constraint.name,
                "scope": list(constraint.scope),
                "allowed": [list(row) for row in constraint.allowed],
            }
            for constraint in problem.constraints
        ],
    }


def canonical_problem_digest(problem: CanonicalProblemState) -> str:
    encoded = json.dumps(
        canonical_problem_payload(problem),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def semantic_signature(problem: CanonicalProblemState, *, max_assignments: int) -> frozenset[tuple[tuple[str, int], ...]]:
    solutions: set[tuple[tuple[str, int], ...]] = set()
    for assignment in problem.enumerate_assignments(limit=max_assignments):
        if problem.is_solution(assignment):
            solutions.add(tuple(sorted(assignment.items())))
    return frozenset(solutions)


@dataclass(frozen=True, slots=True)
class FidelityWitness:
    direction: FidelityDirection
    assignment: tuple[tuple[str, int], ...]
    source_accepts: bool
    candidate_accepts: bool
    source_constraint_checks: int
    candidate_constraint_checks: int

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "assignment": {name: value for name, value in self.assignment},
            "source_accepts": self.source_accepts,
            "candidate_accepts": self.candidate_accepts,
            "source_constraint_checks": self.source_constraint_checks,
            "candidate_constraint_checks": self.candidate_constraint_checks,
        }


@dataclass(frozen=True, slots=True)
class FidelityCourtReceipt:
    decision: FidelityDecision
    witness: FidelityWitness | None
    assignments_enumerated: int
    assignment_space_size: int
    source_constraint_checks: int
    candidate_constraint_checks: int
    termination_reason: str
    semantic_verification_operations: int
    source_digest: str
    candidate_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "witness": self.witness.as_dict() if self.witness is not None else None,
            "assignments_enumerated": self.assignments_enumerated,
            "assignment_space_size": self.assignment_space_size,
            "source_constraint_checks": self.source_constraint_checks,
            "candidate_constraint_checks": self.candidate_constraint_checks,
            "termination_reason": self.termination_reason,
            "semantic_verification_operations": self.semantic_verification_operations,
            "source_digest": self.source_digest,
            "candidate_digest": self.candidate_digest,
        }


def _assignment_space_size(problem: CanonicalProblemState) -> int:
    size = 1
    for variable in problem.variables:
        size *= len(variable.domain)
    return size


def _evaluate(problem: CanonicalProblemState, assignment: dict[str, int]) -> tuple[bool, int]:
    checks = 0
    domains = problem.domains
    if set(assignment) != set(problem.variable_names):
        return False, checks
    for name, value in assignment.items():
        checks += 1
        if value not in domains[name]:
            return False, checks
    for constraint in problem.constraints:
        checks += 1
        if not constraint.is_satisfied(assignment):
            return False, checks
    return True, checks


class FidelityCourt:
    def __init__(self, *, max_exact_assignments: int = 4096) -> None:
        if max_exact_assignments <= 0:
            raise ValueError("max_exact_assignments must be positive")
        self.max_exact_assignments = max_exact_assignments

    def adjudicate(self, source: CanonicalProblemState, candidate: CanonicalProblemState) -> FidelityCourtReceipt:
        source_digest = canonical_problem_digest(source)
        candidate_digest = canonical_problem_digest(candidate)
        if source.variable_names != candidate.variable_names or source.domains != candidate.domains:
            witness = FidelityWitness(
                direction="structural_mismatch",
                assignment=(),
                source_accepts=False,
                candidate_accepts=False,
                source_constraint_checks=0,
                candidate_constraint_checks=0,
            )
            return FidelityCourtReceipt(
                decision="court_reject",
                witness=witness,
                assignments_enumerated=0,
                assignment_space_size=max(_assignment_space_size(source), _assignment_space_size(candidate)),
                source_constraint_checks=0,
                candidate_constraint_checks=0,
                termination_reason="variable_or_domain_structure_mismatch",
                semantic_verification_operations=2,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
            )

        assignment_space_size = _assignment_space_size(source)
        if assignment_space_size > self.max_exact_assignments:
            return FidelityCourtReceipt(
                decision="court_inconclusive",
                witness=None,
                assignments_enumerated=0,
                assignment_space_size=assignment_space_size,
                source_constraint_checks=0,
                candidate_constraint_checks=0,
                termination_reason="assignment_space_exceeds_exact_ceiling",
                semantic_verification_operations=1,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
            )

        assignments_enumerated = 0
        source_checks_total = 0
        candidate_checks_total = 0
        for assignment in source.enumerate_assignments(limit=self.max_exact_assignments):
            assignments_enumerated += 1
            source_accepts, source_checks = _evaluate(source, assignment)
            candidate_accepts, candidate_checks = _evaluate(candidate, assignment)
            source_checks_total += source_checks
            candidate_checks_total += candidate_checks
            if source_accepts != candidate_accepts:
                direction: FidelityDirection = (
                    "source_to_candidate" if source_accepts else "candidate_to_source"
                )
                witness = FidelityWitness(
                    direction=direction,
                    assignment=tuple(sorted(assignment.items())),
                    source_accepts=source_accepts,
                    candidate_accepts=candidate_accepts,
                    source_constraint_checks=source_checks,
                    candidate_constraint_checks=candidate_checks,
                )
                operations = assignments_enumerated + source_checks_total + candidate_checks_total + 1
                return FidelityCourtReceipt(
                    decision="court_reject",
                    witness=witness,
                    assignments_enumerated=assignments_enumerated,
                    assignment_space_size=assignment_space_size,
                    source_constraint_checks=source_checks_total,
                    candidate_constraint_checks=candidate_checks_total,
                    termination_reason="bidirectional_semantic_witness_found",
                    semantic_verification_operations=operations,
                    source_digest=source_digest,
                    candidate_digest=candidate_digest,
                )

        operations = assignments_enumerated + source_checks_total + candidate_checks_total
        return FidelityCourtReceipt(
            decision="court_accept",
            witness=None,
            assignments_enumerated=assignments_enumerated,
            assignment_space_size=assignment_space_size,
            source_constraint_checks=source_checks_total,
            candidate_constraint_checks=candidate_checks_total,
            termination_reason="exact_bidirectional_equivalence_closed",
            semantic_verification_operations=operations,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
        )

    def accept(self, source: CanonicalProblemState, candidate: CanonicalProblemState) -> bool:
        return self.adjudicate(source, candidate).decision == "court_accept"
