from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    domain: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name must be non-empty")
        if not self.domain:
            raise ValueError(f"variable {self.name!r} requires a non-empty domain")
        if len(set(self.domain)) != len(self.domain):
            raise ValueError(f"variable {self.name!r} domain values must be unique")


@dataclass(frozen=True, slots=True)
class TableConstraint:
    name: str
    scope: tuple[str, ...]
    allowed: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name must be non-empty")
        if not self.scope:
            raise ValueError(f"constraint {self.name!r} requires a non-empty scope")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError(f"constraint {self.name!r} scope cannot repeat variables")
        if any(len(row) != len(self.scope) for row in self.allowed):
            raise ValueError(f"constraint {self.name!r} has malformed allowed tuple")

    def is_satisfied(self, assignment: Mapping[str, int]) -> bool:
        if not all(name in assignment for name in self.scope):
            return True
        row = tuple(assignment[name] for name in self.scope)
        return row in self.allowed

    def has_support(self, variable: str, value: int, domains: Mapping[str, tuple[int, ...]]) -> tuple[bool, int]:
        if variable not in self.scope:
            raise KeyError(variable)
        checks = 0
        index = self.scope.index(variable)
        for row in self.allowed:
            checks += 1
            if row[index] != value:
                continue
            if all(row[i] in domains[name] for i, name in enumerate(self.scope)):
                return True, checks
        return False, checks


@dataclass(frozen=True, slots=True)
class CanonicalProblemState:
    variables: tuple[Variable, ...]
    constraints: tuple[TableConstraint, ...]
    oracle_conflict_variables: tuple[str, ...] = ()
    world_id: str = ""

    def __post_init__(self) -> None:
        names = [v.name for v in self.variables]
        if not names:
            raise ValueError("problem requires at least one variable")
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique")
        known = set(names)
        constraint_names = [c.name for c in self.constraints]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("constraint names must be unique")
        for constraint in self.constraints:
            unknown = set(constraint.scope) - known
            if unknown:
                raise ValueError(f"constraint {constraint.name!r} references unknown variables {sorted(unknown)}")
        if set(self.oracle_conflict_variables) - known:
            raise ValueError("oracle conflict variables must exist in problem")

    @property
    def domains(self) -> dict[str, tuple[int, ...]]:
        return {v.name: v.domain for v in self.variables}

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    def is_solution(self, assignment: Mapping[str, int] | None) -> bool:
        if assignment is None:
            return False
        if set(assignment) != set(self.variable_names):
            return False
        domains = self.domains
        if any(value not in domains[name] for name, value in assignment.items()):
            return False
        return all(constraint.is_satisfied(assignment) for constraint in self.constraints)

    def enumerate_assignments(self, *, limit: int | None = None):
        count = 1
        for variable in self.variables:
            count *= len(variable.domain)
        if limit is not None and count > limit:
            raise ValueError(f"assignment space {count} exceeds exact limit {limit}")
        names = self.variable_names
        for values in product(*(self.domains[name] for name in names)):
            yield dict(zip(names, values, strict=True))
