from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import random

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralOutcome, BehavioralProbe
from nolane_ai.protocol.evidence import canonical_sha256

EXPECTED_STRATA = (
    "faithful_equivalent",
    "edge_reversal",
    "mediator_omission",
    "wrong_intervention_target",
    "effect_polarity_flip",
    "exogenous_binding_swap",
    "observationally_equivalent_interventionally_wrong",
    "spurious_direct_edge",
)


@dataclass(frozen=True, slots=True)
class EquationSpec:
    op: str
    args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CausalDiagnosisProblem:
    exogenous_names: tuple[str, str]
    endogenous_names: tuple[str, str, str]
    equations: tuple[tuple[str, EquationSpec], ...]
    intervention_alias: tuple[tuple[str, str], ...]
    diagnosis_target: str


@dataclass(frozen=True, slots=True)
class CausalCandidateCase:
    candidate_id: str
    stratum: str
    is_faithful: bool
    candidate: CausalDiagnosisProblem
    candidate_digest: str

    def arm_view(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "candidate": self.candidate,
        }


@dataclass(frozen=True, slots=True)
class CausalWorldBatch:
    seed: int
    source: CausalDiagnosisProblem
    source_digest: str
    candidates: tuple[CausalCandidateCase, ...]


def canonical_causal_problem_digest(problem: CausalDiagnosisProblem) -> str:
    return canonical_sha256(
        {
            "exogenous_names": list(problem.exogenous_names),
            "endogenous_names": list(problem.endogenous_names),
            "equations": [
                [name, {"op": spec.op, "args": list(spec.args)}]
                for name, spec in problem.equations
            ],
            "intervention_alias": [list(item) for item in problem.intervention_alias],
            "diagnosis_target": problem.diagnosis_target,
        }
    )


def _value(name: str, env: dict[str, int]) -> int:
    if name not in env:
        raise ValueError(f"unavailable causal input: {name}")
    return int(env[name])


def _eval_equation(spec: EquationSpec, env: dict[str, int]) -> int:
    values = tuple(_value(name, env) for name in spec.args)
    if spec.op == "identity" and len(values) == 1:
        return values[0]
    if spec.op == "not" and len(values) == 1:
        return 1 - values[0]
    if spec.op == "xor" and len(values) == 2:
        return values[0] ^ values[1]
    if spec.op == "and" and len(values) == 2:
        return values[0] & values[1]
    if spec.op == "or" and len(values) == 2:
        return values[0] | values[1]
    raise ValueError(f"invalid causal equation: {spec}")


def _run_scm(
    problem: CausalDiagnosisProblem,
    *,
    u: int,
    v: int,
    intervention_target: str,
    intervention_value: int,
) -> tuple[int, int, int]:
    env: dict[str, int] = {"U": int(u), "V": int(v)}
    aliases = dict(problem.intervention_alias)
    actual_target = aliases.get(intervention_target, intervention_target)
    for name, spec in problem.equations:
        if intervention_target != "none" and name == actual_target:
            env[name] = int(intervention_value)
        else:
            env[name] = _eval_equation(spec, env)
    return tuple(env[name] for name in problem.endogenous_names)


class CausalDiagnosisSemantics:
    domain_id = "causal_diagnosis"

    def __init__(self, source: CausalDiagnosisProblem) -> None:
        probes: list[BehavioralProbe] = []
        index = 0
        interventions: tuple[tuple[str, int], ...] = (
            ("none", -1),
            ("A", 0),
            ("A", 1),
            ("B", 0),
            ("B", 1),
        )
        for u, v in product((0, 1), repeat=2):
            for target, value in interventions:
                probes.append(
                    BehavioralProbe(
                        probe_id=f"probe-{index:03d}",
                        payload=(u, v, target, value),
                    )
                )
                index += 1
        self._probes = tuple(probes)

    def probes(self) -> tuple[BehavioralProbe, ...]:
        return self._probes

    def evaluate(
        self, problem: CausalDiagnosisProblem, probe: BehavioralProbe
    ) -> BehavioralOutcome:
        u, v, target, value = probe.payload
        assignment = _run_scm(
            problem,
            u=int(u),
            v=int(v),
            intervention_target=str(target),
            intervention_value=int(value),
        )
        target_index = problem.endogenous_names.index(problem.diagnosis_target)
        return BehavioralOutcome(
            accepted=True,
            value=assignment + (assignment[target_index],),
        )

    def canonical_digest(self, problem: CausalDiagnosisProblem) -> str:
        return canonical_causal_problem_digest(problem)


def _source_problem(seed: int) -> CausalDiagnosisProblem:
    # Seed affects candidate ordering/identity but not the frozen causal family.
    _ = seed
    return CausalDiagnosisProblem(
        exogenous_names=("U", "V"),
        endogenous_names=("A", "B", "C"),
        equations=(
            ("A", EquationSpec("identity", ("U",))),
            ("B", EquationSpec("xor", ("A", "V"))),
            ("C", EquationSpec("and", ("A", "B"))),
        ),
        intervention_alias=(("A", "A"), ("B", "B")),
        diagnosis_target="C",
    )


def _replace(
    source: CausalDiagnosisProblem,
    *,
    equations: tuple[tuple[str, EquationSpec], ...] | None = None,
    aliases: tuple[tuple[str, str], ...] | None = None,
) -> CausalDiagnosisProblem:
    return CausalDiagnosisProblem(
        source.exogenous_names,
        source.endogenous_names,
        source.equations if equations is None else equations,
        source.intervention_alias if aliases is None else aliases,
        source.diagnosis_target,
    )


def _wrong_candidate(source: CausalDiagnosisProblem, stratum: str) -> CausalDiagnosisProblem:
    if stratum == "edge_reversal":
        return _replace(
            source,
            equations=(
                ("A", EquationSpec("identity", ("V",))),
                ("B", EquationSpec("xor", ("A", "U"))),
                ("C", EquationSpec("and", ("A", "B"))),
            ),
        )
    if stratum == "mediator_omission":
        return _replace(
            source,
            equations=source.equations[:2] + (("C", EquationSpec("identity", ("A",))),),
        )
    if stratum == "wrong_intervention_target":
        return _replace(source, aliases=(("A", "B"), ("B", "A")))
    if stratum == "effect_polarity_flip":
        return _replace(
            source,
            equations=source.equations[:2] + (("C", EquationSpec("not", ("B",))),),
        )
    if stratum == "exogenous_binding_swap":
        return _replace(
            source,
            equations=(
                ("A", EquationSpec("identity", ("V",))),
                source.equations[1],
                source.equations[2],
            ),
        )
    if stratum == "observationally_equivalent_interventionally_wrong":
        # Since source A=U observationally, B=U xor V matches B=A xor V on
        # every passive observation, but fails under interventions on A.
        return _replace(
            source,
            equations=(
                source.equations[0],
                ("B", EquationSpec("xor", ("U", "V"))),
                source.equations[2],
            ),
        )
    if stratum == "spurious_direct_edge":
        return _replace(
            source,
            equations=source.equations[:2] + (("C", EquationSpec("xor", ("A", "B"))),),
        )
    raise ValueError(f"unknown causal stratum: {stratum}")


def generate_exp298_causal_world(seed: int) -> CausalWorldBatch:
    source = _source_problem(seed)
    source_digest = canonical_causal_problem_digest(source)
    drafts: list[tuple[str, bool, CausalDiagnosisProblem, str]] = []

    for _ in range(8):
        drafts.append(("faithful_equivalent", True, source, source_digest))

    wrong_strata = EXPECTED_STRATA[1:] + (
        "observationally_equivalent_interventionally_wrong",
    )
    for stratum in wrong_strata:
        candidate = _wrong_candidate(source, stratum)
        drafts.append(
            (stratum, False, candidate, canonical_causal_problem_digest(candidate))
        )

    order_rng = random.Random(seed ^ 0x298CA55)
    order_rng.shuffle(drafts)
    cases: list[CausalCandidateCase] = []
    for slot, (stratum, is_faithful, candidate, digest) in enumerate(drafts):
        nonce = order_rng.getrandbits(128)
        opaque = sha256(f"causal|{seed}|{slot}|{nonce}".encode("ascii")).hexdigest()[:20]
        cases.append(
            CausalCandidateCase(
                candidate_id=f"case-{opaque}",
                stratum=stratum,
                is_faithful=is_faithful,
                candidate=candidate,
                candidate_digest=digest,
            )
        )

    return CausalWorldBatch(
        seed=seed,
        source=source,
        source_digest=source_digest,
        candidates=tuple(cases),
    )
