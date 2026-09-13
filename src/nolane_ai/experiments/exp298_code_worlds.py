from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import random

from nolane_ai.experiments.exp298_behavioral_fidelity import (
    BehavioralOutcome,
    BehavioralProbe,
)
from nolane_ai.protocol.evidence import canonical_sha256

EXPECTED_STRATA = (
    "faithful_equivalent",
    "off_by_one_boundary",
    "stale_variable_reference",
    "branch_omission",
    "variable_binding_swap",
    "predicate_polarity_flip",
    "invariant_strengthen",
    "invariant_weaken",
    "incorrect_post_state_reference",
)

State = tuple[int, int, int]
Transition = tuple[State, State]


@dataclass(frozen=True, slots=True)
class CodeInvariantProblem:
    variable_names: tuple[str, str, str]
    domains: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    transitions: tuple[Transition, ...]
    invariant_states: tuple[State, ...]


@dataclass(frozen=True, slots=True)
class CodeInvariantCandidateCase:
    candidate_id: str
    stratum: str
    is_faithful: bool
    candidate: CodeInvariantProblem
    candidate_digest: str

    def arm_view(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "candidate": self.candidate,
        }


@dataclass(frozen=True, slots=True)
class CodeInvariantWorldBatch:
    seed: int
    source: CodeInvariantProblem
    source_digest: str
    candidates: tuple[CodeInvariantCandidateCase, ...]


def canonical_code_problem_digest(problem: CodeInvariantProblem) -> str:
    return canonical_sha256(
        {
            "variable_names": list(problem.variable_names),
            "domains": [list(domain) for domain in problem.domains],
            "transitions": [[list(pre), list(post)] for pre, post in problem.transitions],
            "invariant_states": [list(state) for state in problem.invariant_states],
        }
    )


class CodeInvariantSemantics:
    domain_id = "code_invariant"

    def __init__(self, source: CodeInvariantProblem) -> None:
        self._source = source
        states = tuple(product(*source.domains))
        self._probes = tuple(
            BehavioralProbe(probe_id=f"state-{index:03d}", payload=tuple(state))
            for index, state in enumerate(states)
        )

    def probes(self) -> tuple[BehavioralProbe, ...]:
        return self._probes

    def evaluate(
        self, problem: CodeInvariantProblem, probe: BehavioralProbe
    ) -> BehavioralOutcome:
        pre = tuple(int(value) for value in probe.payload)
        table = dict(problem.transitions)
        if pre not in table:
            return BehavioralOutcome(accepted=False, value=("missing_transition",))
        post = table[pre]
        invariant_ok = post in set(problem.invariant_states)
        return BehavioralOutcome(
            accepted=invariant_ok,
            value=tuple(post) + (int(invariant_ok),),
        )

    def canonical_digest(self, problem: CodeInvariantProblem) -> str:
        return canonical_code_problem_digest(problem)


def _base_problem(seed: int) -> CodeInvariantProblem:
    rng = random.Random(seed)
    domains = ((0, 1, 2), (0, 1, 2), (0, 1))
    shift = 1 + rng.randrange(2)
    transitions: list[Transition] = []
    for x, y, flag in product(*domains):
        if flag:
            post = ((x + y + shift) % 3, (y + 1) % 3, flag)
        else:
            post = ((x + 1) % 3, (y + x + shift) % 3, flag)
        transitions.append(((x, y, flag), post))
    transitions.sort()
    unique_posts = sorted({post for _, post in transitions})
    invariant_states = tuple(
        post for post in unique_posts if (post[0] + 2 * post[1] + post[2] + shift) % 3 != 0
    )
    return CodeInvariantProblem(
        variable_names=("x", "y", "flag"),
        domains=domains,
        transitions=tuple(transitions),
        invariant_states=invariant_states,
    )


def _replace_transition(
    problem: CodeInvariantProblem, index: int, post: State
) -> CodeInvariantProblem:
    rows = list(problem.transitions)
    pre, _ = rows[index]
    rows[index] = (pre, post)
    return CodeInvariantProblem(
        problem.variable_names,
        problem.domains,
        tuple(rows),
        problem.invariant_states,
    )


def _wrong_candidate(problem: CodeInvariantProblem, stratum: str) -> CodeInvariantProblem:
    rows = problem.transitions
    index = {
        "off_by_one_boundary": len(rows) - 1,
        "stale_variable_reference": 3,
        "branch_omission": 5,
        "variable_binding_swap": 7,
        "predicate_polarity_flip": 9,
        "incorrect_post_state_reference": 11,
    }.get(stratum)
    if index is not None:
        pre, post = rows[index]
        x, y, flag = post
        if stratum == "off_by_one_boundary":
            changed = ((x + 1) % 3, y, flag)
        elif stratum == "stale_variable_reference":
            changed = (pre[0], y, flag)
        elif stratum == "branch_omission":
            changed = ((pre[0] + 1) % 3, (pre[1] + pre[0] + 1) % 3, pre[2])
        elif stratum == "variable_binding_swap":
            changed = (y, x, flag) if x != y else ((x + 1) % 3, y, flag)
        elif stratum == "predicate_polarity_flip":
            changed = ((pre[0] + pre[1] + 1) % 3, (pre[1] + 1) % 3, pre[2])
        else:
            changed = (x, (y + 1) % 3, flag)
        if changed == post:
            changed = ((x + 1) % 3, y, flag)
        return _replace_transition(problem, index, changed)

    invariant = list(problem.invariant_states)
    observed_posts = sorted({post for _, post in rows})
    if stratum == "invariant_strengthen":
        if not invariant:
            raise RuntimeError("source invariant unexpectedly empty")
        invariant.pop(0)
    elif stratum == "invariant_weaken":
        extra = next(post for post in observed_posts if post not in set(invariant))
        invariant.append(extra)
        invariant.sort()
    else:
        raise ValueError(f"unknown code stratum: {stratum}")
    return CodeInvariantProblem(
        problem.variable_names,
        problem.domains,
        problem.transitions,
        tuple(invariant),
    )


def generate_exp298_code_world(seed: int) -> CodeInvariantWorldBatch:
    source = _base_problem(seed)
    drafts: list[tuple[str, bool, CodeInvariantProblem, str]] = []
    source_digest = canonical_code_problem_digest(source)

    for slot in range(8):
        drafts.append(("faithful_equivalent", True, source, source_digest))

    for stratum in EXPECTED_STRATA[1:]:
        candidate = _wrong_candidate(source, stratum)
        drafts.append((stratum, False, candidate, canonical_code_problem_digest(candidate)))

    order_rng = random.Random(seed ^ 0x298C0DE)
    order_rng.shuffle(drafts)
    cases: list[CodeInvariantCandidateCase] = []
    for slot, (stratum, is_faithful, candidate, digest) in enumerate(drafts):
        nonce = order_rng.getrandbits(128)
        opaque = sha256(f"code|{seed}|{slot}|{nonce}".encode("ascii")).hexdigest()[:20]
        cases.append(
            CodeInvariantCandidateCase(
                candidate_id=f"case-{opaque}",
                stratum=stratum,
                is_faithful=is_faithful,
                candidate=candidate,
                candidate_digest=digest,
            )
        )

    return CodeInvariantWorldBatch(
        seed=seed,
        source=source,
        source_digest=source_digest,
        candidates=tuple(cases),
    )
