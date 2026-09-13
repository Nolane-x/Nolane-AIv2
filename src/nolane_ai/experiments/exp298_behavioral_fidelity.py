from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from nolane_ai.protocol.evidence import canonical_sha256


@dataclass(frozen=True, slots=True)
class BehavioralProbe:
    probe_id: str
    payload: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")


@dataclass(frozen=True, slots=True)
class BehavioralOutcome:
    accepted: bool
    value: Any


class BehavioralSemantics(Protocol):
    domain_id: str

    def probes(self) -> tuple[BehavioralProbe, ...]: ...

    def evaluate(self, problem: object, probe: BehavioralProbe) -> BehavioralOutcome: ...

    def canonical_digest(self, problem: object) -> str: ...


@dataclass(frozen=True, slots=True)
class BehavioralFidelityReceipt:
    decision: str
    domain_id: str
    source_digest: str
    candidate_digest: str
    complete_probe_space: bool
    probes_evaluated: int
    semantic_domain_operations: int
    termination_reason: str
    witness: dict[str, Any] | None
    receipt_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _outcome_dict(outcome: BehavioralOutcome) -> dict[str, Any]:
    return {"accepted": bool(outcome.accepted), "value": outcome.value}


def _digest_payload(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("receipt_digest", None)
    return canonical_sha256(clean)


def _direction(source: BehavioralOutcome, candidate: BehavioralOutcome) -> str:
    if source.accepted and not candidate.accepted:
        return "source_to_candidate"
    if candidate.accepted and not source.accepted:
        return "candidate_to_source"
    return "source_to_candidate"


class BehavioralFidelityCourt:
    def __init__(self, *, max_exact_probes: int = 4096) -> None:
        if max_exact_probes <= 0:
            raise ValueError("max_exact_probes must be positive")
        self.max_exact_probes = int(max_exact_probes)

    def adjudicate(
        self,
        source: object,
        candidate: object,
        semantics: BehavioralSemantics,
    ) -> BehavioralFidelityReceipt:
        probes = tuple(semantics.probes())
        source_digest = semantics.canonical_digest(source)
        candidate_digest = semantics.canonical_digest(candidate)

        if len(probes) > self.max_exact_probes:
            payload = {
                "decision": "court_inconclusive",
                "domain_id": semantics.domain_id,
                "source_digest": source_digest,
                "candidate_digest": candidate_digest,
                "complete_probe_space": False,
                "probes_evaluated": 0,
                "semantic_domain_operations": 0,
                "termination_reason": "probe_space_exceeds_exact_limit",
                "witness": None,
            }
            return BehavioralFidelityReceipt(
                **payload,
                receipt_digest=_digest_payload(payload),
            )

        operations = 0
        for index, probe in enumerate(probes, start=1):
            source_outcome = semantics.evaluate(source, probe)
            candidate_outcome = semantics.evaluate(candidate, probe)
            operations += 2
            if source_outcome != candidate_outcome:
                witness = {
                    "direction": _direction(source_outcome, candidate_outcome),
                    "probe_id": probe.probe_id,
                    "probe_payload": list(probe.payload),
                    "source_outcome": _outcome_dict(source_outcome),
                    "candidate_outcome": _outcome_dict(candidate_outcome),
                }
                payload = {
                    "decision": "court_reject",
                    "domain_id": semantics.domain_id,
                    "source_digest": source_digest,
                    "candidate_digest": candidate_digest,
                    "complete_probe_space": False,
                    "probes_evaluated": index,
                    "semantic_domain_operations": operations,
                    "termination_reason": "directional_behavioral_witness",
                    "witness": witness,
                }
                return BehavioralFidelityReceipt(
                    **payload,
                    receipt_digest=_digest_payload(payload),
                )

        payload = {
            "decision": "court_accept",
            "domain_id": semantics.domain_id,
            "source_digest": source_digest,
            "candidate_digest": candidate_digest,
            "complete_probe_space": True,
            "probes_evaluated": len(probes),
            "semantic_domain_operations": operations,
            "termination_reason": "exact_behavioral_equivalence_closed",
            "witness": None,
        }
        return BehavioralFidelityReceipt(
            **payload,
            receipt_digest=_digest_payload(payload),
        )


def validate_behavioral_fidelity_receipt(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("decision") not in {
        "court_accept",
        "court_reject",
        "court_inconclusive",
    }:
        errors.append("behavioral fidelity receipt decision invalid")
    if not isinstance(payload.get("domain_id"), str) or not payload.get("domain_id"):
        errors.append("behavioral fidelity receipt domain missing")
    if payload.get("receipt_digest") != _digest_payload(payload):
        errors.append("behavioral fidelity receipt digest mismatch")
    return errors
