from __future__ import annotations

from dataclasses import dataclass
import importlib


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_behavioral_fidelity")


@dataclass(frozen=True)
class _Problem:
    values: tuple[int, ...]


class _Semantics:
    domain_id = "toy"

    def __init__(self, *, probes: int = 3) -> None:
        self._count = probes

    def probes(self):
        m = _module()
        return tuple(m.BehavioralProbe(probe_id=f"p{i}", payload=(i,)) for i in range(self._count))

    def evaluate(self, problem, probe):
        m = _module()
        i = int(probe.payload[0])
        return m.BehavioralOutcome(accepted=True, value=problem.values[i])

    def canonical_digest(self, problem):
        return "toy:" + ",".join(str(v) for v in problem.values)


def test_exact_equivalence_accepts_only_after_complete_probe_space():
    m = _module()
    source = _Problem((1, 2, 3))
    receipt = m.BehavioralFidelityCourt(max_exact_probes=8).adjudicate(
        source, source, _Semantics()
    )
    assert receipt.decision == "court_accept"
    assert receipt.complete_probe_space is True
    assert receipt.probes_evaluated == 3
    assert receipt.witness is None


def test_directional_difference_emits_canonical_witness():
    m = _module()
    source = _Problem((1, 2, 3))
    candidate = _Problem((1, 9, 3))
    receipt = m.BehavioralFidelityCourt(max_exact_probes=8).adjudicate(
        source, candidate, _Semantics()
    )
    assert receipt.decision == "court_reject"
    assert receipt.complete_probe_space is False
    assert receipt.probes_evaluated == 2
    assert receipt.witness == {
        "direction": "source_to_candidate",
        "probe_id": "p1",
        "probe_payload": [1],
        "source_outcome": {"accepted": True, "value": 2},
        "candidate_outcome": {"accepted": True, "value": 9},
    }


def test_probe_ceiling_is_fail_closed_inconclusive():
    m = _module()
    source = _Problem((1, 2, 3, 4))
    receipt = m.BehavioralFidelityCourt(max_exact_probes=3).adjudicate(
        source, source, _Semantics(probes=4)
    )
    assert receipt.decision == "court_inconclusive"
    assert receipt.complete_probe_space is False
    assert receipt.probes_evaluated == 0
    assert receipt.termination_reason == "probe_space_exceeds_exact_limit"


def test_receipt_digest_is_stable_and_tamper_sensitive():
    m = _module()
    source = _Problem((4, 5, 6))
    receipt = m.BehavioralFidelityCourt(max_exact_probes=8).adjudicate(
        source, source, _Semantics()
    )
    payload = receipt.as_dict()
    assert payload["receipt_digest"] == receipt.receipt_digest
    assert m.validate_behavioral_fidelity_receipt(payload) == []
    tampered = dict(payload)
    tampered["probes_evaluated"] = 2
    assert "behavioral fidelity receipt digest mismatch" in m.validate_behavioral_fidelity_receipt(tampered)
