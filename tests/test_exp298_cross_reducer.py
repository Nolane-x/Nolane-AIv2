from __future__ import annotations

from copy import deepcopy
import importlib

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp298_paired_runner import run_exp298_root
from nolane_ai.protocol.evidence import canonical_sha256


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_cross_reducer")


def _root(index: int, *, max_exact_probes: int = 4096):
    return run_exp298_root(
        root_seed=f"20260913-exp298-cross-test-{index}",
        canonical_index=index,
        eval_replicates=1,
        eval_start_replicate=60000,
        d_model=8,
        hidden_size=8,
        target_parameters=20000,
        max_exact_probes=max_exact_probes,
        protocol_digest="a" * 64,
        geometry_digest="b" * 64,
        code_digest="c" * 64,
        repository_head="d" * 40,
    )


def test_cross_reducer_authorizes_only_on_four_of_four_established_roots():
    m = _module()
    roots = [_root(index) for index in range(4)]
    payload = m.reduce_exp298_roots(roots)
    assert payload["schema"] == "NLM-EXP-298-CROSS-DOMAIN-FIDELITY-CROSS-V1"
    assert payload["canonical_indices"] == [0, 1, 2, 3]
    assert payload["decision"] == "CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT"
    assert payload["successor_design_authorized"] is True
    assert payload["authorization_scope"] == "DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY"
    assert payload["exp300_authorized"] is False
    assert len(payload["source_root_artifact_digests"]) == 4
    assert m.validate_exp298_cross(payload) == []


def test_cross_reducer_keeps_successor_closed_when_one_validated_root_is_negative(monkeypatch):
    m = _module()
    runner = importlib.import_module("nolane_ai.experiments.exp298_paired_runner")
    roots = [_root(index) for index in range(4)]
    negative = deepcopy(roots[-1])
    negative["decision"] = "CROSS_DOMAIN_FIDELITY_TRANSFER_NOT_ESTABLISHED"
    clean = deepcopy(negative)
    clean.pop("artifact_digest", None)
    negative["artifact_digest"] = canonical_sha256(clean)
    roots[-1] = negative

    # This unit test isolates reducer decision semantics. Root receipt integrity and
    # reconstruction are independently enforced by validate_exp298_root tests.
    monkeypatch.setattr(runner, "validate_exp298_root", lambda payload: [])
    payload = m.reduce_exp298_roots(roots)
    assert payload["decision"] == "CROSS_DOMAIN_FIDELITY_TRANSFER_NOT_RECURRENT"
    assert payload["successor_design_authorized"] is False
    assert payload["authorization_scope"] == "NONE"
    assert payload["exp300_authorized"] is False


def test_cross_reducer_rejects_missing_duplicate_and_identity_drift_before_science():
    m = _module()
    roots = [_root(index) for index in range(4)]
    with pytest.raises(ValueError, match="exactly four"):
        m.reduce_exp298_roots(roots[:3])
    duplicate = [roots[0], roots[1], roots[2], roots[2]]
    with pytest.raises(ValueError, match="canonical indices"):
        m.reduce_exp298_roots(duplicate)

    drifted = deepcopy(roots)
    drifted[3]["protocol_digest"] = "e" * 64
    clean = deepcopy(drifted[3])
    clean.pop("artifact_digest", None)
    drifted[3]["artifact_digest"] = canonical_sha256(clean)
    with pytest.raises(ValueError, match="identity drift"):
        m.reduce_exp298_roots(drifted)

    geometry_drifted = roots[:3] + [_root(3, max_exact_probes=1)]
    assert geometry_drifted[-1]["decision"] == "CROSS_DOMAIN_FIDELITY_TRANSFER_NOT_ESTABLISHED"
    with pytest.raises(ValueError, match="identity drift"):
        m.reduce_exp298_roots(geometry_drifted)


def test_cross_receipt_tamper_is_detected_after_rehash():
    m = _module()
    payload = m.reduce_exp298_roots([_root(index) for index in range(4)])
    payload["authorization_scope"] = "DESIGN_EXP300"
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    payload["artifact_digest"] = canonical_sha256(clean)
    errors = m.validate_exp298_cross(payload)
    assert any("authorization" in error for error in errors), errors
