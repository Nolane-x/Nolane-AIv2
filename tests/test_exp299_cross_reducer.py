from __future__ import annotations

from copy import deepcopy
import importlib
from functools import lru_cache

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp299_native_runner import run_exp299_root
from nolane_ai.protocol.evidence import canonical_sha256


def _module():
    return importlib.import_module("nolane_ai.experiments.exp299_cross_reducer")


@lru_cache(maxsize=8)
def _cached_root(index: int, max_exact_probes: int = 4096):
    return run_exp299_root(
        root_seed=f"20260913-exp299-cross-test-{index}",
        canonical_index=index,
        fit_replicates=1,
        fit_start_replicate=0,
        eval_replicates=1,
        eval_start_replicate=10_000,
        d_model=8,
        hidden_size=8,
        target_parameters=12_000,
        batch_size=16,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        authority_threshold=0.5,
        max_exact_probes=max_exact_probes,
        protocol_digest="a" * 64,
        geometry_digest="b" * 64,
        code_digest="c" * 64,
        repository_head="d" * 40,
    )


def _root(index: int, *, max_exact_probes: int = 4096):
    return deepcopy(_cached_root(index, max_exact_probes))


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    payload["artifact_digest"] = canonical_sha256(clean)
    return payload


def test_cross_reducer_authorizes_only_on_four_of_four_positive_roots(monkeypatch):
    m = _module()
    runner = importlib.import_module("nolane_ai.experiments.exp299_native_runner")
    roots = [_root(index) for index in range(4)]
    for root in roots:
        root["decision"] = "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL"
        _rehash(root)

    # This unit test isolates reducer decision semantics. Root scientific receipt
    # reconstruction is independently enforced by validate_exp299_root tests.
    monkeypatch.setattr(runner, "validate_exp299_root", lambda payload: [])
    payload = m.reduce_exp299_roots(roots)
    assert payload["schema"] == "NLM-EXP-299-NATIVE-FIDELITY-CROSS-V1"
    assert payload["canonical_indices"] == [0, 1, 2, 3]
    assert payload["decision"] == "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL_RECURRENT"
    assert payload["successor_design_authorized"] is True
    assert payload["authorization_scope"] == "DESIGN_EXP300_INTEGRATED_COURT_ONLY"
    assert payload["exp300_execution_authorized"] is False
    assert len(payload["source_root_artifact_digests"]) == 4
    assert m.validate_exp299_cross(payload) == []


def test_cross_reducer_keeps_exp300_closed_for_real_negative_or_mixed_roots(monkeypatch):
    m = _module()
    roots = [_root(index) for index in range(4)]
    payload = m.reduce_exp299_roots(roots)
    assert payload["decision"] == "NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT"
    assert payload["successor_design_authorized"] is False
    assert payload["authorization_scope"] == "NONE"
    assert payload["exp300_execution_authorized"] is False
    assert m.validate_exp299_cross(payload) == []

    runner = importlib.import_module("nolane_ai.experiments.exp299_native_runner")
    mixed = [_root(index) for index in range(4)]
    mixed[0]["decision"] = "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL"
    _rehash(mixed[0])
    monkeypatch.setattr(runner, "validate_exp299_root", lambda candidate: [])
    mixed_payload = m.reduce_exp299_roots(mixed)
    assert mixed_payload["decision"] == "NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT"
    assert mixed_payload["successor_design_authorized"] is False


def test_cross_reducer_rejects_missing_duplicate_and_identity_drift_before_science():
    m = _module()
    roots = [_root(index) for index in range(4)]
    with pytest.raises(ValueError, match="exactly four"):
        m.reduce_exp299_roots(roots[:3])
    with pytest.raises(ValueError, match="canonical indices"):
        m.reduce_exp299_roots([roots[0], roots[1], roots[2], roots[2]])

    drifted = deepcopy(roots)
    drifted[3]["protocol_digest"] = "e" * 64
    _rehash(drifted[3])
    with pytest.raises(ValueError, match="identity drift"):
        m.reduce_exp299_roots(drifted)

    # Identity-drift coverage must not create a scientifically invalid root.
    # Mutate only sealed configuration metadata after a valid root exists, then
    # rehash it. The root validator accepts the internally coherent receipt and
    # the cross reducer must reject its mismatch against the other three roots.
    geometry_drifted = deepcopy(roots)
    geometry_drifted[3]["config"]["batch_size"] = 17
    _rehash(geometry_drifted[3])
    with pytest.raises(ValueError, match="identity drift"):
        m.reduce_exp299_roots(geometry_drifted)


def test_cross_receipt_tamper_is_detected_even_after_rehash():
    m = _module()
    payload = m.reduce_exp299_roots([_root(index) for index in range(4)])
    payload["authorization_scope"] = "DESIGN_EXP300_UNFROZEN"
    _rehash(payload)
    errors = m.validate_exp299_cross(payload)
    assert any("authorization" in error for error in errors), errors
