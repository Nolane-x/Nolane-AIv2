from __future__ import annotations

from copy import deepcopy
import importlib

import pytest

pytest.importorskip("torch")

from nolane_ai.protocol.evidence import canonical_sha256


def _module():
    return importlib.import_module("nolane_ai.experiments.exp298_paired_runner")


def _root():
    return _module().run_exp298_root(
        root_seed="20260913-exp298-tamper-root",
        canonical_index=1,
        eval_replicates=1,
        eval_start_replicate=60000,
        d_model=16,
        hidden_size=16,
        target_parameters=50000,
        max_exact_probes=4096,
        protocol_digest="1" * 64,
        geometry_digest="2" * 64,
        code_digest="3" * 64,
        repository_head="4" * 40,
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    payload["artifact_digest"] = canonical_sha256(clean)
    return payload


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda p: p["evaluation"]["raw_candidates"][0].__setitem__("is_faithful", not p["evaluation"]["raw_candidates"][0]["is_faithful"]),
            "reconstruction mismatch",
        ),
        (
            lambda p: p["evaluation"]["raw_candidates"][0]["court_receipt"].__setitem__("decision", "court_accept"),
            "reconstruction mismatch",
        ),
        (
            lambda p: p["evaluation"]["raw_candidates"][0]["arms"]["cross_domain_fidelity_fabric"].__setitem__("semantic_domain_operations", 0),
            "reconstruction mismatch",
        ),
        (
            lambda p: p["evaluation"]["domains"]["code_invariant"].__setitem__("balanced_accuracy_gain", 0.99),
            "reconstruction mismatch",
        ),
        (
            lambda p: p.__setitem__("unrestricted_semantic_authority_claimed", True),
            "forbidden flag enabled",
        ),
        (
            lambda p: p.__setitem__("successor_design_authorized", True),
            "root successor authority must remain closed",
        ),
    ],
)
def test_rehashed_tampering_is_rejected(mutate, expected):
    m = _module()
    payload = deepcopy(_root())
    mutate(payload)
    _rehash(payload)
    errors = m.validate_exp298_root(payload)
    assert any(expected in error for error in errors), errors


def test_candidate_drop_is_rejected_even_after_rehash():
    m = _module()
    payload = deepcopy(_root())
    payload["evaluation"]["raw_candidates"].pop()
    payload["evaluation_candidate_count"] -= 1
    _rehash(payload)
    errors = m.validate_exp298_root(payload)
    assert any("reconstruction mismatch" in error for error in errors), errors
