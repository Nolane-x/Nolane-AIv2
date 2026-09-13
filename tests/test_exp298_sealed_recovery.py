from __future__ import annotations

from copy import deepcopy
import importlib
import math

import pytest

pytest.importorskip("torch")

from nolane_ai.protocol.evidence import canonical_sha256


IDENTITY = {
    "repository_head": "4" * 40,
    "protocol_digest": "1" * 64,
    "geometry_digest": "2" * 64,
    "code_digest": "3" * 64,
}
CONFIG = {
    "eval_replicates": 1,
    "eval_start_replicate": 60000,
    "d_model": 16,
    "hidden_size": 16,
    "target_parameters": 50000,
    "max_exact_probes": 4096,
    "domains": [
        "code_invariant",
        "causal_diagnosis",
        "grounded_language_ambiguity",
    ],
    "candidates_per_replicate": 16,
}


def _runner():
    return importlib.import_module("nolane_ai.experiments.exp298_paired_runner")


def _recovery():
    return importlib.import_module("nolane_ai.experiments.exp298_sealed_recovery")


def _root(index: int):
    return _runner().run_exp298_root(
        root_seed=f"20260913-exp298-sealed-recovery-root-{index}",
        canonical_index=index,
        eval_replicates=CONFIG["eval_replicates"],
        eval_start_replicate=CONFIG["eval_start_replicate"],
        d_model=CONFIG["d_model"],
        hidden_size=CONFIG["hidden_size"],
        target_parameters=CONFIG["target_parameters"],
        max_exact_probes=CONFIG["max_exact_probes"],
        protocol_digest=IDENTITY["protocol_digest"],
        geometry_digest=IDENTITY["geometry_digest"],
        code_digest=IDENTITY["code_digest"],
        repository_head=IDENTITY["repository_head"],
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    payload["artifact_digest"] = canonical_sha256(clean)
    return payload


def test_sealed_validator_accepts_scientific_root_without_reconstruction() -> None:
    recovery = _recovery()
    root = _root(0)
    assert recovery.validate_exp298_sealed_root(
        root,
        expected_identity=IDENTITY,
        expected_config=CONFIG,
    ) == []


def test_sealed_validator_does_not_make_neural_diagnostics_causal() -> None:
    recovery = _recovery()
    root = deepcopy(_root(0))
    for row in root["evaluation"]["raw_candidates"]:
        for arm in row["arms"].values():
            arm["fidelity_score"] = 0.123456789
            arm["authority_score"] = 0.234567891
            arm["verifier_score"] = 0.345678912
    _rehash(root)
    assert recovery.validate_exp298_sealed_root(
        root,
        expected_identity=IDENTITY,
        expected_config=CONFIG,
    ) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda p: p["evaluation"]["raw_candidates"][0].__setitem__(
                "is_faithful", not p["evaluation"]["raw_candidates"][0]["is_faithful"]
            ),
            "domain metrics mismatch",
        ),
        (
            lambda p: p["evaluation"]["raw_candidates"][0]["court_receipt"].__setitem__(
                "decision", "court_accept"
            ),
            "court/authority mismatch",
        ),
        (
            lambda p: p["evaluation"]["raw_candidates"][0]["arms"][
                "cross_domain_fidelity_fabric"
            ].__setitem__("semantic_domain_operations", 0),
            "cost ledger mismatch",
        ),
        (
            lambda p: p["evaluation"]["domains"]["code_invariant"].__setitem__(
                "balanced_accuracy_gain", 0.99
            ),
            "domain metrics mismatch",
        ),
        (
            lambda p: p.__setitem__("repository_head", "5" * 40),
            "identity mismatch",
        ),
    ],
)
def test_sealed_validator_rejects_causal_or_identity_tampering(mutate, expected) -> None:
    recovery = _recovery()
    root = deepcopy(_root(0))
    mutate(root)
    _rehash(root)
    errors = recovery.validate_exp298_sealed_root(
        root,
        expected_identity=IDENTITY,
        expected_config=CONFIG,
    )
    assert any(expected in error for error in errors), errors


def test_sealed_validator_requires_finite_diagnostics() -> None:
    recovery = _recovery()
    root = deepcopy(_root(0))
    root["evaluation"]["raw_candidates"][0]["arms"]["compile_only_control"][
        "authority_score"
    ] = math.nan
    _rehash(root)
    errors = recovery.validate_exp298_sealed_root(
        root,
        expected_identity=IDENTITY,
        expected_config=CONFIG,
    )
    assert any("diagnostic score invalid" in error for error in errors), errors


def test_four_sealed_roots_recover_frozen_cross_decision_without_model_rerun() -> None:
    recovery = _recovery()
    roots = [_root(index) for index in range(4)]
    receipt = recovery.recover_exp298_cross_from_sealed_roots(
        roots,
        expected_identity=IDENTITY,
        expected_config=CONFIG,
        source_development_run_id=123456,
        source_cross_failure="ROOT_RECONSTRUCTION_MISMATCH_CROSS_PROCESS",
        artifact_manifest_digest="6" * 64,
    )
    assert receipt["decision"] == "CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT"
    assert receipt["established_root_count"] == 4
    assert receipt["successor_design_authorized"] is True
    assert receipt["authorization_scope"] == "DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY"
    assert receipt["exp300_authorized"] is False
    assert receipt["recovery_mode"] == "SEALED_ROOT_FORENSIC_REDUCTION"
    assert receipt["source_development_run_id"] == 123456
    assert receipt["artifact_manifest_digest"] == "6" * 64
    assert importlib.import_module(
        "nolane_ai.experiments.exp298_cross_reducer"
    ).validate_exp298_cross(receipt) == []
