from __future__ import annotations

import importlib
import importlib.util

import pytest


MODULE = "nolane_ai.experiments.exp301_identity"
PREREG_V2 = "1660a728990290f1c605941d531be1d52fe82591c619a327d104e4b87c1e6bad"


def _identity_module():
    spec = importlib.util.find_spec(MODULE)
    assert spec is not None, "EXP-301 identity authority must live in a torch-free module"
    return importlib.import_module(MODULE)


def _frozen(**changes):
    identity = _identity_module()
    values = dict(
        source_commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        prereg_semantic_digest=PREREG_V2,
        architecture_receipts_digest="c" * 64,
        scientific_execution_contract_digest="d" * 64,
        train_generator_digest="e" * 64,
        development_generator_digest="f" * 64,
        challenge_generator_digest="1" * 64,
        parameter_audit_digest="2" * 64,
        compute_ledger_version="exp301-flops-v2",
        analysis_digest="3" * 64,
        workflow_digest="4" * 64,
    )
    values.update(changes)
    return identity.build_frozen_implementation_identity(**values)


def test_predata_identity_is_rootless_and_binds_only_frozen_authority() -> None:
    identity = _identity_module()
    frozen = _frozen()
    assert frozen.schema == "EXP301-FROZEN-IMPLEMENTATION-IDENTITY-V1"
    assert frozen.prereg_semantic_digest == PREREG_V2
    assert frozen.compute_ledger_version == "exp301-flops-v2"
    assert not hasattr(frozen, "root")
    assert not hasattr(frozen, "selected_hyperparameter_receipt_digest")
    assert not hasattr(frozen, "challenge_beacon_digest")
    assert frozen.frozen_implementation_digest == identity.canonical_frozen_implementation_digest(frozen)


def test_predata_identity_fails_closed_on_prereg_compute_or_hex_drift() -> None:
    with pytest.raises(ValueError, match="prereg"):
        _frozen(prereg_semantic_digest="0" * 64)
    with pytest.raises(ValueError, match="compute ledger"):
        _frozen(compute_ledger_version="exp301-flops-v1")
    with pytest.raises(ValueError, match="source_commit_sha"):
        _frozen(source_commit_sha="abc")


def test_runtime_root_identity_adds_only_post_selection_challenge_authority() -> None:
    identity = _identity_module()
    frozen = _frozen()
    runtime = identity.build_runtime_root_identity(
        frozen_implementation_digest=frozen.frozen_implementation_digest,
        root=2,
        selected_hyperparameter_receipt_digest="5" * 64,
        challenge_beacon_digest="6" * 64,
        challenge_materialization_digest="7" * 64,
        scientific_evidence_eligible=True,
    )
    assert runtime.schema == "EXP301-RUNTIME-ROOT-IDENTITY-V1"
    assert runtime.root == 2
    assert runtime.frozen_implementation_digest == frozen.frozen_implementation_digest
    assert runtime.run_identity == identity.canonical_runtime_root_identity_digest(runtime)
    assert runtime.scientific_evidence_eligible is True


def test_runtime_identity_rejects_bad_root_and_unbound_challenge() -> None:
    identity = _identity_module()
    frozen = _frozen()
    kwargs = dict(
        frozen_implementation_digest=frozen.frozen_implementation_digest,
        selected_hyperparameter_receipt_digest="5" * 64,
        challenge_beacon_digest="6" * 64,
        challenge_materialization_digest="7" * 64,
        scientific_evidence_eligible=True,
    )
    with pytest.raises(ValueError, match="root"):
        identity.build_runtime_root_identity(root=9, **kwargs)
    with pytest.raises(ValueError, match="challenge_beacon_digest"):
        identity.build_runtime_root_identity(root=0, **{**kwargs, "challenge_beacon_digest": ""})


def test_runtime_manifest_requires_exactly_four_unique_roots() -> None:
    identity = _identity_module()
    frozen = _frozen()
    roots = [
        identity.build_runtime_root_identity(
            frozen_implementation_digest=frozen.frozen_implementation_digest,
            root=root,
            selected_hyperparameter_receipt_digest=f"{root + 5:x}" * 64,
            challenge_beacon_digest="a" * 64,
            challenge_materialization_digest=f"{root + 1:x}" * 64,
            scientific_evidence_eligible=True,
        )
        for root in range(4)
    ]
    manifest = identity.build_runtime_identity_manifest(roots)
    assert manifest.roots == (0, 1, 2, 3)
    assert len(set(manifest.run_identities)) == 4
    assert manifest.frozen_implementation_digest == frozen.frozen_implementation_digest

    with pytest.raises(ValueError, match="exactly roots"):
        identity.build_runtime_identity_manifest(roots[:3])
    with pytest.raises(ValueError, match="unique"):
        identity.build_runtime_identity_manifest([roots[0], roots[0], roots[2], roots[3]])


def test_identity_module_does_not_import_torch() -> None:
    identity = _identity_module()
    assert "torch" not in identity.__dict__
