from __future__ import annotations

import inspect

import pytest

from nolane_ai.experiments.exp301_freeze import (
    EXP301_MARKER_PATHS,
    architecture_receipts_digest,
    build_freeze_identity_from_components,
    generator_contract_digest,
    parameter_audit_payload,
    source_tree_digest_from_git_tree_sha,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp301_identity import EXP301_PREREG_V2_DIGEST


def test_source_tree_digest_binds_git_tree_object_without_materializing_files() -> None:
    left = source_tree_digest_from_git_tree_sha("a" * 40)
    right = source_tree_digest_from_git_tree_sha("b" * 40)
    assert len(left) == 64
    assert len(right) == 64
    assert left != right
    with pytest.raises(ValueError, match="git tree"):
        source_tree_digest_from_git_tree_sha("not-a-tree")


def test_architecture_and_parameter_constitution_are_frozen_and_exact_10m() -> None:
    payload = parameter_audit_payload()
    assert payload["total_budget"] == 10_000_000
    assert payload["frozen_base"] == 9_120_832
    assert payload["capacity_exchange_envelope"] == 879_168
    assert payload["arm_trainable_parameters"] == {
        "A_FIXED": 10_000_000,
        "B_LOOP_SIMPLE": 10_000_000,
        "C_NRS_CORE": 10_000_000,
    }
    assert payload["dead_capacity_forbidden"] is True
    assert len(architecture_receipts_digest()) == 64


def test_generator_contract_digest_is_split_specific_and_challenge_stays_unmaterialized() -> None:
    blob = "c" * 40
    train = generator_contract_digest(source_blob_sha=blob, split="train", per_family=128)
    dev = generator_contract_digest(source_blob_sha=blob, split="development", per_family=64)
    challenge = generator_contract_digest(source_blob_sha=blob, split="challenge", per_family=128)
    assert len({train, dev, challenge}) == 3
    signature = inspect.signature(generator_contract_digest)
    assert "challenge_beacon" not in signature.parameters
    assert "challenge_nonce" not in signature.parameters


def test_freeze_identity_builder_binds_every_required_component() -> None:
    identity = build_freeze_identity_from_components(
        source_commit_sha="d" * 40,
        git_tree_sha="e" * 40,
        worlds_blob_sha="f" * 40,
        analysis_blob_shas=("1" * 40, "2" * 40, "3" * 40),
        workflow_blob_sha="4" * 40,
    )
    assert identity.source_commit_sha == "d" * 40
    assert identity.prereg_semantic_digest == EXP301_PREREG_V2_DIGEST
    assert identity.train_generator_digest != identity.development_generator_digest
    assert identity.development_generator_digest != identity.challenge_generator_digest
    assert len(identity.frozen_implementation_digest) == 64


def test_marker_commit_may_change_only_the_two_execution_identity_files() -> None:
    assert set(EXP301_MARKER_PATHS) == {
        "protocols/v017/exp301_execution_identity_v1.json",
        "protocols/v017/exp301_execution_identity_v1.sha256",
    }
    validate_marker_changed_paths(EXP301_MARKER_PATHS)
    with pytest.raises(ValueError, match="marker-only"):
        validate_marker_changed_paths((*EXP301_MARKER_PATHS, "src/nolane_ai/model/v017_recursive.py"))
