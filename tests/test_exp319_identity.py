from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from nolane_ai.experiments.exp319_identity import (
    APPROVED_PREREGISTRATION_DIGEST,
    EXP301_CROSS_ROOT_ARTIFACT_DIGEST,
    EXP301R_REPAIR_PROVENANCE_DIGEST,
    EXP319_IDENTITY_SCHEMA,
    build_exp319_execution_identity,
    canonical_exp319_identity_json_bytes,
    source_tree_digest_from_git_tree_sha,
)


def _h(ch: str, n: int) -> str:
    return ch * n


def _identity():
    return build_exp319_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        generator_contract_digest=_h("c", 64),
        training_contract_digest=_h("d", 64),
        selection_contract_digest=_h("e", 64),
        scoring_contract_digest=_h("f", 64),
        workflow_sha256=_h("1", 64),
    )


def test_source_tree_digest_is_domain_separated_and_git_tree_bound() -> None:
    first = source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert len(first) == 64
    assert first == source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert first != source_tree_digest_from_git_tree_sha(_h("b", 40))
    with pytest.raises(ValueError, match="git tree"):
        source_tree_digest_from_git_tree_sha("bad")


def test_execution_identity_binds_parent_science_preregistration_and_geometry() -> None:
    identity = _identity()

    assert identity.schema == EXP319_IDENTITY_SCHEMA
    assert identity.approved_preregistration_digest == APPROVED_PREREGISTRATION_DIGEST
    assert identity.parent_exp301_cross_root_artifact_digest == EXP301_CROSS_ROOT_ARTIFACT_DIGEST
    assert identity.parent_exp301r_repair_provenance_digest == EXP301R_REPAIR_PROVENANCE_DIGEST
    assert identity.shard_count == 16
    assert identity.worlds_per_shard == 32
    assert identity.device == "cpu"
    assert len(identity.exp319_execution_digest) == 64

    assert APPROVED_PREREGISTRATION_DIGEST == "0e5871316a69522ee762ff1f1684a5735cec86f2c0a4b036ecb5584ac07ac056"
    assert EXP301_CROSS_ROOT_ARTIFACT_DIGEST == "8ca0ac7a8c2a278914efafb6390a913fcd95c346df8e01d94539304fc40e4286"
    assert EXP301R_REPAIR_PROVENANCE_DIGEST == "d6491ecb3808350f4ced31d854324ba718dfc6eef5cb30ef5d8d0e5195380884"


def test_execution_digest_changes_when_any_frozen_contract_changes() -> None:
    base = _identity()
    mutated = (
        build_exp319_execution_identity(
            source_commit_sha=_h("9", 40),
            git_tree_sha=_h("b", 40),
            generator_contract_digest=_h("c", 64),
            training_contract_digest=_h("d", 64),
            selection_contract_digest=_h("e", 64),
            scoring_contract_digest=_h("f", 64),
            workflow_sha256=_h("1", 64),
        ),
        build_exp319_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("9", 40),
            generator_contract_digest=_h("c", 64),
            training_contract_digest=_h("d", 64),
            selection_contract_digest=_h("e", 64),
            scoring_contract_digest=_h("f", 64),
            workflow_sha256=_h("1", 64),
        ),
        build_exp319_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("b", 40),
            generator_contract_digest=_h("2", 64),
            training_contract_digest=_h("d", 64),
            selection_contract_digest=_h("e", 64),
            scoring_contract_digest=_h("f", 64),
            workflow_sha256=_h("1", 64),
        ),
        build_exp319_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("b", 40),
            generator_contract_digest=_h("c", 64),
            training_contract_digest=_h("2", 64),
            selection_contract_digest=_h("e", 64),
            scoring_contract_digest=_h("f", 64),
            workflow_sha256=_h("1", 64),
        ),
        build_exp319_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("b", 40),
            generator_contract_digest=_h("c", 64),
            training_contract_digest=_h("d", 64),
            selection_contract_digest=_h("2", 64),
            scoring_contract_digest=_h("f", 64),
            workflow_sha256=_h("1", 64),
        ),
        build_exp319_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("b", 40),
            generator_contract_digest=_h("c", 64),
            training_contract_digest=_h("d", 64),
            selection_contract_digest=_h("e", 64),
            scoring_contract_digest=_h("2", 64),
            workflow_sha256=_h("1", 64),
        ),
        build_exp319_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("b", 40),
            generator_contract_digest=_h("c", 64),
            training_contract_digest=_h("d", 64),
            selection_contract_digest=_h("e", 64),
            scoring_contract_digest=_h("f", 64),
            workflow_sha256=_h("2", 64),
        ),
    )
    assert all(item.exp319_execution_digest != base.exp319_execution_digest for item in mutated)


def test_canonical_identity_json_is_compact_sorted_and_digest_checked() -> None:
    identity = _identity()
    payload = canonical_exp319_identity_json_bytes(identity)
    assert payload.endswith(b"\n")
    assert b"  " not in payload
    assert payload == canonical_exp319_identity_json_bytes(identity)
    assert b'"exp319_execution_digest"' in payload
    assert set(asdict(identity)) == {
        "schema",
        "approved_preregistration_digest",
        "parent_exp301_cross_root_artifact_digest",
        "parent_exp301r_repair_provenance_digest",
        "source_commit_sha",
        "source_tree_digest",
        "generator_contract_digest",
        "training_contract_digest",
        "selection_contract_digest",
        "scoring_contract_digest",
        "workflow_sha256",
        "shard_count",
        "worlds_per_shard",
        "device",
        "exp319_execution_digest",
    }

    with pytest.raises(ValueError, match="execution digest"):
        canonical_exp319_identity_json_bytes(replace(identity, exp319_execution_digest="0" * 64))


def test_identity_rejects_malformed_sha_and_contract_digests() -> None:
    kwargs = dict(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        generator_contract_digest=_h("c", 64),
        training_contract_digest=_h("d", 64),
        selection_contract_digest=_h("e", 64),
        scoring_contract_digest=_h("f", 64),
        workflow_sha256=_h("1", 64),
    )
    with pytest.raises(ValueError, match="source commit"):
        build_exp319_execution_identity(**(kwargs | {"source_commit_sha": "bad"}))
    with pytest.raises(ValueError, match="generator"):
        build_exp319_execution_identity(**(kwargs | {"generator_contract_digest": "bad"}))
    with pytest.raises(ValueError, match="workflow"):
        build_exp319_execution_identity(**(kwargs | {"workflow_sha256": "bad"}))
