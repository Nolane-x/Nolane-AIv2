from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

EXP319_IDENTITY_SCHEMA = "EXP319-LEARNABILITY-FOUNDATION-EXECUTION-IDENTITY-V1"
APPROVED_PREREGISTRATION_DIGEST = "0e5871316a69522ee762ff1f1684a5735cec86f2c0a4b036ecb5584ac07ac056"
EXP301_CROSS_ROOT_ARTIFACT_DIGEST = "8ca0ac7a8c2a278914efafb6390a913fcd95c346df8e01d94539304fc40e4286"
EXP301R_REPAIR_PROVENANCE_DIGEST = "d6491ecb3808350f4ced31d854324ba718dfc6eef5cb30ef5d8d0e5195380884"
EXP319_SHARD_COUNT = 16
EXP319_WORLDS_PER_SHARD = 32
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp319ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_exp301_cross_root_artifact_digest: str
    parent_exp301r_repair_provenance_digest: str
    source_commit_sha: str
    source_tree_digest: str
    generator_contract_digest: str
    training_contract_digest: str
    selection_contract_digest: str
    scoring_contract_digest: str
    workflow_sha256: str
    shard_count: int
    worlds_per_shard: int
    device: str
    exp319_execution_digest: str


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _require_hex(value: str, *, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")


def source_tree_digest_from_git_tree_sha(git_tree_sha: str) -> str:
    _require_hex(git_tree_sha, length=40, field="git tree")
    return hashlib.sha256(f"EXP319-GIT-TREE-V1|{git_tree_sha}".encode("ascii")).hexdigest()


def canonical_exp319_execution_digest(identity: Exp319ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp319_execution_digest", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_exp319_execution_identity(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    generator_contract_digest: str,
    training_contract_digest: str,
    selection_contract_digest: str,
    scoring_contract_digest: str,
    workflow_sha256: str,
) -> Exp319ExecutionIdentity:
    _require_hex(source_commit_sha, length=40, field="source commit")
    _require_hex(generator_contract_digest, length=64, field="generator contract digest")
    _require_hex(training_contract_digest, length=64, field="training contract digest")
    _require_hex(selection_contract_digest, length=64, field="selection contract digest")
    _require_hex(scoring_contract_digest, length=64, field="scoring contract digest")
    _require_hex(workflow_sha256, length=64, field="workflow sha256")

    values = {
        "schema": EXP319_IDENTITY_SCHEMA,
        "approved_preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "parent_exp301_cross_root_artifact_digest": EXP301_CROSS_ROOT_ARTIFACT_DIGEST,
        "parent_exp301r_repair_provenance_digest": EXP301R_REPAIR_PROVENANCE_DIGEST,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest_from_git_tree_sha(git_tree_sha),
        "generator_contract_digest": generator_contract_digest,
        "training_contract_digest": training_contract_digest,
        "selection_contract_digest": selection_contract_digest,
        "scoring_contract_digest": scoring_contract_digest,
        "workflow_sha256": workflow_sha256,
        "shard_count": EXP319_SHARD_COUNT,
        "worlds_per_shard": EXP319_WORLDS_PER_SHARD,
        "device": "cpu",
    }
    provisional = Exp319ExecutionIdentity(**values, exp319_execution_digest="")
    return Exp319ExecutionIdentity(
        **values,
        exp319_execution_digest=canonical_exp319_execution_digest(provisional),
    )


def canonical_exp319_identity_json_bytes(identity: Exp319ExecutionIdentity) -> bytes:
    if identity.exp319_execution_digest != canonical_exp319_execution_digest(identity):
        raise ValueError("EXP-319 execution digest mismatch")
    return _canonical_bytes(asdict(identity)) + b"\n"
