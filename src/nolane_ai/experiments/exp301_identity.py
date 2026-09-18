from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from .exp301_compute import COMPUTE_LEDGER_VERSION


EXP301_PREREG_V2_DIGEST = "1660a728990290f1c605941d531be1d52fe82591c619a327d104e4b87c1e6bad"
FROZEN_IDENTITY_SCHEMA = "EXP301-FROZEN-IMPLEMENTATION-IDENTITY-V1"
RUNTIME_ROOT_IDENTITY_SCHEMA = "EXP301-RUNTIME-ROOT-IDENTITY-V1"
RUNTIME_MANIFEST_SCHEMA = "EXP301-RUNTIME-IDENTITY-MANIFEST-V1"
EXP301_ROOTS = (0, 1, 2, 3)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class FrozenImplementationIdentity:
    schema: str
    source_commit_sha: str
    source_tree_digest: str
    prereg_semantic_digest: str
    architecture_receipts_digest: str
    scientific_execution_contract_digest: str
    train_generator_digest: str
    development_generator_digest: str
    challenge_generator_digest: str
    parameter_audit_digest: str
    compute_ledger_version: str
    analysis_digest: str
    workflow_digest: str
    frozen_implementation_digest: str


@dataclass(frozen=True, slots=True)
class RuntimeRootIdentity:
    schema: str
    frozen_implementation_digest: str
    root: int
    selected_hyperparameter_receipt_digest: str
    challenge_beacon_digest: str
    challenge_materialization_digest: str
    scientific_evidence_eligible: bool
    run_identity: str


@dataclass(frozen=True, slots=True)
class RuntimeIdentityManifest:
    schema: str
    frozen_implementation_digest: str
    roots: tuple[int, ...]
    run_identities: tuple[str, ...]
    manifest_digest: str


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _require_hex(value: str, *, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")


def _frozen_payload(identity: FrozenImplementationIdentity) -> dict[str, object]:
    payload = asdict(identity)
    payload.pop("frozen_implementation_digest", None)
    return payload


def canonical_frozen_implementation_digest(identity: FrozenImplementationIdentity) -> str:
    return _canonical_digest(_frozen_payload(identity))


def build_frozen_implementation_identity(
    *,
    source_commit_sha: str,
    source_tree_digest: str,
    prereg_semantic_digest: str,
    architecture_receipts_digest: str,
    scientific_execution_contract_digest: str,
    train_generator_digest: str,
    development_generator_digest: str,
    challenge_generator_digest: str,
    parameter_audit_digest: str,
    compute_ledger_version: str,
    analysis_digest: str,
    workflow_digest: str,
) -> FrozenImplementationIdentity:
    _require_hex(source_commit_sha, length=40, field="source_commit_sha")
    for field, value in (
        ("source_tree_digest", source_tree_digest),
        ("architecture_receipts_digest", architecture_receipts_digest),
        ("scientific_execution_contract_digest", scientific_execution_contract_digest),
        ("train_generator_digest", train_generator_digest),
        ("development_generator_digest", development_generator_digest),
        ("challenge_generator_digest", challenge_generator_digest),
        ("parameter_audit_digest", parameter_audit_digest),
        ("analysis_digest", analysis_digest),
        ("workflow_digest", workflow_digest),
    ):
        _require_hex(value, length=64, field=field)
    if prereg_semantic_digest != EXP301_PREREG_V2_DIGEST:
        raise ValueError("prereg semantic digest does not match frozen EXP-301 V2 authority")
    if compute_ledger_version != COMPUTE_LEDGER_VERSION:
        raise ValueError(
            f"compute ledger version must be {COMPUTE_LEDGER_VERSION}, got {compute_ledger_version!r}"
        )

    values = dict(
        schema=FROZEN_IDENTITY_SCHEMA,
        source_commit_sha=source_commit_sha,
        source_tree_digest=source_tree_digest,
        prereg_semantic_digest=prereg_semantic_digest,
        architecture_receipts_digest=architecture_receipts_digest,
        scientific_execution_contract_digest=scientific_execution_contract_digest,
        train_generator_digest=train_generator_digest,
        development_generator_digest=development_generator_digest,
        challenge_generator_digest=challenge_generator_digest,
        parameter_audit_digest=parameter_audit_digest,
        compute_ledger_version=compute_ledger_version,
        analysis_digest=analysis_digest,
        workflow_digest=workflow_digest,
    )
    provisional = FrozenImplementationIdentity(**values, frozen_implementation_digest="")
    return FrozenImplementationIdentity(
        **values,
        frozen_implementation_digest=canonical_frozen_implementation_digest(provisional),
    )


def _runtime_payload(identity: RuntimeRootIdentity) -> dict[str, object]:
    payload = asdict(identity)
    payload.pop("run_identity", None)
    return payload


def canonical_runtime_root_identity_digest(identity: RuntimeRootIdentity) -> str:
    return _canonical_digest(_runtime_payload(identity))


def build_runtime_root_identity(
    *,
    frozen_implementation_digest: str,
    root: int,
    selected_hyperparameter_receipt_digest: str,
    challenge_beacon_digest: str,
    challenge_materialization_digest: str,
    scientific_evidence_eligible: bool,
) -> RuntimeRootIdentity:
    _require_hex(frozen_implementation_digest, length=64, field="frozen_implementation_digest")
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    for field, value in (
        ("selected_hyperparameter_receipt_digest", selected_hyperparameter_receipt_digest),
        ("challenge_beacon_digest", challenge_beacon_digest),
        ("challenge_materialization_digest", challenge_materialization_digest),
    ):
        _require_hex(value, length=64, field=field)
    if not isinstance(scientific_evidence_eligible, bool):
        raise ValueError("scientific_evidence_eligible must be boolean")

    values = dict(
        schema=RUNTIME_ROOT_IDENTITY_SCHEMA,
        frozen_implementation_digest=frozen_implementation_digest,
        root=root,
        selected_hyperparameter_receipt_digest=selected_hyperparameter_receipt_digest,
        challenge_beacon_digest=challenge_beacon_digest,
        challenge_materialization_digest=challenge_materialization_digest,
        scientific_evidence_eligible=scientific_evidence_eligible,
    )
    provisional = RuntimeRootIdentity(**values, run_identity="")
    return RuntimeRootIdentity(
        **values,
        run_identity=canonical_runtime_root_identity_digest(provisional),
    )


def build_runtime_identity_manifest(
    identities: Iterable[RuntimeRootIdentity],
) -> RuntimeIdentityManifest:
    materialized = tuple(identities)
    roots = tuple(identity.root for identity in materialized)
    if len(set(roots)) != len(roots):
        raise ValueError("runtime identity roots must be unique")
    if set(roots) != set(EXP301_ROOTS) or len(roots) != len(EXP301_ROOTS):
        raise ValueError(f"runtime identity manifest must contain exactly roots {EXP301_ROOTS}")

    frozen_digests = {identity.frozen_implementation_digest for identity in materialized}
    if len(frozen_digests) != 1:
        raise ValueError("all runtime identities must bind the same frozen implementation digest")
    for identity in materialized:
        if identity.run_identity != canonical_runtime_root_identity_digest(identity):
            raise ValueError(f"runtime root {identity.root} run identity digest mismatch")

    ordered = tuple(sorted(materialized, key=lambda identity: identity.root))
    frozen_digest = ordered[0].frozen_implementation_digest
    run_identities = tuple(identity.run_identity for identity in ordered)
    payload = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "frozen_implementation_digest": frozen_digest,
        "roots": EXP301_ROOTS,
        "run_identities": run_identities,
    }
    return RuntimeIdentityManifest(
        schema=RUNTIME_MANIFEST_SCHEMA,
        frozen_implementation_digest=frozen_digest,
        roots=EXP301_ROOTS,
        run_identities=run_identities,
        manifest_digest=_canonical_digest(payload),
    )
