from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from .exp337_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    PARENT_EXP336_CHUNK7_ZIP_SHA256,
    PARENT_EXP336_EVIDENCE_DIGEST,
    PARENT_EXP336_FINAL_DECISION,
    PARENT_PROJECT_MODEL_STATE_DIGEST,
    PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
    PARENT_PROJECT_RNG_STATE_DIGEST,
    STARTING_CUMULATIVE_STEP,
    WORLD_IDS,
    BoundaryResult,
    aggregate_pass,
    canonical_digest,
    data_order_digest,
    reduce_full32,
    sham_equivalent,
    validate_boundary,
    world_pass_vector,
)
from .exp337_identity import Exp337ExecutionIdentity, validate_execution_identity

CHUNK_RECEIPT_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-RECEIPT-V1"
CHUNK_SUMMARY_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-CHUNK-SUMMARY-V1"
FINAL_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-FINAL-EVIDENCE-V1"

_STEMS = {
    "CONTROL_PROJECT_GOLD_PREFIX": "control",
    "SHAM_SELF_ROLLIN_MEASURE_PROJECT": "sham",
    "SELF_ROLLIN_RECOVERY_PROJECT": "recovery",
}


def _normalize_digest(value: str) -> str:
    raw = value[7:] if value.startswith("sha256:") else value
    if len(raw) != 64:
        raise ValueError("EXP-337 audit digest length")
    try:
        int(raw, 16)
    except ValueError as exc:
        raise ValueError("EXP-337 audit digest hex") from exc
    return raw.lower()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_sha256(archive: ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    try:
        handle = archive.open(name, "r")
    except KeyError as exc:
        raise ValueError(f"EXP-337 audit missing ZIP member: {name}") from exc
    with handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_member(archive: ZipFile, name: str) -> dict[str, Any]:
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise ValueError(f"EXP-337 audit missing ZIP member: {name}") from exc
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXP-337 audit JSON object required: {name}")
    return value


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    materialized = dict(receipt)
    materialized.pop("receipt_digest", None)
    return canonical_digest(materialized)


def _identity_fields(identity: Mapping[str, Any]) -> tuple[str, str]:
    try:
        materialized = Exp337ExecutionIdentity(**dict(identity))
        validate_execution_identity(materialized)
    except (TypeError, ValueError) as exc:
        raise ValueError("EXP-337 audit execution identity") from exc
    return (
        _normalize_digest(materialized.source_tree_digest),
        _normalize_digest(materialized.exp337_execution_digest),
    )


def _pass_count(boundary: BoundaryResult) -> int:
    return sum(world_pass_vector(boundary))


def audit_chunk_zip(
    zip_path: str | Path,
    *,
    identity: Mapping[str, Any],
    expected_chunk_index: int,
    expected_parent_artifact_digest: str,
) -> dict[str, Any]:
    if not 0 <= expected_chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-337 audit chunk index")

    source_tree_digest, execution_digest = _identity_fields(identity)
    expected_parent = _normalize_digest(expected_parent_artifact_digest)
    artifact_path = Path(zip_path)
    artifact_digest = _file_sha256(artifact_path)

    with ZipFile(artifact_path, "r") as archive:
        summary = _json_member(archive, "chunk-summary.json")
        if summary.get("schema") != CHUNK_SUMMARY_SCHEMA:
            raise ValueError("EXP-337 audit summary schema")
        if summary.get("chunk_index") != expected_chunk_index:
            raise ValueError("EXP-337 audit summary chunk index")
        if summary.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
            raise ValueError("EXP-337 audit preregistration digest")
        if _normalize_digest(str(summary.get("source_tree_digest"))) != source_tree_digest:
            raise ValueError("EXP-337 audit source tree digest")
        if _normalize_digest(str(summary.get("execution_digest"))) != execution_digest:
            raise ValueError("EXP-337 audit execution digest")
        if _normalize_digest(str(summary.get("parent_artifact_digest"))) != expected_parent:
            raise ValueError("EXP-337 audit parent artifact chain")

        expected_exposure = (expected_chunk_index + 1) * EXPOSURES_PER_CHUNK
        expected_updates = expected_exposure * len(WORLD_IDS)
        expected_step = STARTING_CUMULATIVE_STEP + expected_updates
        if summary.get("cumulative_exposure_per_world") != expected_exposure:
            raise ValueError("EXP-337 audit summary exposure")
        if summary.get("cumulative_source_updates") != expected_updates:
            raise ValueError("EXP-337 audit summary updates")
        if summary.get("cumulative_training_step") != expected_step:
            raise ValueError("EXP-337 audit summary cumulative step")

        for key, expected in AUTHORIZATION_FLAGS.items():
            if summary.get(key) is not expected:
                raise ValueError(f"EXP-337 audit authorization drift: {key}")

        materialized = dict(summary)
        claimed_bundle = materialized.pop("bundle_digest", None)
        if not isinstance(claimed_bundle, str) or canonical_digest(materialized) != claimed_bundle:
            raise ValueError("EXP-337 audit bundle digest")

        arms = summary.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            raise ValueError("EXP-337 audit arm coverage")

        boundaries: dict[str, BoundaryResult] = {}
        receipt_digests: dict[str, str] = {}
        checkpoint_shas: dict[str, str] = {}

        for arm in ARMS:
            row = arms[arm]
            if not isinstance(row, Mapping):
                raise ValueError("EXP-337 audit arm row")
            boundary_raw = row.get("boundary")
            files = row.get("files")
            if not isinstance(boundary_raw, Mapping) or not isinstance(files, Mapping):
                raise ValueError("EXP-337 audit arm boundary/files")
            boundary = BoundaryResult(**boundary_raw)
            validate_boundary(boundary)
            if boundary.arm != arm or boundary.chunk_index != expected_chunk_index:
                raise ValueError("EXP-337 audit boundary identity")
            boundaries[arm] = boundary

            stem = _STEMS[arm]
            checkpoint_name = f"{stem}.pt"
            receipt_name = f"{stem}-receipt.json"
            if files.get("checkpoint") != checkpoint_name or files.get("receipt") != receipt_name:
                raise ValueError("EXP-337 audit file manifest")

            checkpoint_sha = _member_sha256(archive, checkpoint_name)
            checkpoint_shas[arm] = checkpoint_sha
            if files.get("checkpoint_sha256") != checkpoint_sha:
                raise ValueError("EXP-337 audit checkpoint manifest SHA")

            receipt = _json_member(archive, receipt_name)
            if receipt.get("schema") != CHUNK_RECEIPT_SCHEMA:
                raise ValueError("EXP-337 audit receipt schema")
            if receipt.get("arm") != arm or receipt.get("chunk_index") != expected_chunk_index:
                raise ValueError("EXP-337 audit receipt chain position")
            if receipt.get("cumulative_exposure_per_world") != expected_exposure:
                raise ValueError("EXP-337 audit receipt exposure")
            if receipt.get("cumulative_source_updates") != expected_updates:
                raise ValueError("EXP-337 audit receipt updates")
            if receipt.get("cumulative_training_step") != expected_step:
                raise ValueError("EXP-337 audit receipt cumulative step")
            if receipt.get("data_order_digest") != data_order_digest(expected_exposure):
                raise ValueError("EXP-337 audit data order digest")
            if _normalize_digest(str(receipt.get("parent_artifact_digest"))) != expected_parent:
                raise ValueError("EXP-337 audit receipt parent artifact")
            if _normalize_digest(str(receipt.get("source_tree_digest"))) != source_tree_digest:
                raise ValueError("EXP-337 audit receipt source tree")
            if receipt.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
                raise ValueError("EXP-337 audit receipt preregistration")
            if _normalize_digest(str(receipt.get("execution_digest"))) != execution_digest:
                raise ValueError("EXP-337 audit receipt execution digest")
            if receipt.get("checkpoint_sha256") != checkpoint_sha:
                raise ValueError("EXP-337 audit receipt checkpoint SHA")

            for key in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
                value = receipt.get(key)
                if not isinstance(value, str):
                    raise ValueError(f"EXP-337 audit receipt state digest: {key}")
                _normalize_digest(value)
                if value != getattr(boundary, key):
                    raise ValueError(f"EXP-337 audit receipt/boundary mismatch: {key}")

            claimed_receipt = receipt.get("receipt_digest")
            if not isinstance(claimed_receipt, str) or _receipt_digest(receipt) != claimed_receipt:
                raise ValueError("EXP-337 audit receipt digest")
            if files.get("receipt_digest") != claimed_receipt:
                raise ValueError("EXP-337 audit receipt manifest digest")
            receipt_digests[arm] = claimed_receipt

            for key, expected in AUTHORIZATION_FLAGS.items():
                if receipt.get(key) is not expected:
                    raise ValueError(f"EXP-337 audit receipt authorization drift: {key}")

        if summary.get("control_sham_exact") is not True:
            raise ValueError("EXP-337 audit summary CONTROL/SHAM flag")
        if not sham_equivalent(
            boundaries["CONTROL_PROJECT_GOLD_PREFIX"],
            boundaries["SHAM_SELF_ROLLIN_MEASURE_PROJECT"],
        ):
            raise ValueError("EXP-337 audit CONTROL/SHAM mismatch")

    return {
        "chunk_index": expected_chunk_index,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_digest,
        "parent_artifact_digest": expected_parent,
        "bundle_digest": claimed_bundle,
        "cumulative_exposure_per_world": expected_exposure,
        "cumulative_source_updates": expected_updates,
        "cumulative_training_step": expected_step,
        "control_sham_exact": True,
        "pass_counts": {arm: _pass_count(boundaries[arm]) for arm in ARMS},
        "aggregate_pass": {arm: aggregate_pass(boundaries[arm]) for arm in ARMS},
        "checkpoint_sha256": checkpoint_shas,
        "receipt_digests": receipt_digests,
        "nonfinite_events": {arm: boundaries[arm].nonfinite_events for arm in ARMS},
        "self_rollin_measurement_count": {
            arm: boundaries[arm].self_rollin_measurement_count for arm in ARMS
        },
        "self_rollin_active_update_count": {
            arm: boundaries[arm].self_rollin_active_update_count for arm in ARMS
        },
        "self_rollin_divergent_update_count": {
            arm: boundaries[arm].self_rollin_divergent_update_count for arm in ARMS
        },
        "authorization_flags": dict(AUTHORIZATION_FLAGS),
        "boundaries": {arm: dict(arms[arm]["boundary"]) for arm in ARMS},
    }


def audit_chunk_chain(
    zip_paths: Sequence[str | Path],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not zip_paths:
        raise ValueError("EXP-337 audit requires at least one chunk artifact")
    if len(zip_paths) > CHUNK_COUNT:
        raise ValueError("EXP-337 audit too many chunks")

    reports: list[dict[str, Any]] = []
    parent = PARENT_EXP336_CHUNK7_ZIP_SHA256
    for index, zip_path in enumerate(zip_paths):
        report = audit_chunk_zip(
            zip_path,
            identity=identity,
            expected_chunk_index=index,
            expected_parent_artifact_digest=parent,
        )
        reports.append(report)
        parent = report["artifact_sha256"]

    return {
        "schema": "EXP337-INDEPENDENT-ARTIFACT-CHAIN-AUDIT-V1",
        "verified_chunk_count": len(reports),
        "complete_chunk_chain": len(reports) == CHUNK_COUNT,
        "first_parent_exp336_chunk7_zip_sha256": PARENT_EXP336_CHUNK7_ZIP_SHA256,
        "last_artifact_sha256": reports[-1]["artifact_sha256"],
        "chunks": reports,
        "authorization_flags": dict(AUTHORIZATION_FLAGS),
    }


def audit_final_evidence(
    final_json_path: str | Path,
    *,
    chain_report: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if chain_report.get("complete_chunk_chain") is not True:
        raise ValueError("EXP-337 final audit requires complete 8-chunk chain")
    chunks = chain_report.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != CHUNK_COUNT:
        raise ValueError("EXP-337 final audit chunk report coverage")

    payload = json.loads(Path(final_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EXP-337 final audit JSON object")
    if payload.get("schema") != FINAL_SCHEMA:
        raise ValueError("EXP-337 final audit schema")

    raw_identity = payload.get("execution_identity")
    if not isinstance(raw_identity, Mapping) or dict(raw_identity) != dict(identity):
        raise ValueError("EXP-337 final audit execution identity")
    _identity_fields(raw_identity)

    fixed_parent = {
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "parent_exp336_final_decision": PARENT_EXP336_FINAL_DECISION,
        "parent_exp336_evidence_digest": PARENT_EXP336_EVIDENCE_DIGEST,
        "parent_exp336_chunk7_zip_sha256": PARENT_EXP336_CHUNK7_ZIP_SHA256,
        "parent_project_model_state_digest": PARENT_PROJECT_MODEL_STATE_DIGEST,
        "parent_project_optimizer_state_digest": PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
        "parent_project_rng_state_digest": PARENT_PROJECT_RNG_STATE_DIGEST,
    }
    for key, expected in fixed_parent.items():
        if payload.get(key) != expected:
            raise ValueError(f"EXP-337 final audit parent/prereg drift: {key}")

    expected_bundle_digests = [row["bundle_digest"] for row in chunks]
    expected_artifact_digests = [row["artifact_sha256"] for row in chunks]
    expected_parent_digests = [row["parent_artifact_digest"] for row in chunks]
    if payload.get("chunk_bundle_digests") != expected_bundle_digests:
        raise ValueError("EXP-337 final audit chunk bundle digests")
    if payload.get("chunk_artifact_digests") != expected_artifact_digests:
        raise ValueError("EXP-337 final audit chunk artifact digests")
    if payload.get("chunk_parent_artifact_digests") != expected_parent_digests:
        raise ValueError("EXP-337 final audit parent artifact digests")
    if payload.get("control_sham_exact_by_chunk") != [True] * CHUNK_COUNT:
        raise ValueError("EXP-337 final audit CONTROL/SHAM chain")

    raw_boundaries = payload.get("final_boundaries")
    if not isinstance(raw_boundaries, Mapping) or set(raw_boundaries) != set(ARMS):
        raise ValueError("EXP-337 final audit boundary coverage")

    final_chunk_boundaries = chunks[-1].get("boundaries")
    if not isinstance(final_chunk_boundaries, Mapping):
        raise ValueError("EXP-337 final audit missing audited chunk7 boundaries")
    if {arm: dict(raw_boundaries[arm]) for arm in ARMS} != {
        arm: dict(final_chunk_boundaries[arm]) for arm in ARMS
    }:
        raise ValueError("EXP-337 final audit boundary/chunk7 mismatch")

    control = BoundaryResult(**raw_boundaries["CONTROL_PROJECT_GOLD_PREFIX"])
    sham = BoundaryResult(**raw_boundaries["SHAM_SELF_ROLLIN_MEASURE_PROJECT"])
    recovery = BoundaryResult(**raw_boundaries["SELF_ROLLIN_RECOVERY_PROJECT"])
    for boundary in (control, sham, recovery):
        validate_boundary(boundary)

    decision, vectors = reduce_full32(
        control,
        sham,
        recovery,
        parent_authority_valid=True,
        invalid=False,
    )
    if payload.get("decision") != decision:
        raise ValueError("EXP-337 final audit reducer decision")
    for key, expected in vectors.items():
        if payload.get(key) != expected:
            raise ValueError(f"EXP-337 final audit reducer vector: {key}")

    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-337 final audit authorization drift: {key}")

    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-337 final audit evidence digest")

    return {
        "schema": "EXP337-INDEPENDENT-FINAL-AUDIT-V1",
        "final_json_path": str(final_json_path),
        "decision": decision,
        "evidence_digest": claimed,
        "chunk_artifact_digests": expected_artifact_digests,
        "chunk_bundle_digests": expected_bundle_digests,
        "control_sham_exact_by_chunk": [True] * CHUNK_COUNT,
        "reducer_vectors": vectors,
        "authorization_flags": dict(AUTHORIZATION_FLAGS),
    }
