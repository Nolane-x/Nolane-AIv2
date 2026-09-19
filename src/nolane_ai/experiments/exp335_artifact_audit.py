from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from .exp335_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FULL_EXACT_FLOOR,
    RECONSTRUCTION_ZIP_DIGEST,
    TOKEN_FLOOR,
    WORLD_IDS,
    BoundaryResult,
    canonical_digest,
    data_order_digest,
    sham_equivalent,
    validate_boundary,
)

CHUNK_RECEIPT_SCHEMA = "EXP335-CONTINUATION-RECEIPT-V1"
CHUNK_SUMMARY_SCHEMA = "EXP335-CHUNK-SUMMARY-V1"

_STEMS = {
    "CONTROL_FULL32": "control",
    "SHAM_MEASURE_FULL32": "sham",
    "SUBSPACE_PROJECT_FULL32": "project",
}


def _normalize_digest(value: str) -> str:
    raw = value[7:] if value.startswith("sha256:") else value
    if len(raw) != 64:
        raise ValueError("EXP-335 audit digest length")
    try:
        int(raw, 16)
    except ValueError as exc:
        raise ValueError("EXP-335 audit digest hex") from exc
    return raw.lower()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_sha256(archive: ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_member(archive: ZipFile, name: str) -> dict[str, Any]:
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise ValueError(f"EXP-335 audit missing ZIP member: {name}") from exc
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXP-335 audit JSON object required: {name}")
    return value


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    materialized = dict(receipt)
    materialized.pop("receipt_digest", None)
    return canonical_digest(materialized)


def _identity_fields(identity: Mapping[str, Any]) -> tuple[str, str]:
    source_tree = identity.get("source_tree_digest")
    execution = identity.get("exp335_execution_digest")
    if not isinstance(source_tree, str):
        raise ValueError("EXP-335 audit identity source tree")
    if not isinstance(execution, str):
        raise ValueError("EXP-335 audit identity execution digest")
    return _normalize_digest(source_tree), _normalize_digest(execution)


def _pass_count(boundary: BoundaryResult) -> int:
    return sum(
        1
        for token, exact in zip(
            boundary.world_token_accuracies, boundary.world_full_answer_exact
        )
        if boundary.nonfinite_events == 0
        and float(token) >= TOKEN_FLOOR
        and float(exact) >= FULL_EXACT_FLOOR
    )


def audit_chunk_zip(
    zip_path: str | Path,
    *,
    identity: Mapping[str, Any],
    expected_chunk_index: int,
    expected_parent_artifact_digest: str,
) -> dict[str, Any]:
    if not 0 <= expected_chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-335 audit chunk index")

    source_tree_digest, execution_digest = _identity_fields(identity)
    expected_parent = _normalize_digest(expected_parent_artifact_digest)
    artifact_path = Path(zip_path)
    artifact_digest = _file_sha256(artifact_path)

    with ZipFile(artifact_path, "r") as archive:
        summary = _json_member(archive, "chunk-summary.json")
        if summary.get("schema") != CHUNK_SUMMARY_SCHEMA:
            raise ValueError("EXP-335 audit summary schema")
        if summary.get("chunk_index") != expected_chunk_index:
            raise ValueError("EXP-335 audit summary chunk index")
        if summary.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
            raise ValueError("EXP-335 audit preregistration digest")
        if _normalize_digest(str(summary.get("source_tree_digest"))) != source_tree_digest:
            raise ValueError("EXP-335 audit source tree digest")
        if _normalize_digest(str(summary.get("execution_digest"))) != execution_digest:
            raise ValueError("EXP-335 audit execution digest")
        if _normalize_digest(str(summary.get("parent_artifact_digest"))) != expected_parent:
            raise ValueError("EXP-335 audit parent artifact chain")

        expected_exposure = (expected_chunk_index + 1) * EXPOSURES_PER_CHUNK
        expected_updates = expected_exposure * len(WORLD_IDS)
        if summary.get("cumulative_exposure_per_world") != expected_exposure:
            raise ValueError("EXP-335 audit summary exposure")
        if summary.get("cumulative_source_updates") != expected_updates:
            raise ValueError("EXP-335 audit summary updates")

        for key, expected in AUTHORIZATION_FLAGS.items():
            if summary.get(key) is not expected:
                raise ValueError(f"EXP-335 audit authorization drift: {key}")

        materialized = dict(summary)
        claimed_bundle = materialized.pop("bundle_digest", None)
        if not isinstance(claimed_bundle, str) or canonical_digest(materialized) != claimed_bundle:
            raise ValueError("EXP-335 audit bundle digest")

        arms = summary.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            raise ValueError("EXP-335 audit arm coverage")

        boundaries: dict[str, BoundaryResult] = {}
        receipt_digests: dict[str, str] = {}
        checkpoint_shas: dict[str, str] = {}

        for arm in ARMS:
            row = arms[arm]
            if not isinstance(row, Mapping):
                raise ValueError("EXP-335 audit arm row")
            boundary_raw = row.get("boundary")
            files = row.get("files")
            if not isinstance(boundary_raw, Mapping) or not isinstance(files, Mapping):
                raise ValueError("EXP-335 audit arm boundary/files")
            boundary = BoundaryResult(**boundary_raw)
            validate_boundary(boundary)
            if boundary.arm != arm or boundary.chunk_index != expected_chunk_index:
                raise ValueError("EXP-335 audit boundary identity")
            boundaries[arm] = boundary

            stem = _STEMS[arm]
            checkpoint_name = f"{stem}.pt"
            receipt_name = f"{stem}-receipt.json"
            if files.get("checkpoint") != checkpoint_name or files.get("receipt") != receipt_name:
                raise ValueError("EXP-335 audit file manifest")

            checkpoint_sha = _member_sha256(archive, checkpoint_name)
            checkpoint_shas[arm] = checkpoint_sha
            if files.get("checkpoint_sha256") != checkpoint_sha:
                raise ValueError("EXP-335 audit checkpoint manifest SHA")

            receipt = _json_member(archive, receipt_name)
            if receipt.get("schema") != CHUNK_RECEIPT_SCHEMA:
                raise ValueError("EXP-335 audit receipt schema")
            if receipt.get("arm") != arm or receipt.get("chunk_index") != expected_chunk_index:
                raise ValueError("EXP-335 audit receipt chain position")
            if receipt.get("cumulative_exposure_per_world") != expected_exposure:
                raise ValueError("EXP-335 audit receipt exposure")
            if receipt.get("cumulative_source_updates") != expected_updates:
                raise ValueError("EXP-335 audit receipt updates")
            if receipt.get("data_order_digest") != data_order_digest(expected_exposure):
                raise ValueError("EXP-335 audit data order digest")
            if _normalize_digest(str(receipt.get("parent_artifact_digest"))) != expected_parent:
                raise ValueError("EXP-335 audit receipt parent artifact")
            if _normalize_digest(str(receipt.get("source_tree_digest"))) != source_tree_digest:
                raise ValueError("EXP-335 audit receipt source tree")
            if receipt.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
                raise ValueError("EXP-335 audit receipt preregistration")
            if receipt.get("checkpoint_sha256") != checkpoint_sha:
                raise ValueError("EXP-335 audit receipt checkpoint SHA")

            for key in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
                value = receipt.get(key)
                if not isinstance(value, str):
                    raise ValueError(f"EXP-335 audit receipt state digest: {key}")
                _normalize_digest(value)
                if value != getattr(boundary, key):
                    raise ValueError(f"EXP-335 audit receipt/boundary mismatch: {key}")

            claimed_receipt = receipt.get("receipt_digest")
            if not isinstance(claimed_receipt, str) or _receipt_digest(receipt) != claimed_receipt:
                raise ValueError("EXP-335 audit receipt digest")
            if files.get("receipt_digest") != claimed_receipt:
                raise ValueError("EXP-335 audit receipt manifest digest")
            receipt_digests[arm] = claimed_receipt

            for key, expected in AUTHORIZATION_FLAGS.items():
                if receipt.get(key) is not expected:
                    raise ValueError(f"EXP-335 audit receipt authorization drift: {key}")

        if summary.get("control_sham_exact") is not True:
            raise ValueError("EXP-335 audit summary CONTROL/SHAM flag")
        if not sham_equivalent(
            boundaries["CONTROL_FULL32"], boundaries["SHAM_MEASURE_FULL32"]
        ):
            raise ValueError("EXP-335 audit CONTROL/SHAM mismatch")

    return {
        "chunk_index": expected_chunk_index,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_digest,
        "parent_artifact_digest": expected_parent,
        "bundle_digest": claimed_bundle,
        "cumulative_exposure_per_world": expected_exposure,
        "cumulative_source_updates": expected_updates,
        "control_sham_exact": True,
        "pass_counts": {arm: _pass_count(boundaries[arm]) for arm in ARMS},
        "checkpoint_sha256": checkpoint_shas,
        "receipt_digests": receipt_digests,
        "projection_update_count": boundaries["SUBSPACE_PROJECT_FULL32"].projection_update_count,
        "projected_target_count": boundaries["SUBSPACE_PROJECT_FULL32"].projected_target_count,
        "nonfinite_events": {arm: boundaries[arm].nonfinite_events for arm in ARMS},
        "authorization_flags": dict(AUTHORIZATION_FLAGS),
    }


def audit_chunk_chain(
    zip_paths: Sequence[str | Path],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not zip_paths:
        raise ValueError("EXP-335 audit requires at least one chunk artifact")
    if len(zip_paths) > CHUNK_COUNT:
        raise ValueError("EXP-335 audit too many chunks")

    reports: list[dict[str, Any]] = []
    parent = RECONSTRUCTION_ZIP_DIGEST
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
        "schema": "EXP335-INDEPENDENT-ARTIFACT-CHAIN-AUDIT-V1",
        "verified_chunk_count": len(reports),
        "complete_chunk_chain": len(reports) == CHUNK_COUNT,
        "first_parent_reconstruction_zip_digest": RECONSTRUCTION_ZIP_DIGEST,
        "last_artifact_sha256": reports[-1]["artifact_sha256"],
        "chunks": reports,
        "authorization_flags": dict(AUTHORIZATION_FLAGS),
    }
