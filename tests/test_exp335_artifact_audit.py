from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from nolane_ai.experiments.exp335_artifact_audit import (
    CHUNK_RECEIPT_SCHEMA,
    CHUNK_SUMMARY_SCHEMA,
    audit_chunk_chain,
    audit_chunk_zip,
)
from nolane_ai.experiments.exp335_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    EXPOSURES_PER_CHUNK,
    RECONSTRUCTION_ZIP_DIGEST,
    WORLD_IDS,
    canonical_digest,
    data_order_digest,
)


IDENTITY = {
    "source_tree_digest": "1" * 64,
    "exp335_execution_digest": "2" * 64,
}

_STEMS = {
    "CONTROL_FULL32": "control",
    "SHAM_MEASURE_FULL32": "sham",
    "SUBSPACE_PROJECT_FULL32": "project",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundary(arm: str, chunk_index: int) -> dict[str, object]:
    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    shared = arm in {"CONTROL_FULL32", "SHAM_MEASURE_FULL32"}
    return {
        "arm": arm,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": exposure,
        "cumulative_source_updates": exposure * len(WORLD_IDS),
        "world_token_accuracies": [1.0] * len(WORLD_IDS),
        "world_full_answer_exact": [1.0] * len(WORLD_IDS),
        "model_state_digest": ("a" if shared else "d") * 64,
        "optimizer_state_digest": ("b" if shared else "e") * 64,
        "rng_state_digest": ("c" if shared else "f") * 64,
        "nonfinite_events": 0,
        "negative_target_count": 0 if arm == "CONTROL_FULL32" else 17,
        "projection_update_count": 3 if arm == "SUBSPACE_PROJECT_FULL32" else 0,
        "projected_target_count": 11 if arm == "SUBSPACE_PROJECT_FULL32" else 0,
    }


def _receipt(
    *,
    arm: str,
    chunk_index: int,
    parent_digest: str,
    checkpoint_sha: str,
    boundary: dict[str, object],
) -> dict[str, object]:
    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    value: dict[str, object] = {
        "schema": CHUNK_RECEIPT_SCHEMA,
        "arm": arm,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": exposure,
        "cumulative_source_updates": exposure * len(WORLD_IDS),
        "model_state_digest": boundary["model_state_digest"],
        "optimizer_state_digest": boundary["optimizer_state_digest"],
        "rng_state_digest": boundary["rng_state_digest"],
        "data_order_digest": data_order_digest(exposure),
        "parent_artifact_digest": parent_digest,
        "source_tree_digest": IDENTITY["source_tree_digest"],
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "checkpoint_sha256": checkpoint_sha,
        **AUTHORIZATION_FLAGS,
    }
    value["receipt_digest"] = canonical_digest(value)
    return value


def _build_chunk_zip(
    path: Path,
    *,
    chunk_index: int,
    parent_digest: str,
    tamper_checkpoint_after_receipt: bool = False,
) -> Path:
    rows: dict[str, dict[str, object]] = {}
    members: dict[str, bytes] = {}

    for arm in ARMS:
        stem = _STEMS[arm]
        checkpoint_name = f"{stem}.pt"
        receipt_name = f"{stem}-receipt.json"
        checkpoint = f"{arm}:{chunk_index}:checkpoint".encode()
        checkpoint_sha = _sha256_bytes(checkpoint)
        boundary = _boundary(arm, chunk_index)
        receipt = _receipt(
            arm=arm,
            chunk_index=chunk_index,
            parent_digest=parent_digest,
            checkpoint_sha=checkpoint_sha,
            boundary=boundary,
        )
        rows[arm] = {
            "boundary": boundary,
            "files": {
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": checkpoint_sha,
                "receipt": receipt_name,
                "receipt_digest": receipt["receipt_digest"],
            },
            "measurements": [],
            "metrics": {},
        }
        members[checkpoint_name] = checkpoint
        members[receipt_name] = (
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )

    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    summary: dict[str, object] = {
        "schema": CHUNK_SUMMARY_SCHEMA,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": exposure,
        "cumulative_source_updates": exposure * len(WORLD_IDS),
        "parent_artifact_digest": parent_digest,
        "source_tree_digest": IDENTITY["source_tree_digest"],
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "execution_digest": IDENTITY["exp335_execution_digest"],
        "control_sham_exact": True,
        "arms": rows,
        **AUTHORIZATION_FLAGS,
    }
    summary["bundle_digest"] = canonical_digest(summary)
    members["chunk-summary.json"] = (
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    if tamper_checkpoint_after_receipt:
        members["project.pt"] += b":tampered"

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    return path


def test_independent_auditor_accepts_two_chunk_chain(tmp_path: Path):
    chunk0 = _build_chunk_zip(
        tmp_path / "chunk0.zip",
        chunk_index=0,
        parent_digest=RECONSTRUCTION_ZIP_DIGEST,
    )
    chunk0_digest = _sha256_file(chunk0)
    chunk1 = _build_chunk_zip(
        tmp_path / "chunk1.zip",
        chunk_index=1,
        parent_digest=chunk0_digest,
    )

    report = audit_chunk_chain([chunk0, chunk1], identity=IDENTITY)

    assert report["verified_chunk_count"] == 2
    assert report["complete_chunk_chain"] is False
    assert report["chunks"][0]["artifact_sha256"] == chunk0_digest
    assert report["chunks"][1]["parent_artifact_digest"] == chunk0_digest
    assert report["chunks"][1]["pass_counts"] == {arm: 32 for arm in ARMS}
    assert report["authorization_flags"] == AUTHORIZATION_FLAGS


def test_independent_auditor_rejects_checkpoint_tamper(tmp_path: Path):
    invalid = _build_chunk_zip(
        tmp_path / "tampered.zip",
        chunk_index=0,
        parent_digest=RECONSTRUCTION_ZIP_DIGEST,
        tamper_checkpoint_after_receipt=True,
    )

    with pytest.raises(ValueError, match="checkpoint"):
        audit_chunk_zip(
            invalid,
            identity=IDENTITY,
            expected_chunk_index=0,
            expected_parent_artifact_digest=RECONSTRUCTION_ZIP_DIGEST,
        )


def test_independent_auditor_rejects_parent_chain_splice(tmp_path: Path):
    chunk0 = _build_chunk_zip(
        tmp_path / "chunk0.zip",
        chunk_index=0,
        parent_digest=RECONSTRUCTION_ZIP_DIGEST,
    )
    chunk1 = _build_chunk_zip(
        tmp_path / "chunk1.zip",
        chunk_index=1,
        parent_digest="9" * 64,
    )

    with pytest.raises(ValueError, match="parent artifact chain"):
        audit_chunk_chain([chunk0, chunk1], identity=IDENTITY)
