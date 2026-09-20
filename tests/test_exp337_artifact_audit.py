from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from nolane_ai.experiments.exp337_artifact_audit import (
    CHUNK_RECEIPT_SCHEMA,
    CHUNK_SUMMARY_SCHEMA,
    audit_chunk_chain,
    audit_chunk_zip,
    audit_final_evidence,
)
from nolane_ai.experiments.exp337_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    PARENT_EXP336_CHUNK7_BUNDLE_DIGEST,
    PARENT_EXP336_CHUNK7_ZIP_SHA256,
    PARENT_EXP336_CLOSURE_DIGEST,
    PARENT_EXP336_EVIDENCE_DIGEST,
    PARENT_EXP336_HEAD_SHA,
    PARENT_EXP336_RUN_ID,
    PARENT_PROJECT_CHECKPOINT_SHA256,
    PARENT_PROJECT_MODEL_STATE_DIGEST,
    PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
    PARENT_PROJECT_RECEIPT_DIGEST,
    PARENT_PROJECT_RECEIPT_JSON_SHA256,
    PARENT_PROJECT_RNG_STATE_DIGEST,
    STARTING_CUMULATIVE_STEP,
    WORLD_IDS,
    BoundaryResult,
    canonical_digest,
    data_order_digest,
    reduce_full32,
)
from nolane_ai.experiments.exp337_identity import (
    Exp337ExecutionIdentity,
    execution_digest,
)

_IDENTITY_SEED = Exp337ExecutionIdentity(
    schema="EXP337-CNRS-SELF-ROLLIN-EXECUTION-IDENTITY-V1",
    source_commit_sha="1" * 40,
    source_tree_digest="2" * 64,
    workflow_sha256="3" * 64,
    approved_preregistration_digest=APPROVED_PREREGISTRATION_DIGEST,
    parent_exp336_run_id=PARENT_EXP336_RUN_ID,
    parent_exp336_head_sha=PARENT_EXP336_HEAD_SHA,
    parent_exp336_closure_digest=PARENT_EXP336_CLOSURE_DIGEST,
    parent_exp336_evidence_digest=PARENT_EXP336_EVIDENCE_DIGEST,
    parent_chunk7_artifact_id=10601286249,
    parent_chunk7_zip_sha256=PARENT_EXP336_CHUNK7_ZIP_SHA256,
    parent_chunk7_bundle_digest=PARENT_EXP336_CHUNK7_BUNDLE_DIGEST,
    parent_project_checkpoint_sha256=PARENT_PROJECT_CHECKPOINT_SHA256,
    parent_project_receipt_json_sha256=PARENT_PROJECT_RECEIPT_JSON_SHA256,
    parent_project_receipt_digest=PARENT_PROJECT_RECEIPT_DIGEST,
    parent_project_model_state_digest=PARENT_PROJECT_MODEL_STATE_DIGEST,
    parent_project_optimizer_state_digest=PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
    parent_project_rng_state_digest=PARENT_PROJECT_RNG_STATE_DIGEST,
    exp337_execution_digest="0" * 64,
)
IDENTITY = asdict(
    replace(
        _IDENTITY_SEED,
        exp337_execution_digest=execution_digest(_IDENTITY_SEED),
    )
)

_STEMS = {
    "CONTROL_PROJECT_GOLD_PREFIX": "control",
    "SHAM_SELF_ROLLIN_MEASURE_PROJECT": "sham",
    "SELF_ROLLIN_RECOVERY_PROJECT": "recovery",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundary(
    arm: str,
    chunk_index: int,
    *,
    sham_mismatch: bool = False,
) -> dict[str, object]:
    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    updates = exposure * len(WORLD_IDS)
    control_or_sham = arm in {
        "CONTROL_PROJECT_GOLD_PREFIX",
        "SHAM_SELF_ROLLIN_MEASURE_PROJECT",
    }
    state_key = "a" if control_or_sham else "d"
    optimizer_key = "b" if control_or_sham else "e"
    rng_key = "c" if control_or_sham else "f"
    if sham_mismatch and arm == "SHAM_SELF_ROLLIN_MEASURE_PROJECT":
        state_key = "9"

    if arm == "CONTROL_PROJECT_GOLD_PREFIX":
        measured = active = divergent = 0
    elif arm == "SHAM_SELF_ROLLIN_MEASURE_PROJECT":
        measured, active, divergent = updates, 0, 0
    else:
        measured, active, divergent = updates, updates, 0

    return {
        "arm": arm,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": exposure,
        "cumulative_source_updates": updates,
        "cumulative_training_step": STARTING_CUMULATIVE_STEP + updates,
        "world_token_accuracies": [1.0] * len(WORLD_IDS),
        "world_full_answer_exact": [1.0] * len(WORLD_IDS),
        "aggregate_answer_only_loss": 1.0,
        "aggregate_answer_token_accuracy": 1.0,
        "aggregate_greedy_exact_match": 1.0,
        "aggregate_eos_correctness": 1.0,
        "aggregate_invalid_output_rate": 0.0,
        "aggregate_loss_fraction_of_original_initial": 0.01,
        "model_state_digest": state_key * 64,
        "optimizer_state_digest": optimizer_key * 64,
        "rng_state_digest": rng_key * 64,
        "nonfinite_events": 0,
        "negative_target_count": 17,
        "projection_update_count": 3,
        "projected_target_count": 11,
        "self_rollin_measurement_count": measured,
        "self_rollin_active_update_count": active,
        "self_rollin_divergent_update_count": divergent,
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
    updates = exposure * len(WORLD_IDS)
    value: dict[str, object] = {
        "schema": CHUNK_RECEIPT_SCHEMA,
        "arm": arm,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": exposure,
        "cumulative_source_updates": updates,
        "cumulative_training_step": STARTING_CUMULATIVE_STEP + updates,
        "model_state_digest": boundary["model_state_digest"],
        "optimizer_state_digest": boundary["optimizer_state_digest"],
        "rng_state_digest": boundary["rng_state_digest"],
        "data_order_digest": data_order_digest(exposure),
        "parent_artifact_digest": parent_digest,
        "source_tree_digest": IDENTITY["source_tree_digest"],
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "execution_digest": IDENTITY["exp337_execution_digest"],
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
    sham_mismatch: bool = False,
) -> Path:
    rows: dict[str, dict[str, object]] = {}
    members: dict[str, bytes] = {}
    for arm in ARMS:
        stem = _STEMS[arm]
        checkpoint_name = f"{stem}.pt"
        receipt_name = f"{stem}-receipt.json"
        checkpoint = f"{arm}:{chunk_index}:checkpoint".encode()
        checkpoint_sha = _sha256_bytes(checkpoint)
        boundary = _boundary(arm, chunk_index, sham_mismatch=sham_mismatch)
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
            "aggregate_metrics": {},
        }
        members[checkpoint_name] = checkpoint
        members[receipt_name] = (
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )

    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    updates = exposure * len(WORLD_IDS)
    summary: dict[str, object] = {
        "schema": CHUNK_SUMMARY_SCHEMA,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": exposure,
        "cumulative_source_updates": updates,
        "cumulative_training_step": STARTING_CUMULATIVE_STEP + updates,
        "parent_artifact_digest": parent_digest,
        "parent_exp336_artifact_name": (
            "exp336-chunk-7-35481336946" if chunk_index == 0 else None
        ),
        "parent_exp336_bundle_digest": (
            PARENT_EXP336_CHUNK7_BUNDLE_DIGEST if chunk_index == 0 else None
        ),
        "source_tree_digest": IDENTITY["source_tree_digest"],
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "execution_digest": IDENTITY["exp337_execution_digest"],
        "control_sham_exact": not sham_mismatch,
        "arms": rows,
        **AUTHORIZATION_FLAGS,
    }
    summary["bundle_digest"] = canonical_digest(summary)
    members["chunk-summary.json"] = (
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    if tamper_checkpoint_after_receipt:
        members["recovery.pt"] += b":tampered"

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    return path


def test_independent_auditor_accepts_two_chunk_chain(tmp_path: Path):
    chunk0 = _build_chunk_zip(
        tmp_path / "chunk0.zip",
        chunk_index=0,
        parent_digest=PARENT_EXP336_CHUNK7_ZIP_SHA256,
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
    assert report["chunks"][1]["aggregate_pass"] == {arm: True for arm in ARMS}
    assert report["chunks"][1]["self_rollin_measurement_count"] == {
        "CONTROL_PROJECT_GOLD_PREFIX": 0,
        "SHAM_SELF_ROLLIN_MEASURE_PROJECT": 256,
        "SELF_ROLLIN_RECOVERY_PROJECT": 256,
    }


def test_independent_auditor_rejects_checkpoint_tamper(tmp_path: Path):
    invalid = _build_chunk_zip(
        tmp_path / "tampered.zip",
        chunk_index=0,
        parent_digest=PARENT_EXP336_CHUNK7_ZIP_SHA256,
        tamper_checkpoint_after_receipt=True,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        audit_chunk_zip(
            invalid,
            identity=IDENTITY,
            expected_chunk_index=0,
            expected_parent_artifact_digest=PARENT_EXP336_CHUNK7_ZIP_SHA256,
        )


def test_independent_auditor_rejects_parent_chain_splice(tmp_path: Path):
    chunk0 = _build_chunk_zip(
        tmp_path / "chunk0.zip",
        chunk_index=0,
        parent_digest=PARENT_EXP336_CHUNK7_ZIP_SHA256,
    )
    chunk1 = _build_chunk_zip(
        tmp_path / "chunk1.zip",
        chunk_index=1,
        parent_digest="9" * 64,
    )
    with pytest.raises(ValueError, match="parent artifact chain"):
        audit_chunk_chain([chunk0, chunk1], identity=IDENTITY)


def test_independent_auditor_rejects_control_sham_perturbation(tmp_path: Path):
    invalid = _build_chunk_zip(
        tmp_path / "sham-mismatch.zip",
        chunk_index=0,
        parent_digest=PARENT_EXP336_CHUNK7_ZIP_SHA256,
        sham_mismatch=True,
    )
    with pytest.raises(ValueError, match="CONTROL/SHAM"):
        audit_chunk_zip(
            invalid,
            identity=IDENTITY,
            expected_chunk_index=0,
            expected_parent_artifact_digest=PARENT_EXP336_CHUNK7_ZIP_SHA256,
        )


def _build_complete_chain(tmp_path: Path) -> tuple[list[Path], dict[str, object]]:
    chunks: list[Path] = []
    parent = PARENT_EXP336_CHUNK7_ZIP_SHA256
    for index in range(CHUNK_COUNT):
        path = _build_chunk_zip(
            tmp_path / f"chunk{index}.zip",
            chunk_index=index,
            parent_digest=parent,
        )
        chunks.append(path)
        parent = _sha256_file(path)
    report = audit_chunk_chain(chunks, identity=IDENTITY)
    assert report["complete_chunk_chain"] is True
    return chunks, report


def _build_final_payload(chain_report: dict[str, object]) -> dict[str, object]:
    chunks = chain_report["chunks"]
    assert isinstance(chunks, list)
    final_boundaries = chunks[-1]["boundaries"]
    control = BoundaryResult(**final_boundaries["CONTROL_PROJECT_GOLD_PREFIX"])
    sham = BoundaryResult(**final_boundaries["SHAM_SELF_ROLLIN_MEASURE_PROJECT"])
    recovery = BoundaryResult(**final_boundaries["SELF_ROLLIN_RECOVERY_PROJECT"])
    decision, vectors = reduce_full32(
        control, sham, recovery, parent_authority_valid=True, invalid=False
    )
    payload: dict[str, object] = {
        "schema": "EXP337-CNRS-SELF-ROLLIN-FINAL-EVIDENCE-V1",
        "execution_identity": dict(IDENTITY),
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "parent_exp336_final_decision": "CNRS_FULL32_STAGE_A_NOT_ESTABLISHED",
        "parent_exp336_evidence_digest": PARENT_EXP336_EVIDENCE_DIGEST,
        "parent_exp336_chunk7_zip_sha256": PARENT_EXP336_CHUNK7_ZIP_SHA256,
        "parent_project_model_state_digest": PARENT_PROJECT_MODEL_STATE_DIGEST,
        "parent_project_optimizer_state_digest": PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
        "parent_project_rng_state_digest": PARENT_PROJECT_RNG_STATE_DIGEST,
        "chunk_bundle_digests": [row["bundle_digest"] for row in chunks],
        "chunk_artifact_digests": [row["artifact_sha256"] for row in chunks],
        "chunk_parent_artifact_digests": [
            row["parent_artifact_digest"] for row in chunks
        ],
        "control_sham_exact_by_chunk": [True] * CHUNK_COUNT,
        "final_boundaries": final_boundaries,
        "decision": decision,
        **vectors,
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = canonical_digest(payload)
    return payload


def test_independent_final_auditor_recomputes_reducer_from_chunk7(tmp_path: Path):
    _, chain_report = _build_complete_chain(tmp_path)
    final_payload = _build_final_payload(chain_report)
    final_path = tmp_path / "final.json"
    final_path.write_text(
        json.dumps(final_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report = audit_final_evidence(
        final_path,
        chain_report=chain_report,
        identity=IDENTITY,
    )
    assert report["decision"] == "EXP337_STAGE_A_ESTABLISHED_BOTH"
    assert report["evidence_digest"] == final_payload["evidence_digest"]
    assert len(report["chunk_artifact_digests"]) == CHUNK_COUNT
    assert report["authorization_flags"] == AUTHORIZATION_FLAGS


def test_independent_final_auditor_rejects_forged_decision(tmp_path: Path):
    _, chain_report = _build_complete_chain(tmp_path)
    final_payload = _build_final_payload(chain_report)
    final_payload["decision"] = "EXP337_SELF_ROLLIN_NO_CAUSAL_RESCUE"
    final_payload["evidence_digest"] = canonical_digest(
        {key: value for key, value in final_payload.items() if key != "evidence_digest"}
    )
    final_path = tmp_path / "forged-final.json"
    final_path.write_text(
        json.dumps(final_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reducer decision"):
        audit_final_evidence(
            final_path,
            chain_report=chain_report,
            identity=IDENTITY,
        )
