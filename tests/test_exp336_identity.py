from dataclasses import replace

import pytest

from nolane_ai.experiments.exp336_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PARENT_CNRS_ARTIFACT_ID,
    PARENT_CNRS_CHECKPOINT_SHA256,
    PARENT_CNRS_MODEL_STATE_DIGEST,
    PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
    PARENT_CNRS_RECEIPT_SHA256,
    PARENT_CNRS_RNG_STATE_DIGEST,
    PARENT_CNRS_SUMMARY_SHA256,
    PARENT_CNRS_ZIP_DIGEST,
    PARENT_EXP319_RUN_ID,
    PARENT_EXP335_AUDIT_ARTIFACT_ID,
    PARENT_EXP335_AUDIT_REPORT_SHA256,
    PARENT_EXP335_AUDIT_RUN_ID,
    PARENT_EXP335_AUDIT_ZIP_DIGEST,
    PARENT_EXP335_EVIDENCE_DIGEST,
    PARENT_EXP335_EXECUTION_DIGEST,
    PARENT_EXP335_FINAL_ARTIFACT_ID,
    PARENT_EXP335_FINAL_JSON_SHA256,
    PARENT_EXP335_FINAL_ZIP_DIGEST,
    PARENT_EXP335_RUN_ID,
    PARENT_EXP335_SCIENTIFIC_SOURCE_SHA,
    PARENT_EXP335_SEALED_MARKER_SHA,
    PARENT_SELECTION_ARTIFACT_ID,
    PARENT_SELECTION_AUTHORITY_DIGEST,
    PARENT_SELECTION_JSON_SHA256,
    PARENT_SELECTION_ZIP_DIGEST,
)
from nolane_ai.experiments.exp336_identity import (
    SCHEMA,
    Exp336ExecutionIdentity,
    execution_digest,
    validate_execution_identity,
)


def _identity() -> Exp336ExecutionIdentity:
    base = Exp336ExecutionIdentity(
        schema=SCHEMA,
        source_commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        workflow_sha256="c" * 64,
        approved_preregistration_digest=APPROVED_PREREGISTRATION_DIGEST,
        parent_exp319_run_id=PARENT_EXP319_RUN_ID,
        parent_selection_artifact_id=PARENT_SELECTION_ARTIFACT_ID,
        parent_selection_zip_digest=PARENT_SELECTION_ZIP_DIGEST,
        parent_selection_json_sha256=PARENT_SELECTION_JSON_SHA256,
        parent_selection_authority_digest=PARENT_SELECTION_AUTHORITY_DIGEST,
        parent_cnrs_artifact_id=PARENT_CNRS_ARTIFACT_ID,
        parent_cnrs_zip_digest=PARENT_CNRS_ZIP_DIGEST,
        parent_cnrs_checkpoint_sha256=PARENT_CNRS_CHECKPOINT_SHA256,
        parent_cnrs_receipt_sha256=PARENT_CNRS_RECEIPT_SHA256,
        parent_cnrs_summary_sha256=PARENT_CNRS_SUMMARY_SHA256,
        parent_cnrs_model_state_digest=PARENT_CNRS_MODEL_STATE_DIGEST,
        parent_cnrs_optimizer_state_digest=PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
        parent_cnrs_rng_state_digest=PARENT_CNRS_RNG_STATE_DIGEST,
        parent_exp335_run_id=PARENT_EXP335_RUN_ID,
        parent_exp335_sealed_marker_sha=PARENT_EXP335_SEALED_MARKER_SHA,
        parent_exp335_scientific_source_sha=PARENT_EXP335_SCIENTIFIC_SOURCE_SHA,
        parent_exp335_execution_digest=PARENT_EXP335_EXECUTION_DIGEST,
        parent_exp335_final_artifact_id=PARENT_EXP335_FINAL_ARTIFACT_ID,
        parent_exp335_final_zip_digest=PARENT_EXP335_FINAL_ZIP_DIGEST,
        parent_exp335_final_json_sha256=PARENT_EXP335_FINAL_JSON_SHA256,
        parent_exp335_evidence_digest=PARENT_EXP335_EVIDENCE_DIGEST,
        parent_exp335_audit_run_id=PARENT_EXP335_AUDIT_RUN_ID,
        parent_exp335_audit_artifact_id=PARENT_EXP335_AUDIT_ARTIFACT_ID,
        parent_exp335_audit_zip_digest=PARENT_EXP335_AUDIT_ZIP_DIGEST,
        parent_exp335_audit_report_sha256=PARENT_EXP335_AUDIT_REPORT_SHA256,
        exp336_execution_digest="0" * 64,
    )
    return replace(base, exp336_execution_digest=execution_digest(base))


def test_identity_validates_frozen_authority():
    validate_execution_identity(_identity())


def test_identity_rejects_parent_or_preregistration_drift():
    with pytest.raises(ValueError):
        validate_execution_identity(
            replace(_identity(), approved_preregistration_digest="0" * 64)
        )
    with pytest.raises(ValueError):
        validate_execution_identity(
            replace(_identity(), parent_cnrs_checkpoint_sha256="0" * 64)
        )
