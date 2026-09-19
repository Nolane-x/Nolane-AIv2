from dataclasses import replace

import pytest

from nolane_ai.experiments.exp335_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PARENT_EXP334_ARTIFACT_ID,
    PARENT_EXP334_EVIDENCE_DIGEST,
    PARENT_EXP334_JSON_SHA256,
    PARENT_EXP334_RUN_ID,
    PARENT_EXP334_ZIP_DIGEST,
    RECONSTRUCTION_ARTIFACT_ID,
    RECONSTRUCTION_CHECKPOINT_SHA256,
    RECONSTRUCTION_RECEIPT_SHA256,
    RECONSTRUCTION_RUN_ID,
    RECONSTRUCTION_ZIP_DIGEST,
)
from nolane_ai.experiments.exp335_identity import (
    SCHEMA,
    Exp335ExecutionIdentity,
    execution_digest,
    validate_execution_identity,
)


def _identity() -> Exp335ExecutionIdentity:
    base = Exp335ExecutionIdentity(
        schema=SCHEMA,
        source_commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        workflow_sha256="c" * 64,
        approved_preregistration_digest=APPROVED_PREREGISTRATION_DIGEST,
        parent_exp334_run_id=PARENT_EXP334_RUN_ID,
        parent_exp334_artifact_id=PARENT_EXP334_ARTIFACT_ID,
        parent_exp334_zip_digest=PARENT_EXP334_ZIP_DIGEST,
        parent_exp334_final_json_sha256=PARENT_EXP334_JSON_SHA256,
        parent_exp334_evidence_digest=PARENT_EXP334_EVIDENCE_DIGEST,
        reconstruction_run_id=RECONSTRUCTION_RUN_ID,
        reconstruction_artifact_id=RECONSTRUCTION_ARTIFACT_ID,
        reconstruction_zip_digest=RECONSTRUCTION_ZIP_DIGEST,
        reconstruction_checkpoint_sha256=RECONSTRUCTION_CHECKPOINT_SHA256,
        reconstruction_receipt_sha256=RECONSTRUCTION_RECEIPT_SHA256,
        exp335_execution_digest="0" * 64,
    )
    return replace(base, exp335_execution_digest=execution_digest(base))


def test_identity_validates_frozen_authority():
    validate_execution_identity(_identity())


def test_identity_rejects_preregistration_drift():
    with pytest.raises(ValueError):
        validate_execution_identity(
            replace(_identity(), approved_preregistration_digest="0" * 64)
        )
