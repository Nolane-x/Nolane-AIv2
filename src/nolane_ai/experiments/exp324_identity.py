from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp324_contract import (
    CUMULATIVE_CHECKPOINTS,
    FAMILIES,
    LOCAL_CHECKPOINTS,
    UPDATES_PER_FAMILY,
    preregistration_digest,
)


SCHEMA = "EXP324-FAMILY-ISOLATED-EXECUTION-IDENTITY-V1"
PARENT_MARKER_SHA = "217477a911aabbb764741e76cca38e54013e7da6"
PARENT_REPAIRED_RUN_ID = 35354410137
PARENT_RECOVERY_RUN_ID = 35356470634
PARENT_FINAL_ARTIFACT_ID = 10552276255
PARENT_FINAL_ARTIFACT_ZIP_DIGEST = "7155d72ab05b57f8b66284f83ed41e63057824a3288adcd27e2d5429f3e0a19c"
PARENT_FINAL_EVIDENCE_DIGEST = "a92d56a45151411a8fa4e0799949d72aed701f52ad9adae5fdb90983be09f051"
PARENT_EXECUTION_DIGEST = "70ebb1bda4c91f457aae1007df0d939f2f2a144257dfebf5481ee6dc55b10cd9"
RECONSTRUCTION_RUN_ID = 35345351869
RECONSTRUCTION_ARTIFACT_ID = 10547681681
RECONSTRUCTION_ARTIFACT_ZIP_DIGEST = "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621"
RECONSTRUCTION_CHECKPOINT_SHA256 = "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
RECONSTRUCTION_RECEIPT_DIGEST = "f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f"
_HEX=frozenset("0123456789abcdef")


@dataclass(frozen=True,slots=True)
class Exp324ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_marker_sha: str
    parent_repaired_run_id: int
    parent_recovery_run_id: int
    parent_final_artifact_id: int
    parent_final_artifact_zip_digest: str
    parent_final_evidence_digest: str
    parent_execution_digest: str
    reconstruction_run_id: int
    reconstruction_artifact_id: int
    reconstruction_artifact_zip_digest: str
    reconstruction_checkpoint_sha256: str
    reconstruction_receipt_digest: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    families: tuple[str,...]
    local_checkpoints: tuple[int,...]
    cumulative_checkpoints: tuple[int,...]
    updates_per_family: int
    device: str
    exp324_execution_digest: str


def canonical_json_bytes(payload: object)->bytes:
    return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")


def _hex(value:str,length:int,field:str)->None:
    if len(value)!=length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be {length} lowercase hex characters")


def source_tree_digest_from_git_tree_sha(tree_sha:str)->str:
    _hex(tree_sha,40,"git tree")
    return hashlib.sha256(f"EXP324-GIT-TREE-V1|{tree_sha}".encode("ascii")).hexdigest()


def execution_digest(identity:Exp324ExecutionIdentity)->str:
    p=asdict(identity); p.pop("exp324_execution_digest",None)
    return hashlib.sha256(canonical_json_bytes(p)).hexdigest()


def validate_execution_identity(identity:Exp324ExecutionIdentity)->None:
    if identity.schema!=SCHEMA: raise ValueError("EXP-324 identity schema mismatch")
    for value,length,field in (
        (identity.parent_marker_sha,40,"parent marker"),
        (identity.parent_final_artifact_zip_digest,64,"parent ZIP"),
        (identity.parent_final_evidence_digest,64,"parent evidence"),
        (identity.parent_execution_digest,64,"parent execution"),
        (identity.reconstruction_artifact_zip_digest,64,"reconstruction ZIP"),
        (identity.reconstruction_checkpoint_sha256,64,"checkpoint SHA"),
        (identity.reconstruction_receipt_digest,64,"receipt digest"),
        (identity.source_commit_sha,40,"source commit"),
        (identity.source_tree_digest,64,"source tree"),
        (identity.workflow_sha256,64,"workflow SHA"),
        (identity.exp324_execution_digest,64,"execution digest"),
    ): _hex(value,length,field)
    if identity.approved_preregistration_digest!=preregistration_digest(): raise ValueError("EXP-324 preregistration mismatch")
    if identity.parent_marker_sha!=PARENT_MARKER_SHA: raise ValueError("EXP-324 parent marker mismatch")
    if identity.parent_repaired_run_id!=PARENT_REPAIRED_RUN_ID or identity.parent_recovery_run_id!=PARENT_RECOVERY_RUN_ID: raise ValueError("EXP-324 parent run mismatch")
    if identity.parent_final_artifact_id!=PARENT_FINAL_ARTIFACT_ID or identity.parent_final_artifact_zip_digest!=PARENT_FINAL_ARTIFACT_ZIP_DIGEST: raise ValueError("EXP-324 parent final artifact mismatch")
    if identity.parent_final_evidence_digest!=PARENT_FINAL_EVIDENCE_DIGEST or identity.parent_execution_digest!=PARENT_EXECUTION_DIGEST: raise ValueError("EXP-324 parent evidence mismatch")
    if identity.reconstruction_run_id!=RECONSTRUCTION_RUN_ID or identity.reconstruction_artifact_id!=RECONSTRUCTION_ARTIFACT_ID: raise ValueError("EXP-324 reconstruction artifact mismatch")
    if identity.reconstruction_artifact_zip_digest!=RECONSTRUCTION_ARTIFACT_ZIP_DIGEST or identity.reconstruction_checkpoint_sha256!=RECONSTRUCTION_CHECKPOINT_SHA256 or identity.reconstruction_receipt_digest!=RECONSTRUCTION_RECEIPT_DIGEST: raise ValueError("EXP-324 reconstruction identity mismatch")
    if tuple(identity.families)!=tuple(FAMILIES): raise ValueError("EXP-324 family geometry mismatch")
    if tuple(identity.local_checkpoints)!=tuple(LOCAL_CHECKPOINTS) or tuple(identity.cumulative_checkpoints)!=tuple(CUMULATIVE_CHECKPOINTS): raise ValueError("EXP-324 checkpoint geometry mismatch")
    if identity.updates_per_family!=UPDATES_PER_FAMILY or identity.device!="cpu": raise ValueError("EXP-324 runtime geometry mismatch")
    if identity.exp324_execution_digest!=execution_digest(identity): raise ValueError("EXP-324 execution digest mismatch")


def build_execution_identity(*,source_commit_sha:str,git_tree_sha:str,workflow_sha256:str)->Exp324ExecutionIdentity:
    values={
        "schema":SCHEMA,
        "approved_preregistration_digest":preregistration_digest(),
        "parent_marker_sha":PARENT_MARKER_SHA,
        "parent_repaired_run_id":PARENT_REPAIRED_RUN_ID,
        "parent_recovery_run_id":PARENT_RECOVERY_RUN_ID,
        "parent_final_artifact_id":PARENT_FINAL_ARTIFACT_ID,
        "parent_final_artifact_zip_digest":PARENT_FINAL_ARTIFACT_ZIP_DIGEST,
        "parent_final_evidence_digest":PARENT_FINAL_EVIDENCE_DIGEST,
        "parent_execution_digest":PARENT_EXECUTION_DIGEST,
        "reconstruction_run_id":RECONSTRUCTION_RUN_ID,
        "reconstruction_artifact_id":RECONSTRUCTION_ARTIFACT_ID,
        "reconstruction_artifact_zip_digest":RECONSTRUCTION_ARTIFACT_ZIP_DIGEST,
        "reconstruction_checkpoint_sha256":RECONSTRUCTION_CHECKPOINT_SHA256,
        "reconstruction_receipt_digest":RECONSTRUCTION_RECEIPT_DIGEST,
        "source_commit_sha":source_commit_sha,
        "source_tree_digest":source_tree_digest_from_git_tree_sha(git_tree_sha),
        "workflow_sha256":workflow_sha256,
        "families":tuple(FAMILIES),
        "local_checkpoints":tuple(LOCAL_CHECKPOINTS),
        "cumulative_checkpoints":tuple(CUMULATIVE_CHECKPOINTS),
        "updates_per_family":UPDATES_PER_FAMILY,
        "device":"cpu",
    }
    provisional=Exp324ExecutionIdentity(**values,exp324_execution_digest="")
    return Exp324ExecutionIdentity(**values,exp324_execution_digest=execution_digest(provisional))


def canonical_identity_json_bytes(identity:Exp324ExecutionIdentity)->bytes:
    validate_execution_identity(identity)
    return canonical_json_bytes(asdict(identity))+b"\n"
