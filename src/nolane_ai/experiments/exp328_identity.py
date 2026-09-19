from __future__ import annotations
from dataclasses import asdict,dataclass,replace
import hashlib,json
from typing import Any,Mapping

SCHEMA="EXP328-P02-ORDER-GEOMETRY-EXECUTION-IDENTITY-V1"
APPROVED_PREREGISTRATION_DIGEST="e87c20965651193f0e927b7c6969fd0d12ec3fc83b9f56485bfbe1bc4e3f03cd"
PARENT_RUN_ID=35413434081
PARENT_ARTIFACT_ID=10574837015
PARENT_ARTIFACT_ZIP_DIGEST="b4ddab033a85762c99451f6532f790fbf4cda9181ff67c1a2093613dd0156d77"
PARENT_FINAL_EVIDENCE_DIGEST="bc9999f3f6e366de8ec41b26ec265514546cf68cb82661295a31b202e3e907e2"
RECONSTRUCTION_ARTIFACT_ID=10547681681

@dataclass(frozen=True,slots=True)
class Exp328ExecutionIdentity:
    schema:str
    source_commit_sha:str
    workflow_sha256:str
    approved_preregistration_digest:str
    parent_run_id:int
    parent_final_artifact_id:int
    parent_final_artifact_zip_digest:str
    parent_final_evidence_digest:str
    reconstruction_artifact_id:int
    exp328_execution_digest:str

def _canonical(p:Mapping[str,Any])->bytes:
    return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def canonical_execution_digest(x:Exp328ExecutionIdentity)->str:
    p=asdict(x)
    p.pop("exp328_execution_digest",None)
    return hashlib.sha256(_canonical(p)).hexdigest()

def validate_execution_identity(x:Exp328ExecutionIdentity)->None:
    if x.schema!=SCHEMA:
        raise ValueError("EXP-328 identity schema")
    if len(x.source_commit_sha)!=40 or len(x.workflow_sha256)!=64:
        raise ValueError("EXP-328 source/workflow identity")
    if x.approved_preregistration_digest!=APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-328 preregistration identity")
    if x.parent_run_id!=PARENT_RUN_ID or x.parent_final_artifact_id!=PARENT_ARTIFACT_ID:
        raise ValueError("EXP-328 parent run identity")
    if x.parent_final_artifact_zip_digest!=PARENT_ARTIFACT_ZIP_DIGEST:
        raise ValueError("EXP-328 parent artifact identity")
    if x.parent_final_evidence_digest!=PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-328 parent evidence identity")
    if x.reconstruction_artifact_id!=RECONSTRUCTION_ARTIFACT_ID:
        raise ValueError("EXP-328 reconstruction identity")
    if canonical_execution_digest(x)!=x.exp328_execution_digest:
        raise ValueError("EXP-328 execution digest")
