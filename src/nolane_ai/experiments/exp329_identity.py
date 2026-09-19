from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from typing import Any,Mapping

SCHEMA="EXP329-P02-CROSS-UPDATE-EXECUTION-IDENTITY-V1"
APPROVED_PREREGISTRATION_DIGEST="3f2e7073ac3c1c39efbcbb7ade65b4381147c90320248870125515091185fae4"
PARENT_RUN_ID=35414453731
PARENT_ARTIFACT_ID=10575945812
PARENT_ARTIFACT_ZIP_DIGEST="b15b7910abc91c4ba1b5fb3f64af3daa784fa400eae98983442c7565bf94e3d5"
PARENT_FINAL_EVIDENCE_DIGEST="04f610460ff5a681fe05b34f1d82dcb28a2fe81ec92b4026cb015841b305428a"
RECONSTRUCTION_ARTIFACT_ID=10547681681

@dataclass(frozen=True,slots=True)
class Exp329ExecutionIdentity:
    schema:str
    source_commit_sha:str
    workflow_sha256:str
    approved_preregistration_digest:str
    parent_run_id:int
    parent_final_artifact_id:int
    parent_final_artifact_zip_digest:str
    parent_final_evidence_digest:str
    reconstruction_artifact_id:int
    exp329_execution_digest:str

def _canonical(p:Mapping[str,Any])->bytes:
    return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def canonical_execution_digest(x:Exp329ExecutionIdentity)->str:
    p=asdict(x);p.pop("exp329_execution_digest",None)
    return hashlib.sha256(_canonical(p)).hexdigest()

def validate_execution_identity(x:Exp329ExecutionIdentity)->None:
    if x.schema!=SCHEMA:raise ValueError("EXP-329 identity schema")
    if len(x.source_commit_sha)!=40 or len(x.workflow_sha256)!=64:raise ValueError("EXP-329 source/workflow identity")
    if x.approved_preregistration_digest!=APPROVED_PREREGISTRATION_DIGEST:raise ValueError("EXP-329 preregistration identity")
    if x.parent_run_id!=PARENT_RUN_ID or x.parent_final_artifact_id!=PARENT_ARTIFACT_ID:raise ValueError("EXP-329 parent run identity")
    if x.parent_final_artifact_zip_digest!=PARENT_ARTIFACT_ZIP_DIGEST:raise ValueError("EXP-329 parent artifact identity")
    if x.parent_final_evidence_digest!=PARENT_FINAL_EVIDENCE_DIGEST:raise ValueError("EXP-329 parent evidence identity")
    if x.reconstruction_artifact_id!=RECONSTRUCTION_ARTIFACT_ID:raise ValueError("EXP-329 reconstruction identity")
    if canonical_execution_digest(x)!=x.exp329_execution_digest:raise ValueError("EXP-329 execution digest")
