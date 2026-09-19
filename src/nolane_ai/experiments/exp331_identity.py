from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from typing import Any,Mapping

SCHEMA="EXP331-PAIR-LATTICE-PROJECTION-EXECUTION-IDENTITY-V1"
APPROVED_PREREGISTRATION_DIGEST="09ef36245d8048bd5bf3810b518c3dbbad5f3e3d84c4bff4786c409515fce167"
PARENT_EXP330_RUN_ID=35419828278
PARENT_EXP330_ARTIFACT_ID=10577706134
PARENT_EXP330_ZIP_DIGEST="cadf92521948973bb319e8280934e559526616097c75f74a1fc1a90d02c33ab7"
PARENT_EXP330_EVIDENCE_DIGEST="6a48ad66fbe9b07a54f1aca0ad8ed5b92759101817543d5fa4c228cddf37fae3"
PARENT_EXP327_RUN_ID=35413434081
PARENT_EXP327_ARTIFACT_ID=10574837015
PARENT_EXP327_ZIP_DIGEST="b4ddab033a85762c99451f6532f790fbf4cda9181ff67c1a2093613dd0156d77"
PARENT_EXP327_EVIDENCE_DIGEST="bc9999f3f6e366de8ec41b26ec265514546cf68cb82661295a31b202e3e907e2"
RECONSTRUCTION_ARTIFACT_ID=10547681681

@dataclass(frozen=True,slots=True)
class Exp331ExecutionIdentity:
    schema:str
    source_commit_sha:str
    workflow_sha256:str
    approved_preregistration_digest:str
    parent_exp330_run_id:int
    parent_exp330_artifact_id:int
    parent_exp330_zip_digest:str
    parent_exp330_evidence_digest:str
    parent_exp327_run_id:int
    parent_exp327_artifact_id:int
    parent_exp327_zip_digest:str
    parent_exp327_evidence_digest:str
    reconstruction_artifact_id:int
    exp331_execution_digest:str

def _canonical(p:Mapping[str,Any])->bytes:return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def canonical_execution_digest(x:Exp331ExecutionIdentity)->str:
    p=asdict(x);p.pop("exp331_execution_digest",None);return hashlib.sha256(_canonical(p)).hexdigest()
def validate_execution_identity(x:Exp331ExecutionIdentity)->None:
    if x.schema!=SCHEMA:raise ValueError("EXP-331 identity schema")
    if len(x.source_commit_sha)!=40 or len(x.workflow_sha256)!=64:raise ValueError("EXP-331 source/workflow identity")
    if x.approved_preregistration_digest!=APPROVED_PREREGISTRATION_DIGEST:raise ValueError("EXP-331 preregistration identity")
    if x.parent_exp330_run_id!=PARENT_EXP330_RUN_ID or x.parent_exp330_artifact_id!=PARENT_EXP330_ARTIFACT_ID or x.parent_exp330_zip_digest!=PARENT_EXP330_ZIP_DIGEST or x.parent_exp330_evidence_digest!=PARENT_EXP330_EVIDENCE_DIGEST:raise ValueError("EXP-331 EXP330 parent identity")
    if x.parent_exp327_run_id!=PARENT_EXP327_RUN_ID or x.parent_exp327_artifact_id!=PARENT_EXP327_ARTIFACT_ID or x.parent_exp327_zip_digest!=PARENT_EXP327_ZIP_DIGEST or x.parent_exp327_evidence_digest!=PARENT_EXP327_EVIDENCE_DIGEST:raise ValueError("EXP-331 EXP327 parent identity")
    if x.reconstruction_artifact_id!=RECONSTRUCTION_ARTIFACT_ID:raise ValueError("EXP-331 reconstruction identity")
    if canonical_execution_digest(x)!=x.exp331_execution_digest:raise ValueError("EXP-331 execution digest")
