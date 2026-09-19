from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from .exp327_contract import FAMILY,GROUP_IDS,WORLD_INDICES,EXPOSURE_CHECKPOINTS,EXPOSURES_PER_WORLD,preregistration_digest

SCHEMA="EXP327-ITERATIVE-MINIMAL-SUBSET-EXECUTION-IDENTITY-V1"
PARENT_MARKER_SHA="fb66defc39bd9861b9bbaf0e6641cda6ae607b1d";PARENT_RUN_ID=35409935457;PARENT_FINAL_ARTIFACT_ID=10573847041
PARENT_FINAL_ARTIFACT_ZIP_DIGEST="92794024c87ff9fd5d1e7131c2a7404c6d843abc74b825bb96d68d6f952ebdc4"
PARENT_FINAL_JSON_SHA256="b5ccf5bb9e709c8dbc4da33e87a5bbdcaa1164f3408787f1ae423bae7a7003ed"
PARENT_FINAL_EVIDENCE_DIGEST="e66d51977856219c6ad4124847da4431cb42877f9814cfe41e1b7644c28d2bf0"
PARENT_EXECUTION_DIGEST="f62c89c9de75b3286c1093a33c17bb6295227247b9fe7892c49044f841d2df50"
RECONSTRUCTION_RUN_ID=35345351869;RECONSTRUCTION_ARTIFACT_ID=10547681681
RECONSTRUCTION_ARTIFACT_ZIP_DIGEST="c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621"
RECONSTRUCTION_CHECKPOINT_SHA256="4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
RECONSTRUCTION_RECEIPT_DIGEST="f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f"
_HEX=frozenset("0123456789abcdef")

@dataclass(frozen=True,slots=True)
class Exp327ExecutionIdentity:
 schema:str;approved_preregistration_digest:str;parent_marker_sha:str;parent_run_id:int;parent_final_artifact_id:int;parent_final_artifact_zip_digest:str;parent_final_json_sha256:str;parent_final_evidence_digest:str;parent_execution_digest:str;reconstruction_run_id:int;reconstruction_artifact_id:int;reconstruction_artifact_zip_digest:str;reconstruction_checkpoint_sha256:str;reconstruction_receipt_digest:str;source_commit_sha:str;source_tree_digest:str;workflow_sha256:str;family:str;world_indices:tuple[int,...];group_ids:tuple[str,...];exposure_checkpoints:tuple[int,...];exposures_per_world:int;device:str;exp327_execution_digest:str

def canonical_json_bytes(p:object)->bytes:return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _hex(v,n,f):
 if len(v)!=n or any(x not in _HEX for x in v):raise ValueError(f"{f} malformed")
def source_tree_digest_from_git_tree_sha(tree_sha:str)->str:
 _hex(tree_sha,40,"git tree");return hashlib.sha256(f"EXP327-GIT-TREE-V1|{tree_sha}".encode()).hexdigest()
def execution_digest(i:Exp327ExecutionIdentity)->str:
 p=asdict(i);p.pop("exp327_execution_digest",None);return hashlib.sha256(canonical_json_bytes(p)).hexdigest()
def validate_execution_identity(i:Exp327ExecutionIdentity)->None:
 if i.schema!=SCHEMA:raise ValueError("EXP-327 identity schema mismatch")
 for v,n,f in ((i.parent_marker_sha,40,"parent marker"),(i.parent_final_artifact_zip_digest,64,"parent zip"),(i.parent_final_json_sha256,64,"parent json"),(i.parent_final_evidence_digest,64,"parent evidence"),(i.parent_execution_digest,64,"parent execution"),(i.reconstruction_artifact_zip_digest,64,"reconstruction zip"),(i.reconstruction_checkpoint_sha256,64,"checkpoint"),(i.reconstruction_receipt_digest,64,"receipt"),(i.source_commit_sha,40,"source"),(i.source_tree_digest,64,"tree"),(i.workflow_sha256,64,"workflow"),(i.exp327_execution_digest,64,"execution")):_hex(v,n,f)
 if i.approved_preregistration_digest!=preregistration_digest():raise ValueError("EXP-327 preregistration mismatch")
 if (i.parent_marker_sha,i.parent_run_id,i.parent_final_artifact_id)!=(PARENT_MARKER_SHA,PARENT_RUN_ID,PARENT_FINAL_ARTIFACT_ID):raise ValueError("EXP-327 parent lineage mismatch")
 if (i.parent_final_artifact_zip_digest,i.parent_final_json_sha256,i.parent_final_evidence_digest,i.parent_execution_digest)!=(PARENT_FINAL_ARTIFACT_ZIP_DIGEST,PARENT_FINAL_JSON_SHA256,PARENT_FINAL_EVIDENCE_DIGEST,PARENT_EXECUTION_DIGEST):raise ValueError("EXP-327 parent authority mismatch")
 if (i.reconstruction_run_id,i.reconstruction_artifact_id,i.reconstruction_artifact_zip_digest,i.reconstruction_checkpoint_sha256,i.reconstruction_receipt_digest)!=(RECONSTRUCTION_RUN_ID,RECONSTRUCTION_ARTIFACT_ID,RECONSTRUCTION_ARTIFACT_ZIP_DIGEST,RECONSTRUCTION_CHECKPOINT_SHA256,RECONSTRUCTION_RECEIPT_DIGEST):raise ValueError("EXP-327 reconstruction mismatch")
 if i.family!=FAMILY or tuple(i.world_indices)!=WORLD_INDICES or tuple(i.group_ids)!=GROUP_IDS:raise ValueError("EXP-327 subset geometry mismatch")
 if tuple(i.exposure_checkpoints)!=EXPOSURE_CHECKPOINTS or i.exposures_per_world!=EXPOSURES_PER_WORLD or i.device!="cpu":raise ValueError("EXP-327 runtime geometry mismatch")
 if i.exp327_execution_digest!=execution_digest(i):raise ValueError("EXP-327 execution digest mismatch")
def build_execution_identity(*,source_commit_sha:str,git_tree_sha:str,workflow_sha256:str)->Exp327ExecutionIdentity:
 values=dict(schema=SCHEMA,approved_preregistration_digest=preregistration_digest(),parent_marker_sha=PARENT_MARKER_SHA,parent_run_id=PARENT_RUN_ID,parent_final_artifact_id=PARENT_FINAL_ARTIFACT_ID,parent_final_artifact_zip_digest=PARENT_FINAL_ARTIFACT_ZIP_DIGEST,parent_final_json_sha256=PARENT_FINAL_JSON_SHA256,parent_final_evidence_digest=PARENT_FINAL_EVIDENCE_DIGEST,parent_execution_digest=PARENT_EXECUTION_DIGEST,reconstruction_run_id=RECONSTRUCTION_RUN_ID,reconstruction_artifact_id=RECONSTRUCTION_ARTIFACT_ID,reconstruction_artifact_zip_digest=RECONSTRUCTION_ARTIFACT_ZIP_DIGEST,reconstruction_checkpoint_sha256=RECONSTRUCTION_CHECKPOINT_SHA256,reconstruction_receipt_digest=RECONSTRUCTION_RECEIPT_DIGEST,source_commit_sha=source_commit_sha,source_tree_digest=source_tree_digest_from_git_tree_sha(git_tree_sha),workflow_sha256=workflow_sha256,family=FAMILY,world_indices=WORLD_INDICES,group_ids=GROUP_IDS,exposure_checkpoints=EXPOSURE_CHECKPOINTS,exposures_per_world=EXPOSURES_PER_WORLD,device="cpu")
 p=Exp327ExecutionIdentity(**values,exp327_execution_digest="");return Exp327ExecutionIdentity(**values,exp327_execution_digest=execution_digest(p))
def canonical_identity_json_bytes(i):
 validate_execution_identity(i);return canonical_json_bytes(asdict(i))+b"\n"
