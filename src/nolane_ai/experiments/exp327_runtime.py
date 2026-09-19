from __future__ import annotations
from dataclasses import asdict
import hashlib,json
from pathlib import Path
from typing import Any,Mapping
from .exp319_training import model_state_digest,optimizer_state_digest,rng_state_digest
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset,validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp327_contract import AUTHORIZATION_FLAGS,EXPOSURE_CHECKPOINTS,EXPOSURES_PER_WORLD,FAMILY,GROUP_IDS,GROUP_MEMBERS,LEARNING_RATE,PAIR_IDS,QUARTET_ID,TRIPLE_IDS,SubsetSnapshot,canonical_json_bytes,group_floor_pass,reduce_subset,training_schedule,validate_snapshot
from .exp327_identity import Exp327ExecutionIdentity,PARENT_EXECUTION_DIGEST,PARENT_FINAL_EVIDENCE_DIGEST,validate_execution_identity

FINAL_SCHEMA="EXP327-FINAL-EVIDENCE-V1"
def _digest(p:Mapping[str,Any])->str:return hashlib.sha256(canonical_json_bytes(p)).hexdigest()
def validate_parent_final(p:Mapping[str,Any])->None:
 if p.get("schema")!="EXP326-FINAL-EVIDENCE-V1" or p.get("decision")!="QUARTET_LEVEL_BREAK_PRESENT":raise ValueError("EXP-327 parent disposition mismatch")
 if p.get("evidence_digest")!=PARENT_FINAL_EVIDENCE_DIGEST:raise ValueError("EXP-327 parent evidence mismatch")
 q=dict(p);q.pop("evidence_digest",None)
 if _digest(q)!=PARENT_FINAL_EVIDENCE_DIGEST:raise ValueError("EXP-327 parent canonical mismatch")
 i=p.get("execution_identity")
 if not isinstance(i,Mapping) or i.get("exp326_execution_digest")!=PARENT_EXECUTION_DIGEST:raise ValueError("EXP-327 parent execution mismatch")
 if p.get("group_pass",{}).get("iterative-grid-and-maze:Q0123") is not False:raise ValueError("EXP-327 parent Q0123 failure missing")
 for k,v in AUTHORIZATION_FLAGS.items():
  if p.get(k) is not v:raise ValueError(f"EXP-327 parent authorization drift: {k}")
def _worlds():
 m={int(getattr(w,"index")):w for w in family_worlds(FAMILY)}
 if tuple(sorted(m))!=tuple(range(8)):raise ValueError("EXP-327 world population drift")
 return m
def _snapshot(gid,exposures,compiled):
 ws=tuple(_worlds()[i] for i in GROUP_MEMBERS[gid]);ind=tuple(_evaluate_subset(compiled,(w,)) for w in ws)
 s=SubsetSnapshot(gid,exposures,len(ws)*exposures,tuple(x["teacher_forced_answer_token_accuracy"] for x in ind),tuple(x["teacher_forced_full_answer_exact"] for x in ind));validate_snapshot(s)
 agg=_evaluate_subset(compiled,ws)
 return s,{"group_teacher_forced_answer_token_accuracy":agg["teacher_forced_answer_token_accuracy"],"group_teacher_forced_full_answer_exact":agg["teacher_forced_full_answer_exact"],"group_greedy_exact":agg["greedy_exact"],"group_answer_only_loss":agg["answer_only_loss"],"per_world_greedy_exact":[x["greedy_exact"] for x in ind],"per_world_answer_only_loss":[x["answer_only_loss"] for x in ind]}
def _run_group(checkpoint_path,receipt_path,selection_lock_path,gid):
 state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path);compiled,optimizer=state.compiled,state.optimizer;validate_optimizer_invariants(optimizer)
 worlds=_worlds();snaps=[];secondary={};nonfinite=0;last_grad=last_update=0.0;invalid=None
 for wi,ei,effort in training_schedule(gid):
  _,last_grad,last_update,observed=_train_one_step_with_effort(compiled,worlds[wi],optimizer=optimizer,effort=effort);nonfinite+=observed;completed=ei+1
  if nonfinite:invalid="NONFINITE_EVENT";break
  if wi==GROUP_MEMBERS[gid][-1] and completed in EXPOSURE_CHECKPOINTS:
   s,metrics=_snapshot(gid,completed,compiled);snaps.append(s);secondary[str(completed)]={**metrics,"gradient_norm_preclip":last_grad,"parameter_update_norm_ratio":last_update,"nonfinite_events":nonfinite}
 model=getattr(compiled,"model")
 return {"group_id":gid,"members":list(GROUP_MEMBERS[gid]),"group_size":len(GROUP_MEMBERS[gid]),"exposures_per_world":EXPOSURES_PER_WORLD,"learning_rate":LEARNING_RATE,"snapshots":[asdict(x) for x in snaps],"secondary":secondary,"passed":any(group_floor_pass(x) for x in snaps) if invalid is None else False,"final_state":{"model_state_digest":model_state_digest(model),"optimizer_state_digest":optimizer_state_digest(optimizer),"rng_state_digest":rng_state_digest()},"invalid_reason":invalid}
def _validate_group(r:Mapping[str,Any])->None:
 gid=r.get("group_id")
 if gid not in GROUP_MEMBERS or r.get("members")!=list(GROUP_MEMBERS[gid]) or r.get("group_size")!=len(GROUP_MEMBERS[gid]):raise ValueError("EXP-327 group identity mismatch")
 if r.get("exposures_per_world")!=EXPOSURES_PER_WORLD or r.get("learning_rate")!=LEARNING_RATE:raise ValueError("EXP-327 group geometry mismatch")
 rows=tuple(SubsetSnapshot(**x) for x in r.get("snapshots",()))
 if r.get("invalid_reason") is None and tuple(x.exposures_per_world for x in rows)!=EXPOSURE_CHECKPOINTS:raise ValueError("EXP-327 checkpoint geometry mismatch")
 for x in rows:
  validate_snapshot(x)
  if x.group_id!=gid:raise ValueError("EXP-327 snapshot identity mismatch")
 if r.get("passed") is not (any(group_floor_pass(x) for x in rows) if r.get("invalid_reason") is None else False):raise ValueError("EXP-327 pass flag mismatch")
 f=r.get("final_state")
 if not isinstance(f,Mapping) or any(not isinstance(f.get(k),str) or len(f[k])!=64 for k in ("model_state_digest","optimizer_state_digest","rng_state_digest")):raise ValueError("EXP-327 final state malformed")
def run_court(*,checkpoint_path,receipt_path,selection_lock_path,parent_final_path,execution_identity:Exp327ExecutionIdentity):
 validate_execution_identity(execution_identity);parent=json.loads(Path(parent_final_path).read_text())
 if not isinstance(parent,Mapping):raise ValueError("EXP-327 parent must be object")
 validate_parent_final(parent)
 authority=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path);reconstruction=dict(authority.reconstruction);del authority
 groups=[_run_group(checkpoint_path,receipt_path,selection_lock_path,g) for g in GROUP_IDS]
 records={g["group_id"]:tuple(SubsetSnapshot(**x) for x in g["snapshots"]) for g in groups};invalid=any(g["invalid_reason"] is not None for g in groups)
 decision,passed,violations=reduce_subset(records,invalid);failed_pairs=[g for g in PAIR_IDS if not passed.get(g,False)];failed_triples=[g for g in TRIPLE_IDS if not passed.get(g,False)]
 minimal=failed_pairs if failed_pairs else (failed_triples if failed_triples else ([QUARTET_ID] if passed and not passed[QUARTET_ID] else []))
 payload={"schema":FINAL_SCHEMA,"execution_identity":asdict(execution_identity),"reconstruction":reconstruction,"decision":decision,"groups":groups,"group_pass":{g:bool(passed.get(g,False)) for g in GROUP_IDS},"failed_pairs":failed_pairs,"failed_triples":failed_triples,"minimal_failing_groups":minimal,"minimal_failing_size":len(GROUP_MEMBERS[minimal[0]]) if minimal else None,"monotonicity_violations":[{"parent_group":p,"child_group":c} for p,c in violations],**AUTHORIZATION_FLAGS}
 for g in groups:_validate_group(g)
 payload["evidence_digest"]=_digest(payload);return payload
def validate_final_evidence(p:Mapping[str,Any])->None:
 if p.get("schema")!=FINAL_SCHEMA:raise ValueError("EXP-327 final schema mismatch")
 raw=p.get("execution_identity")
 if not isinstance(raw,Mapping):raise ValueError("EXP-327 identity missing")
 validate_execution_identity(Exp327ExecutionIdentity(**raw))
 if p.get("reconstruction")!=expected_reconstruction_payload():raise ValueError("EXP-327 reconstruction mismatch")
 groups=p.get("groups")
 if not isinstance(groups,list) or len(groups)!=len(GROUP_IDS):raise ValueError("EXP-327 group coverage mismatch")
 seen=set()
 for g in groups:_validate_group(g);seen.add(g["group_id"])
 if seen!=set(GROUP_IDS):raise ValueError("EXP-327 group coverage mismatch")
 for k,v in AUTHORIZATION_FLAGS.items():
  if p.get(k) is not v:raise ValueError(f"EXP-327 authorization drift: {k}")
 q=dict(p);claimed=q.pop("evidence_digest",None)
 if not isinstance(claimed,str) or _digest(q)!=claimed:raise ValueError("EXP-327 evidence digest mismatch")
