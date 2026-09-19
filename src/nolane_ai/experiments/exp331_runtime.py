from __future__ import annotations
from dataclasses import asdict
import hashlib,json,math
from pathlib import Path
from typing import Any,Mapping
import torch

from .exp301_scientific import forward_scientific_arm
from .exp301_training import compute_answer_only_loss
from .exp319_metrics import global_gradient_norm,nonfinite_report,parameter_update_norm_ratio,snapshot_trainable_parameters
from .exp319_training import model_state_digest,optimizer_state_digest,rng_state_digest
from .exp322_runtime import _encoded_tensors
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset,validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp331_contract import (
    AUTHORIZATION_FLAGS,FAMILY,GRADIENT_CLIP_NORM,MODES,PAIR_IDS,PAIR_MEMBERS,
    PARENT_PAIR_PASS,TARGET_NORM_SQUARED_FLOOR,PairArmResult,arm_pass,
    reduce_pair_lattice,sham_equivalent,training_schedule,validate_arm,
)
from .exp331_identity import (
    PARENT_EXP327_EVIDENCE_DIGEST,PARENT_EXP330_EVIDENCE_DIGEST,
    Exp331ExecutionIdentity,validate_execution_identity,
)

FINAL_SCHEMA="EXP331-FINAL-EVIDENCE-V1"

def _canonical(p:Mapping[str,Any])->bytes:return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _digest(p:Mapping[str,Any])->str:return hashlib.sha256(_canonical(p)).hexdigest()

def validate_exp330_parent(p:Mapping[str,Any])->None:
    if p.get("schema")!="EXP330-FINAL-EVIDENCE-V1" or p.get("decision")!="CONFLICT_PROJECTION_RESCUE":raise ValueError("EXP-331 EXP330 parent")
    if p.get("evidence_digest")!=PARENT_EXP330_EVIDENCE_DIGEST:raise ValueError("EXP-331 EXP330 evidence")
    q=dict(p);q.pop("evidence_digest",None)
    if _digest(q)!=PARENT_EXP330_EVIDENCE_DIGEST:raise ValueError("EXP-331 EXP330 canonical")
    ident=p.get("execution_identity")
    if not isinstance(ident,Mapping) or ident.get("exp330_execution_digest")!="a6428a26626b44e60748fedac93ce8011b317584f70054eddefe88826a4c96b7":raise ValueError("EXP-331 EXP330 execution")
    if p.get("arm_pass")!={"CONTROL_ALT":False,"PROJECT_CONFLICT_ALT":True,"SHAM_MEASURE_ALT":False} or p.get("sham_state_match") is not True or int(p.get("projection_event_count",0))<=0:raise ValueError("EXP-331 EXP330 result")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:raise ValueError(f"EXP-331 EXP330 authorization drift: {k}")

def validate_exp327_parent(p:Mapping[str,Any])->None:
    if p.get("schema")!="EXP327-FINAL-EVIDENCE-V1" or p.get("decision")!="PAIR_MINIMAL_FAILURE_PRESENT":raise ValueError("EXP-331 EXP327 parent")
    if p.get("evidence_digest")!=PARENT_EXP327_EVIDENCE_DIGEST:raise ValueError("EXP-331 EXP327 evidence")
    q=dict(p);q.pop("evidence_digest",None)
    if _digest(q)!=PARENT_EXP327_EVIDENCE_DIGEST:raise ValueError("EXP-331 EXP327 canonical")
    if p.get("failed_pairs")!=["P02"]:raise ValueError("EXP-331 parent failed-pair set")
    if {k:bool(p.get("group_pass",{}).get(k)) for k in PAIR_IDS}!=PARENT_PAIR_PASS:raise ValueError("EXP-331 parent pair pass")
    groups=p.get("groups")
    if not isinstance(groups,list):raise ValueError("EXP-331 parent groups")
    by={g.get("group_id"):g for g in groups if isinstance(g,Mapping)}
    if any(k not in by for k in PAIR_IDS):raise ValueError("EXP-331 parent pair coverage")
    for pair in PAIR_IDS:
        g=by[pair]
        if g.get("members")!=list(PAIR_MEMBERS[pair]) or g.get("invalid_reason") is not None:raise ValueError("EXP-331 parent pair identity")
        snaps=[x for x in g.get("snapshots",[]) if x.get("exposures_per_world")==32]
        if len(snaps)!=1:raise ValueError("EXP-331 parent pair final snapshot")
        state=g.get("final_state")
        if not isinstance(state,Mapping) or any(not isinstance(state.get(k),str) or len(state[k])!=64 for k in ("model_state_digest","optimizer_state_digest","rng_state_digest")):raise ValueError("EXP-331 parent pair state")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:raise ValueError(f"EXP-331 EXP327 authorization drift: {k}")

def _world_map()->dict[int,object]:
    worlds={int(getattr(w,"index")):w for w in family_worlds(FAMILY)}
    if tuple(sorted(worlds))!=tuple(range(8)):raise ValueError("EXP-331 world population")
    return worlds

def _backward_grads(compiled:object,world:object,effort:int)->tuple[float,tuple[torch.Tensor,...],int]:
    model=getattr(compiled,"model");params=tuple(p for p in model.parameters() if p.requires_grad);device=next(model.parameters()).device
    ids,targets,start=_encoded_tensors(world,device=device)
    logits=forward_scientific_arm(compiled,ids,effort=effort);loss=compute_answer_only_loss(logits,targets=targets,answer_start=start);loss.backward()
    report=nonfinite_report(model.parameters(),loss=loss)
    grads=tuple(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params)
    value=float(loss.detach().cpu().item())
    if not math.isfinite(value):raise ValueError("EXP-331 nonfinite measured loss")
    return value,grads,int(report.total)

def _pair_metrics(a:tuple[torch.Tensor,...],b:tuple[torch.Tensor,...])->tuple[float,float,float,float]:
    if len(a)!=len(b):raise ValueError("EXP-331 gradient geometry")
    dot=asq=bsq=0.0
    for x,y in zip(a,b):
        dot+=float(torch.sum(x*y,dtype=torch.float64).item());asq+=float(torch.sum(x*x,dtype=torch.float64).item());bsq+=float(torch.sum(y*y,dtype=torch.float64).item())
    an,bn=math.sqrt(asq),math.sqrt(bsq);cos=dot/(an*bn) if an>0 and bn>0 else 0.0
    if not all(math.isfinite(x) for x in (dot,an,bn,cos)):raise ValueError("EXP-331 nonfinite gradient metric")
    return dot,max(-1.0,min(1.0,cos)),asq,bsq

def _measured_step(compiled,source,target,*,optimizer,effort:int,project:bool,round_index:int,source_index:int,target_index:int)->dict[str,Any]:
    model=getattr(compiled,"model");params=tuple(p for p in model.parameters() if p.requires_grad);before=snapshot_trainable_parameters(model.parameters());model.train();rng=torch.get_rng_state().clone()
    optimizer.zero_grad(set_to_none=True);torch.set_rng_state(rng);source_loss,sg,snf=_backward_grads(compiled,source,effort);rng_after=torch.get_rng_state().clone()
    optimizer.zero_grad(set_to_none=True);torch.set_rng_state(rng);target_loss,tg,tnf=_backward_grads(compiled,target,effort)
    torch.set_rng_state(rng_after);optimizer.zero_grad(set_to_none=True);dot,cos,ssq,tsq=_pair_metrics(sg,tg);negative=dot<0.0;apply=bool(project and negative and tsq>TARGET_NORM_SQUARED_FLOOR);coef=dot/tsq if apply else 0.0
    for p,s,t in zip(params,sg,tg):p.grad=(s-t*coef if apply else s).clone()
    preclip=global_gradient_norm(model.parameters());applied=nonfinite_report(model.parameters(),loss=source_loss);torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP_NORM);optimizer.step();after=snapshot_trainable_parameters(model.parameters());ratio=parameter_update_norm_ratio(before,after);post=nonfinite_report(model.parameters(),loss=source_loss)
    return {"round_index":round_index,"source_world":source_index,"target_world":target_index,"effort":effort,"source_loss":source_loss,"target_loss":target_loss,"gradient_dot":dot,"gradient_cosine":cos,"source_gradient_norm":math.sqrt(ssq),"target_gradient_norm":math.sqrt(tsq),"negative_dot":negative,"projection_applied":apply,"projection_coefficient":coef,"gradient_norm_preclip":preclip,"parameter_update_norm_ratio":ratio,"nonfinite_events":snf+tnf+applied.total+post.total}

def _result(record:Mapping[str,Any])->PairArmResult:
    return PairArmResult(pair_id=str(record["pair_id"]),mode=str(record["mode"]),total_optimizer_updates=int(record["total_optimizer_updates"]),world_token_accuracies=tuple(record["world_token_accuracies"]),world_full_answer_exact=tuple(record["world_full_answer_exact"]),model_state_digest=str(record["model_state_digest"]),optimizer_state_digest=str(record["optimizer_state_digest"]),rng_state_digest=str(record["rng_state_digest"]),nonfinite_events=int(record["nonfinite_events"]),negative_dot_count=int(record["negative_dot_count"]),projection_event_count=int(record["projection_event_count"]))

def _finalize(pair_id,mode,compiled,optimizer,worlds,nonfinite,measurements,invalid):
    members=PAIR_MEMBERS[pair_id];ind=tuple(_evaluate_subset(compiled,(worlds[i],)) for i in members);model=getattr(compiled,"model")
    result=PairArmResult(pair_id,mode,64,tuple(x["teacher_forced_answer_token_accuracy"] for x in ind),tuple(x["teacher_forced_full_answer_exact"] for x in ind),model_state_digest(model),optimizer_state_digest(optimizer),rng_state_digest(),nonfinite,sum(bool(x["negative_dot"]) for x in measurements),sum(bool(x["projection_applied"]) for x in measurements));validate_arm(result)
    return {**asdict(result),"members":list(members),"measurements":measurements,"per_world_greedy_exact":[x["greedy_exact"] for x in ind],"per_world_answer_only_loss":[x["answer_only_loss"] for x in ind],"invalid_reason":invalid}

def _run_arm(checkpoint_path,receipt_path,selection_lock_path,*,pair_id:str,mode:str)->dict[str,Any]:
    state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path);compiled,optimizer=state.compiled,state.optimizer;validate_optimizer_invariants(optimizer);worlds=_world_map();nonfinite=0;invalid=None;measurements=[];members=PAIR_MEMBERS[pair_id]
    for source_index,round_index,effort in training_schedule(pair_id):
        if mode=="CONTROL":
            _,_,_,observed=_train_one_step_with_effort(compiled,worlds[source_index],optimizer=optimizer,effort=effort);nonfinite+=observed
        else:
            target_index=members[1] if source_index==members[0] else members[0]
            m=_measured_step(compiled,worlds[source_index],worlds[target_index],optimizer=optimizer,effort=effort,project=mode=="PROJECT",round_index=round_index,source_index=source_index,target_index=target_index);measurements.append(m);nonfinite+=int(m["nonfinite_events"])
        if nonfinite:invalid="NONFINITE_ARM";break
    return _finalize(pair_id,mode,compiled,optimizer,worlds,nonfinite,measurements,invalid)

def _parent_anchor(parent:Mapping[str,Any],pair_id:str)->Mapping[str,Any]:
    group=next(g for g in parent["groups"] if g["group_id"]==pair_id);snap=next(x for x in group["snapshots"] if x["exposures_per_world"]==32)
    return {"pair_id":pair_id,"members":group["members"],"passed":group["passed"],"world_token_accuracies":snap["world_token_accuracies"],"world_full_answer_exact":snap["world_full_answer_exact"],"final_state":group["final_state"]}

def _control_matches(record:Mapping[str,Any],anchor:Mapping[str,Any])->bool:
    state=anchor["final_state"]
    return record.get("mode")=="CONTROL" and record.get("members")==anchor["members"] and tuple(record.get("world_token_accuracies",()))==tuple(anchor["world_token_accuracies"]) and tuple(record.get("world_full_answer_exact",()))==tuple(anchor["world_full_answer_exact"]) and record.get("model_state_digest")==state["model_state_digest"] and record.get("optimizer_state_digest")==state["optimizer_state_digest"] and record.get("rng_state_digest")==state["rng_state_digest"] and record.get("nonfinite_events")==0 and arm_pass(_result(record)) is bool(anchor["passed"])

def run_court(*,checkpoint_path,receipt_path,selection_lock_path,exp330_parent_path,exp327_parent_path,execution_identity:Exp331ExecutionIdentity)->dict[str,Any]:
    validate_execution_identity(execution_identity);p330=json.loads(Path(exp330_parent_path).read_text());p327=json.loads(Path(exp327_parent_path).read_text())
    if not isinstance(p330,Mapping) or not isinstance(p327,Mapping):raise ValueError("EXP-331 parents")
    validate_exp330_parent(p330);validate_exp327_parent(p327);authority=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path);reconstruction=dict(authority.reconstruction);del authority
    arms=[_run_arm(checkpoint_path,receipt_path,selection_lock_path,pair_id=p,mode=m) for p in PAIR_IDS for m in MODES]
    results={f"{a['pair_id']}:{a['mode']}":_result(a) for a in arms};anchors={p:_parent_anchor(p327,p) for p in PAIR_IDS};parent_reproduced=all(_control_matches(next(a for a in arms if a["pair_id"]==p and a["mode"]=="CONTROL"),anchors[p]) for p in PAIR_IDS);invalid=any(a["invalid_reason"] is not None for a in arms)
    decision,passed,regressions=reduce_pair_lattice(results,parent_reproduced=parent_reproduced,invalid=invalid)
    payload={"schema":FINAL_SCHEMA,"execution_identity":asdict(execution_identity),"reconstruction":reconstruction,"decision":decision,"parent_exp330_evidence_digest":PARENT_EXP330_EVIDENCE_DIGEST,"parent_exp327_evidence_digest":PARENT_EXP327_EVIDENCE_DIGEST,"parent_pair_anchors":anchors,"parent_pair_reproduced":parent_reproduced,"arms":arms,"arm_pass":passed,"pair_sham_match":{p:sham_equivalent(results[f"{p}:CONTROL"],results[f"{p}:SHAM"]) for p in PAIR_IDS},"project_regressions":list(regressions),"project_pass_pairs":[p for p in PAIR_IDS if passed.get(f"{p}:PROJECT",False)],"p02_projection_event_count":results["P02:PROJECT"].projection_event_count,**AUTHORIZATION_FLAGS}
    payload["evidence_digest"]=_digest(payload);validate_final_evidence(payload);return payload

def validate_final_evidence(p:Mapping[str,Any])->None:
    if p.get("schema")!=FINAL_SCHEMA:raise ValueError("EXP-331 final schema")
    raw=p.get("execution_identity")
    if not isinstance(raw,Mapping):raise ValueError("EXP-331 identity")
    validate_execution_identity(Exp331ExecutionIdentity(**raw))
    if p.get("reconstruction")!=expected_reconstruction_payload() or p.get("parent_exp330_evidence_digest")!=PARENT_EXP330_EVIDENCE_DIGEST or p.get("parent_exp327_evidence_digest")!=PARENT_EXP327_EVIDENCE_DIGEST:raise ValueError("EXP-331 authority")
    arms=p.get("arms")
    if not isinstance(arms,list) or len(arms)!=len(PAIR_IDS)*len(MODES):raise ValueError("EXP-331 arm coverage")
    results={f"{a.get('pair_id')}:{a.get('mode')}":_result(a) for a in arms}
    if set(results)!={f"{x}:{m}" for x in PAIR_IDS for m in MODES}:raise ValueError("EXP-331 arm keys")
    invalid=any(a.get("invalid_reason") is not None for a in arms);decision,passed,regressions=reduce_pair_lattice(results,parent_reproduced=p.get("parent_pair_reproduced") is True,invalid=invalid)
    if p.get("decision")!=decision or p.get("arm_pass")!=passed or p.get("project_regressions")!=list(regressions):raise ValueError("EXP-331 reducer")
    expected_sham={pair:sham_equivalent(results[f"{pair}:CONTROL"],results[f"{pair}:SHAM"]) for pair in PAIR_IDS}
    if p.get("pair_sham_match")!=expected_sham or p.get("p02_projection_event_count")!=results["P02:PROJECT"].projection_event_count:raise ValueError("EXP-331 measurement integrity")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:raise ValueError(f"EXP-331 authorization drift: {k}")
    q=dict(p);claimed=q.pop("evidence_digest",None)
    if not isinstance(claimed,str) or _digest(q)!=claimed:raise ValueError("EXP-331 evidence digest")
