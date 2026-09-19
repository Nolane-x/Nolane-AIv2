from __future__ import annotations
import copy
from dataclasses import asdict
import hashlib,json,math
from pathlib import Path
from typing import Any,Mapping
import torch

from .exp301_scientific import forward_scientific_arm
from .exp301_training import compute_answer_only_loss
from .exp319_metrics import nonfinite_report
from .exp319_training import model_state_digest,optimizer_state_digest,rng_state_digest
from .exp322_runtime import _encoded_tensors
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset,validate_optimizer_invariants
from .exp326_runtime import _train_one_step_with_effort
from .exp329_runtime import _gradient_snapshot,_loss_at_effort,_world_map
from .exp330_contract import (
    AUTHORIZATION_FLAGS,CROSS_DAMAGE_EPS,EXPOSURES_PER_WORLD,PARAMETER_GROUPS,
    PARENT_EVIDENCE_DIGEST,PARENT_WORLD0_DAMAGE_ROUNDS,PARENT_WORLD2_DAMAGE_ROUNDS,
    PROBE_ROUNDS,SELF_IMPROVEMENT_EPS,WORLD_INDICES,GroupProbeRecord,directed_damage,
    effort_for_round,exact_parent_reproduction,parameter_group_for_name,
    reduce_group_localization,validate_probe,
)
from .exp330_identity import Exp330ExecutionIdentity,validate_execution_identity

FINAL_SCHEMA="EXP330-FINAL-EVIDENCE-V1"
PARENT_EXECUTION_DIGEST="bd8fe650a7c61bcec66ba4914887b102c8548b47dfaea351633629349b6ff30e"

def _canonical(p:Mapping[str,Any])->bytes:
    return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _digest(p:Mapping[str,Any])->str:return hashlib.sha256(_canonical(p)).hexdigest()

def validate_parent_final(p:Mapping[str,Any])->None:
    if p.get("schema")!="EXP329-FINAL-EVIDENCE-V1" or p.get("decision")!="BIDIRECTIONAL_LOCAL_CROSS_DAMAGE":raise ValueError("EXP-330 parent identity")
    if p.get("evidence_digest")!=PARENT_EVIDENCE_DIGEST:raise ValueError("EXP-330 parent evidence")
    q=dict(p);q.pop("evidence_digest",None)
    if _digest(q)!=PARENT_EVIDENCE_DIGEST:raise ValueError("EXP-330 parent canonical")
    if tuple(p.get("world0_damage_rounds",()))!=PARENT_WORLD0_DAMAGE_ROUNDS or tuple(p.get("world2_damage_rounds",()))!=PARENT_WORLD2_DAMAGE_ROUNDS:raise ValueError("EXP-330 parent damage geometry")
    ident=p.get("execution_identity")
    if not isinstance(ident,Mapping) or ident.get("exp329_execution_digest")!=PARENT_EXECUTION_DIGEST:raise ValueError("EXP-330 parent execution")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:raise ValueError(f"EXP-330 parent authorization drift: {k}")

def _partition(compiled:object)->dict[str,tuple[str,...]]:
    model=getattr(compiled,"model");out={g:[] for g in PARAMETER_GROUPS};count=0
    for name,p in model.named_parameters():
        if p.requires_grad:
            out[parameter_group_for_name(name)].append(name);count+=p.numel()
    if count!=10_000_000 or any(not out[g] for g in PARAMETER_GROUPS):raise ValueError("EXP-330 parameter partition drift")
    return {g:tuple(v) for g,v in out.items()}

def _group_grad_metrics(compiled:object,w0:object,w2:object,effort:int,partition:Mapping[str,tuple[str,...]])->tuple[dict[str,tuple[float,float,float,float]],int]:
    rng=torch.get_rng_state().clone()
    try:
        torch.set_rng_state(rng);_,g0,n0f=_gradient_snapshot(compiled,w0,effort)
        torch.set_rng_state(rng);_,g2,n2f=_gradient_snapshot(compiled,w2,effort)
    finally:torch.set_rng_state(rng)
    names=[n for n,p in getattr(compiled,"model").named_parameters() if p.requires_grad]
    if len(names)!=len(g0) or len(g0)!=len(g2):raise ValueError("EXP-330 gradient geometry")
    d0=dict(zip(names,g0));d2=dict(zip(names,g2));out={}
    for group in PARAMETER_GROUPS:
        dot=a2=b2=0.0
        for name in partition[group]:
            a,b=d0[name],d2[name];dot+=float(torch.sum(a*b).item());a2+=float(torch.sum(a*a).item());b2+=float(torch.sum(b*b).item())
        na,nb=math.sqrt(a2),math.sqrt(b2);cos=dot/(na*nb) if na>0 and nb>0 else 0.0
        if not all(math.isfinite(x) for x in (dot,na,nb,cos)):raise ValueError("EXP-330 nonfinite group gradient")
        out[group]=(dot,max(-1.0,min(1.0,cos)),na,nb)
    return out,n0f+n2f

def _masked_step(compiled:object,world:object,optimizer:torch.optim.Optimizer,effort:int,group:str)->int:
    model=getattr(compiled,"model");device=next(model.parameters()).device
    ids,targets,start=_encoded_tensors(world,device=device);model.train();optimizer.zero_grad(set_to_none=True)
    loss=compute_answer_only_loss(forward_scientific_arm(compiled,ids,effort=effort),targets=targets,answer_start=start);loss.backward()
    before=nonfinite_report(model.parameters(),loss=loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for name,p in model.named_parameters():
        if p.requires_grad and parameter_group_for_name(name) != group:p.grad=None
    optimizer.step();after=nonfinite_report(model.parameters(),loss=loss)
    return int(before.total+after.total)

def _directed(compiled:object,optimizer:torch.optim.Optimizer,source:object,target:object,effort:int,group:str,source_pre:float,target_pre:float)->tuple[float,float,int]:
    rng=torch.get_rng_state().clone();branch,branch_opt=copy.deepcopy((compiled,optimizer))
    try:
        torch.set_rng_state(rng);nf=_masked_step(branch,source,branch_opt,effort,group)
        source_post=_loss_at_effort(branch,source,effort);target_post=_loss_at_effort(branch,target,effort)
    finally:torch.set_rng_state(rng)
    return source_post-source_pre,target_post-target_pre,nf

def _probe(compiled:object,optimizer:torch.optim.Optimizer,worlds:Mapping[int,object],partition:Mapping[str,tuple[str,...]],round_index:int)->tuple[GroupProbeRecord,...]:
    effort=effort_for_round(round_index);metrics,gnf=_group_grad_metrics(compiled,worlds[0],worlds[2],effort,partition)
    l0=_loss_at_effort(compiled,worlds[0],effort);l2=_loss_at_effort(compiled,worlds[2],effort);rows=[]
    for group in PARAMETER_GROUPS:
        dot,cos,n0,n2=metrics[group]
        s0,c02,nf0=_directed(compiled,optimizer,worlds[0],worlds[2],effort,group,l0,l2)
        s2,c20,nf2=_directed(compiled,optimizer,worlds[2],worlds[0],effort,group,l2,l0)
        row=GroupProbeRecord(round_index,effort,group,dot,cos,n0,n2,s0,c02,s2,c20,directed_damage(self_delta=s0,cross_delta=c02),directed_damage(self_delta=s2,cross_delta=c20),gnf+nf0+nf2)
        validate_probe(row);rows.append(row)
    return tuple(rows)

def run_court(*,checkpoint_path:str|Path,receipt_path:str|Path,selection_lock_path:str|Path,parent_final_path:str|Path,execution_identity:Exp330ExecutionIdentity)->dict[str,Any]:
    validate_execution_identity(execution_identity);parent=json.loads(Path(parent_final_path).read_text())
    if not isinstance(parent,Mapping):raise ValueError("EXP-330 parent must be object")
    validate_parent_final(parent);state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    compiled,optimizer=state.compiled,state.optimizer;validate_optimizer_invariants(optimizer);partition=_partition(compiled);worlds=_world_map()
    probes=[];main_nf=0;invalid=None
    for r in range(EXPOSURES_PER_WORLD):
        if r in PROBE_ROUNDS:
            try:probes.extend(_probe(compiled,optimizer,worlds,partition,r))
            except (RuntimeError,ValueError) as exc:invalid=f"PROBE_ERROR:{type(exc).__name__}";break
        effort=effort_for_round(r)
        for wi in WORLD_INDICES:
            _,_,_,nf=_train_one_step_with_effort(compiled,worlds[wi],optimizer=optimizer,effort=effort);main_nf+=nf
            if main_nf:invalid="NONFINITE_MAIN_REPLAY";break
        if invalid is not None:break
    individual=tuple(_evaluate_subset(compiled,(worlds[i],)) for i in WORLD_INDICES)
    reproduction={"world_token_accuracies":[x["teacher_forced_answer_token_accuracy"] for x in individual],"world_full_answer_exact":[x["teacher_forced_full_answer_exact"] for x in individual],"per_world_greedy_exact":[x["greedy_exact"] for x in individual],"per_world_answer_only_loss":[x["answer_only_loss"] for x in individual],"main_nonfinite_events":main_nf,"invalid_reason":invalid}
    model=getattr(compiled,"model");final_state={"model_state_digest":model_state_digest(model),"optimizer_state_digest":optimizer_state_digest(optimizer),"rng_state_digest":rng_state_digest()}
    reproduced=exact_parent_reproduction(reproduction,final_state);decision,w0g,w2g,both=reduce_group_localization(tuple(probes),parent_reproduced=reproduced,invalid=invalid is not None or main_nf!=0)
    rounds={g:{"world0_update_harms_world2":[x.round_index for x in probes if x.parameter_group==g and x.world0_direct_damage],"world2_update_harms_world0":[x.round_index for x in probes if x.parameter_group==g and x.world2_direct_damage]} for g in PARAMETER_GROUPS}
    payload={"schema":FINAL_SCHEMA,"execution_identity":asdict(execution_identity),"reconstruction":dict(state.reconstruction),"parent_evidence_digest":PARENT_EVIDENCE_DIGEST,"decision":decision,"parameter_groups":list(PARAMETER_GROUPS),"parameter_partition":{g:list(v) for g,v in partition.items()},"probe_rounds":list(PROBE_ROUNDS),"cross_damage_epsilon":CROSS_DAMAGE_EPS,"self_improvement_epsilon":SELF_IMPROVEMENT_EPS,"probes":[asdict(x) for x in probes],"world0_damage_groups":list(w0g),"world2_damage_groups":list(w2g),"bidirectional_damage_groups":list(both),"damage_rounds_by_group":rounds,"parent_reproduced":reproduced,"final_reproduction":reproduction,"final_state":final_state,**AUTHORIZATION_FLAGS}
    payload["evidence_digest"]=_digest(payload);validate_final_evidence(payload);return payload

def validate_final_evidence(p:Mapping[str,Any])->None:
    if p.get("schema")!=FINAL_SCHEMA:raise ValueError("EXP-330 final schema")
    raw=p.get("execution_identity")
    if not isinstance(raw,Mapping):raise ValueError("EXP-330 identity missing")
    validate_execution_identity(Exp330ExecutionIdentity(**raw))
    if p.get("reconstruction")!=expected_reconstruction_payload() or p.get("parent_evidence_digest")!=PARENT_EVIDENCE_DIGEST:raise ValueError("EXP-330 authority mismatch")
    if p.get("parameter_groups")!=list(PARAMETER_GROUPS):raise ValueError("EXP-330 group order")
    partition=p.get("parameter_partition")
    if not isinstance(partition,Mapping) or tuple(partition)!=PARAMETER_GROUPS:raise ValueError("EXP-330 partition")
    seen=set()
    for g in PARAMETER_GROUPS:
        names=partition.get(g)
        if not isinstance(names,list) or not names:raise ValueError("EXP-330 empty partition")
        for name in names:
            if not isinstance(name,str) or parameter_group_for_name(name)!=g or name in seen:raise ValueError("EXP-330 partition drift")
            seen.add(name)
    raw_probes=p.get("probes");rep=p.get("final_reproduction");state=p.get("final_state")
    if not isinstance(raw_probes,list) or not isinstance(rep,Mapping) or not isinstance(state,Mapping):raise ValueError("EXP-330 evidence missing")
    probes=tuple(GroupProbeRecord(**x) for x in raw_probes);reproduced=exact_parent_reproduction(rep,state)
    decision,w0g,w2g,both=reduce_group_localization(probes,parent_reproduced=reproduced,invalid=rep.get("invalid_reason") is not None or rep.get("main_nonfinite_events")!=0)
    if p.get("parent_reproduced") is not reproduced or p.get("decision")!=decision or p.get("world0_damage_groups")!=list(w0g) or p.get("world2_damage_groups")!=list(w2g) or p.get("bidirectional_damage_groups")!=list(both):raise ValueError("EXP-330 reducer mismatch")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:raise ValueError(f"EXP-330 authorization drift: {k}")
    q=dict(p);claimed=q.pop("evidence_digest",None)
    if not isinstance(claimed,str) or _digest(q)!=claimed:raise ValueError("EXP-330 evidence digest mismatch")
