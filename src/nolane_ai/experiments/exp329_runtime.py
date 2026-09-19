from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any,Mapping,Sequence

import torch

from .exp301_scientific import forward_scientific_arm
from .exp301_training import compute_answer_only_loss,model_state_digest,optimizer_state_digest,rng_state_digest
from .exp319_metrics import nonfinite_report
from .exp322_runtime import _encoded_tensors
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset,validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp329_contract import (
    AUTHORIZATION_FLAGS,CROSS_DAMAGE_EPS,EXPOSURES_PER_WORLD,FAMILY,
    PROBE_ROUNDS,SELF_IMPROVEMENT_EPS,WORLD_INDICES,ProbeRecord,
    directed_damage,effort_for_round,reduce_cross_update,validate_probe,
)
from .exp329_identity import (
    PARENT_FINAL_EVIDENCE_DIGEST,Exp329ExecutionIdentity,validate_execution_identity,
)

FINAL_SCHEMA="EXP329-FINAL-EVIDENCE-V1"

def _canonical(payload:Mapping[str,Any])->bytes:
    return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def _digest(payload:Mapping[str,Any])->str:
    return hashlib.sha256(_canonical(payload)).hexdigest()

def validate_parent_final(payload:Mapping[str,Any])->None:
    if payload.get("schema")!="EXP328-FINAL-EVIDENCE-V1":raise ValueError("EXP-329 parent schema mismatch")
    if payload.get("decision")!="ORDER_GEOMETRY_INVARIANT_PAIR_FAILURE":raise ValueError("EXP-329 parent disposition mismatch")
    if payload.get("evidence_digest")!=PARENT_FINAL_EVIDENCE_DIGEST:raise ValueError("EXP-329 parent evidence mismatch")
    materialized=dict(payload);materialized.pop("evidence_digest",None)
    if _digest(materialized)!=PARENT_FINAL_EVIDENCE_DIGEST:raise ValueError("EXP-329 parent canonical mismatch")
    if payload.get("arm_pass")!={"ALT_0_2":False,"ALT_2_0":False,"BLOCK_0_2":False,"BLOCK_2_0":False}:
        raise ValueError("EXP-329 parent arm outcome mismatch")
    raw=payload.get("execution_identity")
    if not isinstance(raw,Mapping) or raw.get("exp328_execution_digest")!="c45b7f718d9908796b68550d6a3bf8dabf50f00ac0d32897236e8b9c0445fa14":
        raise ValueError("EXP-329 parent execution mismatch")
    for key,expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:raise ValueError(f"EXP-329 parent authorization drift: {key}")

def _world_map()->dict[int,object]:
    worlds={int(getattr(w,"index")):w for w in family_worlds(FAMILY)}
    if tuple(sorted(worlds))!=tuple(range(8)):raise ValueError("EXP-329 world population drift")
    return worlds

def _loss_at_effort(compiled:object,world:object,effort:int)->float:
    model=getattr(compiled,"model")
    device=next(model.parameters()).device
    input_ids,targets,answer_start=_encoded_tensors(world,device=device)
    was_training=bool(model.training)
    model.eval()
    with torch.inference_mode():
        logits=forward_scientific_arm(compiled,input_ids,effort=effort)
        loss=compute_answer_only_loss(logits,targets=targets,answer_start=answer_start)
        value=float(loss.detach().cpu().item())
    model.train(was_training)
    if not math.isfinite(value):raise ValueError("EXP-329 nonfinite probe loss")
    return value

def _gradient_snapshot(compiled:object,world:object,effort:int)->tuple[float,tuple[torch.Tensor,...],int]:
    model=getattr(compiled,"model")
    device=next(model.parameters()).device
    input_ids,targets,answer_start=_encoded_tensors(world,device=device)
    model.train()
    model.zero_grad(set_to_none=True)
    logits=forward_scientific_arm(compiled,input_ids,effort=effort)
    loss=compute_answer_only_loss(logits,targets=targets,answer_start=answer_start)
    loss.backward()
    report=nonfinite_report(model.parameters(),loss=loss)
    grads=[]
    for parameter in model.parameters():
        if not parameter.requires_grad:continue
        if parameter.grad is None:
            grads.append(torch.zeros_like(parameter,device="cpu"))
        else:
            grads.append(parameter.grad.detach().to(device="cpu",dtype=torch.float64).clone())
    value=float(loss.detach().cpu().item())
    model.zero_grad(set_to_none=True)
    if not math.isfinite(value):raise ValueError("EXP-329 nonfinite gradient loss")
    return value,tuple(grads),int(report.total)

def _gradient_pair_metrics(compiled:object,w0:object,w2:object,effort:int)->tuple[float,float,float,float,float,float,int]:
    rng=torch.get_rng_state().clone()
    try:
        torch.set_rng_state(rng)
        l0,g0,nf0=_gradient_snapshot(compiled,w0,effort)
        torch.set_rng_state(rng)
        l2,g2,nf2=_gradient_snapshot(compiled,w2,effort)
    finally:
        torch.set_rng_state(rng)
    if len(g0)!=len(g2):raise ValueError("EXP-329 gradient geometry mismatch")
    dot=n0sq=n2sq=0.0
    for a,b in zip(g0,g2):
        dot+=float(torch.sum(a*b).item())
        n0sq+=float(torch.sum(a*a).item())
        n2sq+=float(torch.sum(b*b).item())
    n0=math.sqrt(n0sq);n2=math.sqrt(n2sq)
    cosine=dot/(n0*n2) if n0>0.0 and n2>0.0 else 0.0
    if not all(math.isfinite(x) for x in (dot,n0,n2,cosine)):raise ValueError("EXP-329 nonfinite gradient metric")
    cosine=max(-1.0,min(1.0,cosine))
    return l0,l2,dot,cosine,n0,n2,nf0+nf2

def _directed_update_probe(
    compiled:object,
    optimizer:torch.optim.Optimizer,
    *,
    source_world:object,
    target_world:object,
    effort:int,
    source_loss_pre:float,
    target_loss_pre:float,
)->tuple[float,float,int]:
    rng=torch.get_rng_state().clone()
    branch_compiled,branch_optimizer=copy.deepcopy((compiled,optimizer))
    try:
        torch.set_rng_state(rng)
        _,_,_,nonfinite=_train_one_step_with_effort(
            branch_compiled,source_world,optimizer=branch_optimizer,effort=effort
        )
        source_post=_loss_at_effort(branch_compiled,source_world,effort)
        target_post=_loss_at_effort(branch_compiled,target_world,effort)
    finally:
        torch.set_rng_state(rng)
    return source_post-source_loss_pre,target_post-target_loss_pre,int(nonfinite)

def _probe(compiled:object,optimizer:torch.optim.Optimizer,worlds:Mapping[int,object],round_index:int)->ProbeRecord:
    effort=effort_for_round(round_index)
    l0,l2,dot,cosine,n0,n2,gradient_nonfinite=_gradient_pair_metrics(
        compiled,worlds[0],worlds[2],effort
    )
    w0_self,w0_cross,w0_nf=_directed_update_probe(
        compiled,optimizer,source_world=worlds[0],target_world=worlds[2],effort=effort,
        source_loss_pre=l0,target_loss_pre=l2,
    )
    w2_self,w2_cross,w2_nf=_directed_update_probe(
        compiled,optimizer,source_world=worlds[2],target_world=worlds[0],effort=effort,
        source_loss_pre=l2,target_loss_pre=l0,
    )
    record=ProbeRecord(
        round_index=round_index,effort=effort,
        world0_loss_pre=l0,world2_loss_pre=l2,
        gradient_dot=dot,gradient_cosine=cosine,
        world0_gradient_norm=n0,world2_gradient_norm=n2,
        world0_update_self_delta=w0_self,
        world0_update_cross_delta_on_world2=w0_cross,
        world2_update_self_delta=w2_self,
        world2_update_cross_delta_on_world0=w2_cross,
        world0_direct_damage=directed_damage(self_delta=w0_self,cross_delta=w0_cross),
        world2_direct_damage=directed_damage(self_delta=w2_self,cross_delta=w2_cross),
        nonfinite_events=gradient_nonfinite+w0_nf+w2_nf,
    )
    validate_probe(record)
    return record

def run_court(
    *,
    checkpoint_path:str|Path,
    receipt_path:str|Path,
    selection_lock_path:str|Path,
    parent_final_path:str|Path,
    execution_identity:Exp329ExecutionIdentity,
)->dict[str,Any]:
    validate_execution_identity(execution_identity)
    parent=json.loads(Path(parent_final_path).read_text(encoding="utf-8"))
    if not isinstance(parent,Mapping):raise ValueError("EXP-329 parent must be object")
    validate_parent_final(parent)
    state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    compiled,optimizer=state.compiled,state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds=_world_map()
    probes=[]
    main_nonfinite=0
    invalid_reason=None
    for round_index in range(EXPOSURES_PER_WORLD):
        if round_index in PROBE_ROUNDS:
            try:
                probes.append(_probe(compiled,optimizer,worlds,round_index))
            except (RuntimeError,ValueError) as exc:
                invalid_reason=f"PROBE_ERROR:{type(exc).__name__}"
                break
        effort=effort_for_round(round_index)
        for world_index in WORLD_INDICES:
            _,_,_,observed=_train_one_step_with_effort(
                compiled,worlds[world_index],optimizer=optimizer,effort=effort
            )
            main_nonfinite+=observed
            if main_nonfinite:
                invalid_reason="NONFINITE_MAIN_REPLAY"
                break
        if invalid_reason is not None:break

    individual=tuple(_evaluate_subset(compiled,(worlds[i],)) for i in WORLD_INDICES)
    token=tuple(x["teacher_forced_answer_token_accuracy"] for x in individual)
    full=tuple(x["teacher_forced_full_answer_exact"] for x in individual)
    decision,w0_rounds,w2_rounds=reduce_cross_update(
        tuple(probes),
        final_token_accuracies=token,
        final_full_answer_exact=full,
        invalid=invalid_reason is not None or main_nonfinite!=0,
    )
    model=getattr(compiled,"model")
    payload={
        "schema":FINAL_SCHEMA,
        "execution_identity":asdict(execution_identity),
        "reconstruction":dict(state.reconstruction),
        "decision":decision,
        "probe_rounds":list(PROBE_ROUNDS),
        "cross_damage_epsilon":CROSS_DAMAGE_EPS,
        "self_improvement_epsilon":SELF_IMPROVEMENT_EPS,
        "probes":[asdict(x) for x in probes],
        "world0_damage_rounds":list(w0_rounds),
        "world2_damage_rounds":list(w2_rounds),
        "negative_gradient_dot_rounds":[x.round_index for x in probes if x.gradient_dot<0.0],
        "final_reproduction":{
            "world_token_accuracies":list(token),
            "world_full_answer_exact":list(full),
            "per_world_greedy_exact":[x["greedy_exact"] for x in individual],
            "per_world_answer_only_loss":[x["answer_only_loss"] for x in individual],
            "main_nonfinite_events":main_nonfinite,
            "invalid_reason":invalid_reason,
        },
        "final_state":{
            "model_state_digest":model_state_digest(model),
            "optimizer_state_digest":optimizer_state_digest(optimizer),
            "rng_state_digest":rng_state_digest(),
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"]=_digest(payload)
    validate_final_evidence(payload)
    return payload

def validate_final_evidence(payload:Mapping[str,Any])->None:
    if payload.get("schema")!=FINAL_SCHEMA:raise ValueError("EXP-329 final schema")
    raw=payload.get("execution_identity")
    if not isinstance(raw,Mapping):raise ValueError("EXP-329 identity missing")
    validate_execution_identity(Exp329ExecutionIdentity(**raw))
    if payload.get("reconstruction")!=expected_reconstruction_payload():raise ValueError("EXP-329 reconstruction mismatch")
    probes=payload.get("probes")
    if not isinstance(probes,list):raise ValueError("EXP-329 probes missing")
    records=tuple(ProbeRecord(**x) for x in probes)
    final=payload.get("final_reproduction")
    if not isinstance(final,Mapping):raise ValueError("EXP-329 final reproduction missing")
    decision,w0,w2=reduce_cross_update(
        records,
        final_token_accuracies=tuple(final.get("world_token_accuracies",())),
        final_full_answer_exact=tuple(final.get("world_full_answer_exact",())),
        invalid=final.get("invalid_reason") is not None or final.get("main_nonfinite_events")!=0,
    )
    if payload.get("decision")!=decision or payload.get("world0_damage_rounds")!=list(w0) or payload.get("world2_damage_rounds")!=list(w2):
        raise ValueError("EXP-329 reducer mismatch")
    for key,expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:raise ValueError(f"EXP-329 authorization drift: {key}")
    materialized=dict(payload);claimed=materialized.pop("evidence_digest",None)
    if not isinstance(claimed,str) or _digest(materialized)!=claimed:raise ValueError("EXP-329 evidence digest mismatch")
