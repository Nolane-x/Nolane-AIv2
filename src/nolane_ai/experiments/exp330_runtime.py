from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any,Mapping

import torch

from .exp301_scientific import forward_scientific_arm
from .exp301_training import compute_answer_only_loss
from .exp319_metrics import (
    global_gradient_norm,nonfinite_report,parameter_update_norm_ratio,snapshot_trainable_parameters,
)
from .exp319_training import model_state_digest,optimizer_state_digest,rng_state_digest
from .exp322_runtime import _encoded_tensors
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset,validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp330_contract import (
    ARM_IDS,AUTHORIZATION_FLAGS,CONTROL_ARM,EXPOSURES_PER_WORLD,FAMILY,
    GRADIENT_CLIP_NORM,PROJECT_ARM,SHAM_ARM,TARGET_NORM_SQUARED_FLOOR,
    WORLD_INDICES,ArmResult,effort_for_round,reduce_projection,sham_equivalent,
    training_schedule,validate_arm,
)
from .exp330_identity import (
    PARENT_FINAL_EVIDENCE_DIGEST,Exp330ExecutionIdentity,validate_execution_identity,
)

FINAL_SCHEMA="EXP330-FINAL-EVIDENCE-V1"

def _canonical(payload:Mapping[str,Any])->bytes:
    return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def _digest(payload:Mapping[str,Any])->str:
    return hashlib.sha256(_canonical(payload)).hexdigest()

def validate_parent_final(payload:Mapping[str,Any])->None:
    if payload.get("schema")!="EXP329-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-330 parent schema mismatch")
    if payload.get("decision")!="BIDIRECTIONAL_LOCAL_CROSS_DAMAGE":
        raise ValueError("EXP-330 parent disposition mismatch")
    if payload.get("evidence_digest")!=PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-330 parent evidence mismatch")
    materialized=dict(payload);materialized.pop("evidence_digest",None)
    if _digest(materialized)!=PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-330 parent canonical mismatch")
    raw=payload.get("execution_identity")
    if not isinstance(raw,Mapping) or raw.get("exp329_execution_digest")!="bd8fe650a7c61bcec66ba4914887b102c8548b47dfaea351633629349b6ff30e":
        raise ValueError("EXP-330 parent execution mismatch")
    if payload.get("world0_damage_rounds")!=[4,8,16,24,31]:
        raise ValueError("EXP-330 parent world0 damage localization mismatch")
    if payload.get("world2_damage_rounds")!=[1,4,8,16,24,31]:
        raise ValueError("EXP-330 parent world2 damage localization mismatch")
    final=payload.get("final_reproduction")
    if not isinstance(final,Mapping):
        raise ValueError("EXP-330 parent final reproduction missing")
    if final.get("world_token_accuracies")!=[0.5,1.0] or final.get("world_full_answer_exact")!=[0.0,1.0]:
        raise ValueError("EXP-330 parent final reproduction mismatch")
    for key,expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-330 parent authorization drift: {key}")

def _world_map()->dict[int,object]:
    worlds={int(getattr(w,"index")):w for w in family_worlds(FAMILY)}
    if tuple(sorted(worlds))!=tuple(range(8)):
        raise ValueError("EXP-330 world population drift")
    return worlds

def _backward_grads(compiled:object,world:object,effort:int)->tuple[float,tuple[torch.Tensor,...],int]:
    model=getattr(compiled,"model")
    params=tuple(p for p in model.parameters() if p.requires_grad)
    device=next(model.parameters()).device
    input_ids,targets,answer_start=_encoded_tensors(world,device=device)
    logits=forward_scientific_arm(compiled,input_ids,effort=effort)
    loss=compute_answer_only_loss(logits,targets=targets,answer_start=answer_start)
    loss.backward()
    report=nonfinite_report(model.parameters(),loss=loss)
    grads=tuple(
        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for p in params
    )
    value=float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise ValueError("EXP-330 nonfinite measured loss")
    return value,grads,int(report.total)

def _pair_metrics(source_grads:tuple[torch.Tensor,...],target_grads:tuple[torch.Tensor,...])->tuple[float,float,float,float]:
    if len(source_grads)!=len(target_grads):
        raise ValueError("EXP-330 gradient geometry mismatch")
    dot=source_sq=target_sq=0.0
    for source,target in zip(source_grads,target_grads):
        dot+=float(torch.sum(source*target,dtype=torch.float64).item())
        source_sq+=float(torch.sum(source*source,dtype=torch.float64).item())
        target_sq+=float(torch.sum(target*target,dtype=torch.float64).item())
    source_norm=math.sqrt(source_sq);target_norm=math.sqrt(target_sq)
    cosine=dot/(source_norm*target_norm) if source_norm>0.0 and target_norm>0.0 else 0.0
    if not all(math.isfinite(x) for x in (dot,source_norm,target_norm,cosine)):
        raise ValueError("EXP-330 nonfinite gradient geometry")
    return dot,max(-1.0,min(1.0,cosine)),source_sq,target_sq

def _measured_step(
    compiled:object,
    source_world:object,
    target_world:object,
    *,
    optimizer:torch.optim.Optimizer,
    effort:int,
    project:bool,
    round_index:int,
    source_world_index:int,
)->dict[str,Any]:
    model=getattr(compiled,"model")
    params=tuple(p for p in model.parameters() if p.requires_grad)
    before=snapshot_trainable_parameters(model.parameters())
    model.train()
    rng_start=torch.get_rng_state().clone()

    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_start)
    source_loss,source_grads,source_nonfinite=_backward_grads(compiled,source_world,effort)
    rng_after_source=torch.get_rng_state().clone()

    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_start)
    target_loss,target_grads,target_nonfinite=_backward_grads(compiled,target_world,effort)

    torch.set_rng_state(rng_after_source)
    optimizer.zero_grad(set_to_none=True)
    dot,cosine,source_sq,target_sq=_pair_metrics(source_grads,target_grads)
    negative=dot<0.0
    apply_projection=bool(project and negative and target_sq>TARGET_NORM_SQUARED_FLOOR)
    coefficient=dot/target_sq if apply_projection else 0.0
    for parameter,source_grad,target_grad in zip(params,source_grads,target_grads):
        applied=source_grad-target_grad*coefficient if apply_projection else source_grad
        parameter.grad=applied.clone()

    preclip=global_gradient_norm(model.parameters())
    applied_report=nonfinite_report(model.parameters(),loss=source_loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP_NORM)
    optimizer.step()
    after=snapshot_trainable_parameters(model.parameters())
    update_ratio=parameter_update_norm_ratio(before,after)
    after_report=nonfinite_report(model.parameters(),loss=source_loss)
    observed=source_nonfinite+target_nonfinite+applied_report.total+after_report.total

    return {
        "round_index":round_index,
        "source_world":source_world_index,
        "target_world":2 if source_world_index==0 else 0,
        "effort":effort,
        "source_loss":source_loss,
        "target_loss":target_loss,
        "gradient_dot":dot,
        "gradient_cosine":cosine,
        "source_gradient_norm":math.sqrt(source_sq),
        "target_gradient_norm":math.sqrt(target_sq),
        "negative_dot":negative,
        "projection_applied":apply_projection,
        "projection_coefficient":coefficient,
        "gradient_norm_preclip":preclip,
        "parameter_update_norm_ratio":update_ratio,
        "nonfinite_events":observed,
    }

def _final_metrics(compiled:object,worlds:Mapping[int,object])->tuple[tuple[float,float],tuple[float,float],dict[str,Any]]:
    individual=tuple(_evaluate_subset(compiled,(worlds[i],)) for i in WORLD_INDICES)
    token=tuple(x["teacher_forced_answer_token_accuracy"] for x in individual)
    exact=tuple(x["teacher_forced_full_answer_exact"] for x in individual)
    secondary={
        "per_world_greedy_exact":[x["greedy_exact"] for x in individual],
        "per_world_answer_only_loss":[x["answer_only_loss"] for x in individual],
    }
    return token,exact,secondary

def _arm_result(record:Mapping[str,Any])->ArmResult:
    return ArmResult(
        arm_id=str(record["arm_id"]),
        total_optimizer_updates=int(record["total_optimizer_updates"]),
        world_token_accuracies=tuple(record["world_token_accuracies"]),
        world_full_answer_exact=tuple(record["world_full_answer_exact"]),
        model_state_digest=str(record["model_state_digest"]),
        optimizer_state_digest=str(record["optimizer_state_digest"]),
        rng_state_digest=str(record["rng_state_digest"]),
        nonfinite_events=int(record["nonfinite_events"]),
        negative_dot_count=int(record["negative_dot_count"]),
        projection_event_count=int(record["projection_event_count"]),
    )

def _finalize_arm(
    *,
    arm_id:str,
    compiled:object,
    optimizer:torch.optim.Optimizer,
    worlds:Mapping[int,object],
    nonfinite:int,
    measurements:list[dict[str,Any]],
    invalid_reason:str|None,
)->dict[str,Any]:
    token,exact,secondary=_final_metrics(compiled,worlds)
    model=getattr(compiled,"model")
    result=ArmResult(
        arm_id=arm_id,
        total_optimizer_updates=64,
        world_token_accuracies=token,
        world_full_answer_exact=exact,
        model_state_digest=model_state_digest(model),
        optimizer_state_digest=optimizer_state_digest(optimizer),
        rng_state_digest=rng_state_digest(),
        nonfinite_events=nonfinite,
        negative_dot_count=sum(bool(x["negative_dot"]) for x in measurements),
        projection_event_count=sum(bool(x["projection_applied"]) for x in measurements),
    )
    validate_arm(result)
    return {
        **asdict(result),
        "measurements":measurements,
        "secondary":secondary,
        "invalid_reason":invalid_reason,
    }

def _run_control(checkpoint_path,receipt_path,selection_lock_path)->dict[str,Any]:
    state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    compiled,optimizer=state.compiled,state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds=_world_map();nonfinite=0;invalid_reason=None
    for world_index,_,effort in training_schedule():
        _,_,_,observed=_train_one_step_with_effort(compiled,worlds[world_index],optimizer=optimizer,effort=effort)
        nonfinite+=observed
        if nonfinite:
            invalid_reason="NONFINITE_CONTROL"
            break
    return _finalize_arm(
        arm_id=CONTROL_ARM,compiled=compiled,optimizer=optimizer,worlds=worlds,
        nonfinite=nonfinite,measurements=[],invalid_reason=invalid_reason,
    )

def _run_measured(checkpoint_path,receipt_path,selection_lock_path,*,arm_id:str,project:bool)->dict[str,Any]:
    state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    compiled,optimizer=state.compiled,state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds=_world_map();nonfinite=0;invalid_reason=None;measurements=[]
    for world_index,round_index,effort in training_schedule():
        target_index=2 if world_index==0 else 0
        measurement=_measured_step(
            compiled,worlds[world_index],worlds[target_index],
            optimizer=optimizer,effort=effort,project=project,
            round_index=round_index,source_world_index=world_index,
        )
        measurements.append(measurement)
        nonfinite+=int(measurement["nonfinite_events"])
        if nonfinite:
            invalid_reason="NONFINITE_MEASURED_ARM"
            break
    return _finalize_arm(
        arm_id=arm_id,compiled=compiled,optimizer=optimizer,worlds=worlds,
        nonfinite=nonfinite,measurements=measurements,invalid_reason=invalid_reason,
    )

def run_court(
    *,
    checkpoint_path:str|Path,
    receipt_path:str|Path,
    selection_lock_path:str|Path,
    parent_final_path:str|Path,
    execution_identity:Exp330ExecutionIdentity,
)->dict[str,Any]:
    validate_execution_identity(execution_identity)
    parent=json.loads(Path(parent_final_path).read_text(encoding="utf-8"))
    if not isinstance(parent,Mapping):
        raise ValueError("EXP-330 parent must be object")
    validate_parent_final(parent)
    authority=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    reconstruction=dict(authority.reconstruction)
    del authority

    arms=[
        _run_control(checkpoint_path,receipt_path,selection_lock_path),
        _run_measured(checkpoint_path,receipt_path,selection_lock_path,arm_id=SHAM_ARM,project=False),
        _run_measured(checkpoint_path,receipt_path,selection_lock_path,arm_id=PROJECT_ARM,project=True),
    ]
    results={record["arm_id"]:_arm_result(record) for record in arms}
    invalid=any(record.get("invalid_reason") is not None for record in arms)
    decision,passed=reduce_projection(results,invalid=invalid)
    sham_match=sham_equivalent(results[CONTROL_ARM],results[SHAM_ARM]) if not invalid else False
    projected=next(x for x in arms if x["arm_id"]==PROJECT_ARM)
    projection_events=[x for x in projected["measurements"] if x["projection_applied"]]

    payload={
        "schema":FINAL_SCHEMA,
        "execution_identity":asdict(execution_identity),
        "reconstruction":reconstruction,
        "decision":decision,
        "arms":arms,
        "arm_pass":{arm:bool(passed.get(arm,False)) for arm in ARM_IDS},
        "sham_state_match":sham_match,
        "projection_event_count":len(projection_events),
        "projection_events":projection_events,
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"]=_digest(payload)
    validate_final_evidence(payload)
    return payload

def validate_final_evidence(payload:Mapping[str,Any])->None:
    if payload.get("schema")!=FINAL_SCHEMA:
        raise ValueError("EXP-330 final schema mismatch")
    raw=payload.get("execution_identity")
    if not isinstance(raw,Mapping):
        raise ValueError("EXP-330 identity missing")
    validate_execution_identity(Exp330ExecutionIdentity(**raw))
    if payload.get("reconstruction")!=expected_reconstruction_payload():
        raise ValueError("EXP-330 reconstruction mismatch")
    arms=payload.get("arms")
    if not isinstance(arms,list) or {x.get("arm_id") for x in arms}!=set(ARM_IDS):
        raise ValueError("EXP-330 arm coverage mismatch")
    results={str(x["arm_id"]):_arm_result(x) for x in arms}
    invalid=any(x.get("invalid_reason") is not None for x in arms)
    decision,passed=reduce_projection(results,invalid=invalid)
    if payload.get("decision")!=decision:
        raise ValueError("EXP-330 reducer decision mismatch")
    if payload.get("arm_pass")!={arm:bool(passed.get(arm,False)) for arm in ARM_IDS}:
        raise ValueError("EXP-330 arm pass mismatch")
    expected_sham=sham_equivalent(results[CONTROL_ARM],results[SHAM_ARM]) if not invalid else False
    if payload.get("sham_state_match") is not expected_sham:
        raise ValueError("EXP-330 sham integrity mismatch")
    projected=next(x for x in arms if x["arm_id"]==PROJECT_ARM)
    expected_events=[x for x in projected.get("measurements",[]) if x.get("projection_applied") is True]
    if payload.get("projection_event_count")!=len(expected_events) or payload.get("projection_events")!=expected_events:
        raise ValueError("EXP-330 projection evidence mismatch")
    for key,expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-330 authorization drift: {key}")
    materialized=dict(payload);claimed=materialized.pop("evidence_digest",None)
    if not isinstance(claimed,str) or _digest(materialized)!=claimed:
        raise ValueError("EXP-330 evidence digest mismatch")
