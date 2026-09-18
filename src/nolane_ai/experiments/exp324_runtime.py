from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .exp301_scientific import forward_scientific_arm
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss
from .exp319_training import _diagnostic_generate, model_state_digest, optimizer_state_digest, rng_state_digest
from .exp319_worlds import materialize_stage_a, verify_world_answer
from .exp322_runtime import _encoded_tensors, _train_one_step
from .exp323r_repair import load_locked_reconstruction
from .exp324_contract import (
    AUTHORIZATION_FLAGS,
    FAMILIES,
    LEARNING_RATE,
    LOCAL_CHECKPOINTS,
    STARTING_STEP,
    UPDATES_PER_FAMILY,
    FamilySnapshot,
    canonical_json_bytes,
    reduce_family_isolation,
    validate_snapshot,
)
from .exp324_identity import (
    PARENT_EXECUTION_DIGEST,
    PARENT_FINAL_EVIDENCE_DIGEST,
    Exp324ExecutionIdentity,
    validate_execution_identity,
)


ARM_SCHEMA="EXP324-FAMILY-ARM-EVIDENCE-V1"
FINAL_SCHEMA="EXP324-FINAL-EVIDENCE-V1"


def _digest(payload:Mapping[str,Any])->str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_parent_final(payload:Mapping[str,Any])->None:
    if payload.get("schema")!="EXP323R-FINAL-EVIDENCE-V1": raise ValueError("EXP-324 parent final schema mismatch")
    if payload.get("decision")!="NO_REGISTERED_RESCUE": raise ValueError("EXP-324 parent disposition mismatch")
    if payload.get("evidence_digest")!=PARENT_FINAL_EVIDENCE_DIGEST: raise ValueError("EXP-324 parent evidence digest mismatch")
    materialized=dict(payload); materialized.pop("evidence_digest",None)
    if _digest(materialized)!=PARENT_FINAL_EVIDENCE_DIGEST: raise ValueError("EXP-324 parent canonical evidence mismatch")
    identity=payload.get("execution_identity")
    if not isinstance(identity,Mapping) or identity.get("exp323r_execution_digest")!=PARENT_EXECUTION_DIGEST:
        raise ValueError("EXP-324 parent execution identity mismatch")
    for key,expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected: raise ValueError(f"EXP-324 parent authorization drift: {key}")


def family_worlds(family:str)->tuple[object,...]:
    if family not in FAMILIES: raise ValueError("unknown EXP-324 family")
    worlds=tuple(w for w in materialize_stage_a() if str(getattr(w,"family"))==family)
    if len(worlds)!=8: raise ValueError("EXP-324 family must contain exactly eight worlds")
    ids=tuple(str(getattr(w,"content_id")) for w in worlds)
    if len(set(ids))!=8: raise ValueError("EXP-324 family worlds must be unique")
    return worlds


def validate_optimizer_invariants(optimizer:torch.optim.Optimizer)->None:
    if not optimizer.param_groups: raise ValueError("EXP-324 optimizer has no parameter groups")
    for group in optimizer.param_groups:
        if float(group.get("lr",math.nan))!=LEARNING_RATE: raise ValueError("EXP-324 inherited LR is not 5e-5")
        if float(group.get("weight_decay",math.nan))!=0.01: raise ValueError("EXP-324 weight decay drift")


def _evaluate_subset(compiled:object,worlds:Sequence[object])->dict[str,float]:
    if not worlds: raise ValueError("EXP-324 evaluation subset is empty")
    model=getattr(compiled,"model"); device=next(model.parameters()).device
    tokenizer=Exp301ByteTokenizer()
    correct_tokens=target_tokens=full_exact=greedy_exact=0
    weighted_loss=0.0
    model.eval()
    with torch.inference_mode():
        for world in worlds:
            input_ids,targets,answer_start=_encoded_tensors(world,device=device)
            logits=forward_scientific_arm(compiled,input_ids,effort=4)
            first=answer_start-1
            answer_logits=logits[0,first:]; answer_targets=targets[0,first:]
            top1=torch.argmax(answer_logits,dim=-1)
            correct=top1.eq(answer_targets)
            count=int(correct.numel()); hits=int(correct.sum().item())
            correct_tokens+=hits; target_tokens+=count; full_exact+=int(bool(correct.all().item()))
            loss=compute_answer_only_loss(logits,targets=targets,answer_start=answer_start)
            weighted_loss+=float(loss.item())*count
            candidate,_,_=_diagnostic_generate(compiled,prompt=str(getattr(world,"model_input")))
            greedy_exact+=int(verify_world_answer(world,candidate))
    if target_tokens<=0: raise RuntimeError("EXP-324 evaluation found no answer targets")
    return {
        "teacher_forced_answer_token_accuracy":correct_tokens/target_tokens,
        "teacher_forced_full_answer_exact":full_exact/len(worlds),
        "greedy_exact":greedy_exact/len(worlds),
        "answer_only_loss":weighted_loss/target_tokens,
    }


def _build_arm_evidence(
    *,
    family:str,
    identity:Exp324ExecutionIdentity,
    reconstruction:Mapping[str,Any],
    snapshots:Sequence[FamilySnapshot],
    all32_summaries:Mapping[str,Any],
    final_model_digest:str,
    final_optimizer_digest:str,
    final_rng_digest:str,
    invalid_reason:str|None,
)->dict[str,Any]:
    payload={
        "schema":ARM_SCHEMA,
        "family":family,
        "learning_rate":LEARNING_RATE,
        "starting_step":STARTING_STEP,
        "isolated_optimizer_updates":UPDATES_PER_FAMILY,
        "execution_identity":asdict(identity),
        "reconstruction":dict(reconstruction),
        "snapshots":[asdict(x) for x in snapshots],
        "all32_summaries":dict(all32_summaries),
        "final_state":{
            "model_state_digest":final_model_digest,
            "optimizer_state_digest":final_optimizer_digest,
            "rng_state_digest":final_rng_digest,
        },
        "invalid_reason":invalid_reason,
        **AUTHORIZATION_FLAGS,
    }
    validate_arm_evidence(payload,with_digest=False)
    payload["arm_evidence_digest"]=_digest(payload)
    return payload


def validate_arm_evidence(payload:Mapping[str,Any],*,with_digest:bool=True)->None:
    if payload.get("schema")!=ARM_SCHEMA or payload.get("family") not in FAMILIES: raise ValueError("EXP-324 arm schema/family mismatch")
    if payload.get("learning_rate")!=LEARNING_RATE or payload.get("starting_step")!=STARTING_STEP or payload.get("isolated_optimizer_updates")!=UPDATES_PER_FAMILY: raise ValueError("EXP-324 arm geometry mismatch")
    raw=payload.get("execution_identity")
    if not isinstance(raw,Mapping): raise ValueError("EXP-324 execution identity missing")
    validate_execution_identity(Exp324ExecutionIdentity(**raw))
    for key,expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected: raise ValueError(f"EXP-324 authorization drift: {key}")
    rows=tuple(FamilySnapshot(**x) for x in payload.get("snapshots",()))
    if payload.get("invalid_reason") is None:
        if tuple(x.local_update for x in rows)!=LOCAL_CHECKPOINTS: raise ValueError("EXP-324 checkpoint geometry mismatch")
    for row in rows:
        validate_snapshot(row)
        if row.family!=payload["family"]: raise ValueError("EXP-324 snapshot family mismatch")
    summaries=payload.get("all32_summaries")
    if not isinstance(summaries,Mapping): raise ValueError("EXP-324 all32 summaries missing")
    if payload.get("invalid_reason") is None and set(summaries)!=set(str(x) for x in LOCAL_CHECKPOINTS): raise ValueError("EXP-324 all32 checkpoint geometry mismatch")
    if with_digest:
        materialized=dict(payload); claimed=materialized.pop("arm_evidence_digest",None)
        if not isinstance(claimed,str) or _digest(materialized)!=claimed: raise ValueError("EXP-324 arm evidence digest mismatch")


def run_family_arm(
    *,
    checkpoint_path:str|Path,
    receipt_path:str|Path,
    selection_lock_path:str|Path,
    parent_final_path:str|Path,
    family:str,
    execution_identity:Exp324ExecutionIdentity,
)->dict[str,Any]:
    validate_execution_identity(execution_identity)
    parent=json.loads(Path(parent_final_path).read_text(encoding="utf-8"))
    if not isinstance(parent,Mapping): raise ValueError("EXP-324 parent final must be object")
    validate_parent_final(parent)
    state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    compiled,optimizer=state.compiled,state.optimizer
    validate_optimizer_invariants(optimizer)
    isolated=family_worlds(family); all_worlds=tuple(materialize_stage_a())
    snapshots=[]; all32={}
    nonfinite=0; last_grad=last_update=0.0; invalid_reason=None
    for local_index in range(UPDATES_PER_FAMILY):
        global_step=STARTING_STEP+local_index
        world=isolated[local_index%len(isolated)]
        _,last_grad,last_update,observed=_train_one_step(compiled,world,optimizer=optimizer,global_step=global_step)
        nonfinite+=observed
        completed=local_index+1
        if nonfinite:
            invalid_reason="NONFINITE_EVENT"; break
        if completed in LOCAL_CHECKPOINTS:
            primary=_evaluate_subset(compiled,isolated)
            snapshots.append(FamilySnapshot(
                family=family,
                local_update=completed,
                cumulative_step=STARTING_STEP+completed,
                teacher_forced_answer_token_accuracy=primary["teacher_forced_answer_token_accuracy"],
                teacher_forced_full_answer_exact=primary["teacher_forced_full_answer_exact"],
                greedy_exact=primary["greedy_exact"],
                answer_only_loss=primary["answer_only_loss"],
                gradient_norm_preclip=last_grad,
                parameter_update_norm_ratio=last_update,
                nonfinite_events=nonfinite,
            ))
            all32[str(completed)]=_evaluate_subset(compiled,all_worlds)
    model=getattr(compiled,"model")
    return _build_arm_evidence(
        family=family,identity=execution_identity,reconstruction=state.reconstruction,
        snapshots=tuple(snapshots),all32_summaries=all32,
        final_model_digest=model_state_digest(model),
        final_optimizer_digest=optimizer_state_digest(optimizer),
        final_rng_digest=rng_state_digest(),
        invalid_reason=invalid_reason,
    )


def build_final_evidence(arms:Sequence[Mapping[str,Any]])->dict[str,Any]:
    by_family={}
    for arm in arms:
        validate_arm_evidence(arm)
        family=str(arm["family"])
        if family in by_family: raise ValueError("duplicate EXP-324 family arm")
        by_family[family]=arm
    if set(by_family)!=set(FAMILIES): raise ValueError("EXP-324 requires all four family arms")
    identities={canonical_json_bytes(a["execution_identity"]) for a in by_family.values()}
    reconstructions={canonical_json_bytes(a["reconstruction"]) for a in by_family.values()}
    if len(identities)!=1 or len(reconstructions)!=1: raise ValueError("EXP-324 arm authority mismatch")
    invalid=any(a.get("invalid_reason") is not None for a in by_family.values())
    records={f:tuple(FamilySnapshot(**x) for x in by_family[f]["snapshots"]) for f in FAMILIES}
    decision="INVALID_FAMILY_ISOLATION" if invalid else reduce_family_isolation(records)
    passed=[
        f for f in FAMILIES
        if any(
            x.teacher_forced_answer_token_accuracy>=0.99 and x.teacher_forced_full_answer_exact>=0.90
            for x in records[f]
        )
    ]
    payload={
        "schema":FINAL_SCHEMA,
        "execution_identity":by_family[FAMILIES[0]]["execution_identity"],
        "reconstruction":by_family[FAMILIES[0]]["reconstruction"],
        "decision":decision,
        "passed_families":passed,
        "arm_evidence_digests":{f:by_family[f]["arm_evidence_digest"] for f in FAMILIES},
        "families":{
            f:{
                "snapshots":by_family[f]["snapshots"],
                "all32_summaries":by_family[f]["all32_summaries"],
                "final_state":by_family[f]["final_state"],
                "invalid_reason":by_family[f]["invalid_reason"],
            } for f in FAMILIES
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"]=_digest(payload)
    return payload
