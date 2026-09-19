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
from .exp328_contract import (
    ARM_IDS,AUTHORIZATION_FLAGS,FAMILY,LEARNING_RATE,WORLD_INDICES,
    ArmResult,canonical_json_bytes,reduce_order_geometry,training_schedule,
)
from .exp328_identity import Exp328ExecutionIdentity,PARENT_FINAL_EVIDENCE_DIGEST,validate_execution_identity

FINAL_SCHEMA="EXP328-FINAL-EVIDENCE-V1"

def _digest(p:Mapping[str,Any])->str:
    return hashlib.sha256(canonical_json_bytes(p)).hexdigest()

def validate_parent_final(p:Mapping[str,Any])->None:
    if p.get("schema")!="EXP327-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-328 parent schema mismatch")
    if p.get("decision")!="PAIR_MINIMAL_FAILURE_PRESENT":
        raise ValueError("EXP-328 parent disposition mismatch")
    if p.get("evidence_digest")!=PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-328 parent evidence mismatch")
    q=dict(p);q.pop("evidence_digest",None)
    if _digest(q)!=PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-328 parent canonical mismatch")
    if p.get("failed_pairs")!=["P02"] or p.get("minimal_failing_groups")!=["P02"] or p.get("minimal_failing_size")!=2:
        raise ValueError("EXP-328 parent P02 localization mismatch")
    if p.get("monotonicity_violations")!=[]:
        raise ValueError("EXP-328 parent monotonicity mismatch")
    gp=p.get("group_pass")
    if not isinstance(gp,Mapping) or gp.get("P02") is not False:
        raise ValueError("EXP-328 parent P02 failure missing")
    for gid in ("P01","P03","P12","P13","P23"):
        if gp.get(gid) is not True:
            raise ValueError("EXP-328 parent non-P02 pair drift")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:
            raise ValueError(f"EXP-328 parent authorization drift: {k}")

def _worlds():
    m={int(getattr(w,"index")):w for w in family_worlds(FAMILY)}
    if tuple(sorted(m))!=tuple(range(8)):
        raise ValueError("EXP-328 world population drift")
    return m

def _run_arm(checkpoint_path,receipt_path,selection_lock_path,arm_id):
    state=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    compiled,optimizer=state.compiled,state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds=_worlds()
    nonfinite=0
    last_grad=last_update=0.0
    invalid=None
    for wi,_,effort in training_schedule(arm_id):
        _,last_grad,last_update,observed=_train_one_step_with_effort(
            compiled,worlds[wi],optimizer=optimizer,effort=effort
        )
        nonfinite+=observed
        if nonfinite:
            invalid="NONFINITE_EVENT"
            break
    individual=tuple(_evaluate_subset(compiled,(worlds[i],)) for i in WORLD_INDICES)
    result=ArmResult(
        arm_id=arm_id,
        total_optimizer_updates=64,
        world_token_accuracies=tuple(x["teacher_forced_answer_token_accuracy"] for x in individual),
        world_full_answer_exact=tuple(x["teacher_forced_full_answer_exact"] for x in individual),
        nonfinite_events=nonfinite,
    )
    passed=(
        invalid is None
        and all(t>=0.99 and f>=0.90 for t,f in zip(result.world_token_accuracies,result.world_full_answer_exact))
    )
    aggregate=_evaluate_subset(compiled,tuple(worlds[i] for i in WORLD_INDICES))
    model=getattr(compiled,"model")
    return {
        **asdict(result),
        "learning_rate":LEARNING_RATE,
        "passed":passed,
        "invalid_reason":invalid,
        "secondary":{
            "group_teacher_forced_answer_token_accuracy":aggregate["teacher_forced_answer_token_accuracy"],
            "group_teacher_forced_full_answer_exact":aggregate["teacher_forced_full_answer_exact"],
            "group_greedy_exact":aggregate["greedy_exact"],
            "group_answer_only_loss":aggregate["answer_only_loss"],
            "per_world_greedy_exact":[x["greedy_exact"] for x in individual],
            "per_world_answer_only_loss":[x["answer_only_loss"] for x in individual],
            "gradient_norm_preclip":last_grad,
            "parameter_update_norm_ratio":last_update,
        },
        "final_state":{
            "model_state_digest":model_state_digest(model),
            "optimizer_state_digest":optimizer_state_digest(optimizer),
            "rng_state_digest":rng_state_digest(),
        },
    }

def run_court(*,checkpoint_path,receipt_path,selection_lock_path,parent_final_path,execution_identity):
    validate_execution_identity(execution_identity)
    parent=json.loads(Path(parent_final_path).read_text())
    if not isinstance(parent,Mapping):
        raise ValueError("EXP-328 parent must be object")
    validate_parent_final(parent)
    authority=load_locked_reconstruction(checkpoint_path,receipt_path,selection_lock_path)
    reconstruction=dict(authority.reconstruction)
    del authority
    arms=[_run_arm(checkpoint_path,receipt_path,selection_lock_path,a) for a in ARM_IDS]
    records={
        a["arm_id"]:ArmResult(
            arm_id=a["arm_id"],
            total_optimizer_updates=a["total_optimizer_updates"],
            world_token_accuracies=tuple(a["world_token_accuracies"]),
            world_full_answer_exact=tuple(a["world_full_answer_exact"]),
            nonfinite_events=a["nonfinite_events"],
        )
        for a in arms
    }
    invalid=any(a["invalid_reason"] is not None for a in arms)
    decision,passed,rescued=reduce_order_geometry(records,invalid)
    payload={
        "schema":FINAL_SCHEMA,
        "execution_identity":asdict(execution_identity),
        "reconstruction":reconstruction,
        "decision":decision,
        "arms":arms,
        "arm_pass":{a:bool(passed.get(a,False)) for a in ARM_IDS},
        "rescued_arms":list(rescued),
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"]=_digest(payload)
    return payload

def validate_final_evidence(p:Mapping[str,Any])->None:
    if p.get("schema")!=FINAL_SCHEMA:
        raise ValueError("EXP-328 final schema mismatch")
    raw=p.get("execution_identity")
    if not isinstance(raw,Mapping):
        raise ValueError("EXP-328 identity missing")
    validate_execution_identity(Exp328ExecutionIdentity(**raw))
    if p.get("reconstruction")!=expected_reconstruction_payload():
        raise ValueError("EXP-328 reconstruction mismatch")
    arms=p.get("arms")
    if not isinstance(arms,list) or {a.get("arm_id") for a in arms}!=set(ARM_IDS):
        raise ValueError("EXP-328 arm coverage mismatch")
    for k,v in AUTHORIZATION_FLAGS.items():
        if p.get(k) is not v:
            raise ValueError(f"EXP-328 authorization drift: {k}")
    q=dict(p);claimed=q.pop("evidence_digest",None)
    if not isinstance(claimed,str) or _digest(q)!=claimed:
        raise ValueError("EXP-328 evidence digest mismatch")
