from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
from typing import Any,Mapping

SCHEMA_VERSION="EXP328-P02-ORDER-GEOMETRY-V1"
EXPERIMENT_ID="EXP-328"
FAMILY="iterative-grid-and-maze"
WORLD_INDICES=(0,2)
ARM_IDS=("ALT_0_2","ALT_2_0","BLOCK_0_2","BLOCK_2_0")
REPRODUCTION_ARM="ALT_0_2"
EXPOSURES_PER_WORLD=32
EFFORT_CYCLE=(1,2,4,8)
LEARNING_RATE=5e-5
TOKEN_FLOOR=.99
FULL_EXACT_FLOOR=.90
AUTHORIZATION_FLAGS={"exp302_implementation_authorized":False,"exp320_implementation_authorized":False,"scale_authorized":False,"authorized_30m":False,"authorized_100m":False}
APPROVED_PREREGISTRATION_DIGEST="e87c20965651193f0e927b7c6969fd0d12ec3fc83b9f56485bfbe1bc4e3f03cd"
PARENT_EVIDENCE_DIGEST="bc9999f3f6e366de8ec41b26ec265514546cf68cb82661295a31b202e3e907e2"

@dataclass(frozen=True,slots=True)
class ArmResult:
    arm_id:str
    total_optimizer_updates:int
    world_token_accuracies:tuple[float,float]
    world_full_answer_exact:tuple[float,float]
    nonfinite_events:int=0

def canonical_json_bytes(p:Mapping[str,Any])->bytes:
    return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def canonical_digest(p:Mapping[str,Any])->str:
    return hashlib.sha256(canonical_json_bytes(p)).hexdigest()

def effort_for_exposure(i:int)->int:
    if not isinstance(i,int) or not 0<=i<EXPOSURES_PER_WORLD:
        raise ValueError("exposure index")
    return EFFORT_CYCLE[i%len(EFFORT_CYCLE)]

def training_schedule(arm_id:str)->tuple[tuple[int,int,int],...]:
    if arm_id=="ALT_0_2":
        order=(0,2)
        return tuple((w,e,effort_for_exposure(e)) for e in range(EXPOSURES_PER_WORLD) for w in order)
    if arm_id=="ALT_2_0":
        order=(2,0)
        return tuple((w,e,effort_for_exposure(e)) for e in range(EXPOSURES_PER_WORLD) for w in order)
    if arm_id=="BLOCK_0_2":
        order=(0,2)
        return tuple((w,e,effort_for_exposure(e)) for w in order for e in range(EXPOSURES_PER_WORLD))
    if arm_id=="BLOCK_2_0":
        order=(2,0)
        return tuple((w,e,effort_for_exposure(e)) for w in order for e in range(EXPOSURES_PER_WORLD))
    raise ValueError("unknown EXP-328 arm")

def validate_arm_result(r:ArmResult)->None:
    if r.arm_id not in ARM_IDS or r.total_optimizer_updates!=64:
        raise ValueError("arm identity/geometry")
    if not isinstance(r.nonfinite_events,int) or r.nonfinite_events<0:
        raise ValueError("nonfinite counter")
    for xs in (r.world_token_accuracies,r.world_full_answer_exact):
        if len(xs)!=2:
            raise ValueError("world metric geometry")
        for v in xs:
            if not math.isfinite(v) or not 0<=v<=1:
                raise ValueError("metric")

def arm_pass(r:ArmResult)->bool:
    validate_arm_result(r)
    return r.nonfinite_events==0 and all(
        t>=TOKEN_FLOOR and f>=FULL_EXACT_FLOOR
        for t,f in zip(r.world_token_accuracies,r.world_full_answer_exact)
    )

def reduce_order_geometry(results:Mapping[str,ArmResult],invalid:bool=False)->tuple[str,dict[str,bool],tuple[str,...]]:
    if invalid or set(results)!=set(ARM_IDS):
        return "INVALID_ORDER_GEOMETRY_COURT",{},()
    try:
        passed={a:arm_pass(results[a]) for a in ARM_IDS}
    except (TypeError,ValueError):
        return "INVALID_ORDER_GEOMETRY_COURT",{},()
    if passed[REPRODUCTION_ARM]:
        return "PARENT_P02_REPRODUCTION_MISMATCH",passed,()
    rescued=tuple(a for a in ARM_IDS if a!=REPRODUCTION_ARM and passed[a])
    if rescued:
        return "ORDER_GEOMETRY_SENSITIVE_INTERFERENCE",passed,rescued
    return "ORDER_GEOMETRY_INVARIANT_PAIR_FAILURE",passed,()
