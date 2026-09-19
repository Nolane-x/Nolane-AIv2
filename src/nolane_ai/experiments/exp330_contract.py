from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping,Sequence

EXPERIMENT_ID="EXP-330"
SCHEMA_VERSION="EXP330-P02-CONFLICT-PROJECTION-RESCUE-V1"
FAMILY="iterative-grid-and-maze"
WORLD_INDICES=(0,2)
ARM_IDS=("CONTROL_ALT","SHAM_MEASURE_ALT","PROJECT_CONFLICT_ALT")
CONTROL_ARM="CONTROL_ALT"
SHAM_ARM="SHAM_MEASURE_ALT"
PROJECT_ARM="PROJECT_CONFLICT_ALT"
EXPOSURES_PER_WORLD=32
EFFORT_CYCLE=(1,2,4,8)
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
GRADIENT_CLIP_NORM=1.0
TARGET_NORM_SQUARED_FLOOR=1e-24
TOKEN_FLOOR=.99
FULL_EXACT_FLOOR=.90
APPROVED_PREREGISTRATION_DIGEST="c56438ff9df3cd2abe97c5dca49506e920aa610043cb22f20f8572bc9336c4a3"
AUTHORIZATION_FLAGS={"exp302_implementation_authorized":False,"exp320_implementation_authorized":False,"scale_authorized":False,"authorized_30m":False,"authorized_100m":False}

@dataclass(frozen=True,slots=True)
class ArmResult:
    arm_id:str
    total_optimizer_updates:int
    world_token_accuracies:tuple[float,float]
    world_full_answer_exact:tuple[float,float]
    model_state_digest:str
    optimizer_state_digest:str
    rng_state_digest:str
    nonfinite_events:int
    negative_dot_count:int
    projection_event_count:int

def effort_for_round(round_index:int)->int:
    if not isinstance(round_index,int) or not 0<=round_index<EXPOSURES_PER_WORLD:
        raise ValueError("round index")
    return EFFORT_CYCLE[round_index%len(EFFORT_CYCLE)]

def training_schedule()->tuple[tuple[int,int,int],...]:
    return tuple(
        (world,round_index,effort_for_round(round_index))
        for round_index in range(EXPOSURES_PER_WORLD)
        for world in WORLD_INDICES
    )

def validate_arm(result:ArmResult)->None:
    if result.arm_id not in ARM_IDS or result.total_optimizer_updates!=64:
        raise ValueError("arm identity/geometry")
    if len(result.world_token_accuracies)!=2 or len(result.world_full_answer_exact)!=2:
        raise ValueError("world metric geometry")
    for value in (*result.world_token_accuracies,*result.world_full_answer_exact):
        if not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("primary metric")
    for digest in (result.model_state_digest,result.optimizer_state_digest,result.rng_state_digest):
        if not isinstance(digest,str) or len(digest)!=64:
            raise ValueError("state digest")
    if not isinstance(result.nonfinite_events,int) or result.nonfinite_events<0:
        raise ValueError("nonfinite count")
    if not 0<=result.negative_dot_count<=64 or not 0<=result.projection_event_count<=64:
        raise ValueError("measurement count")
    if result.projection_event_count>result.negative_dot_count:
        raise ValueError("projection count exceeds negative-dot count")
    if result.arm_id in (CONTROL_ARM,SHAM_ARM) and result.projection_event_count!=0:
        raise ValueError("non-projected arm projected")
    if result.arm_id==CONTROL_ARM and result.negative_dot_count!=0:
        raise ValueError("control must not claim measured dots")

def world_floor_flags(result:ArmResult)->tuple[bool,bool]:
    validate_arm(result)
    return tuple(
        t>=TOKEN_FLOOR and f>=FULL_EXACT_FLOOR
        for t,f in zip(result.world_token_accuracies,result.world_full_answer_exact)
    )

def arm_pass(result:ArmResult)->bool:
    return result.nonfinite_events==0 and all(world_floor_flags(result))

def sham_equivalent(control:ArmResult,sham:ArmResult)->bool:
    validate_arm(control);validate_arm(sham)
    return (
        sham.nonfinite_events==0
        and control.world_token_accuracies==sham.world_token_accuracies
        and control.world_full_answer_exact==sham.world_full_answer_exact
        and control.model_state_digest==sham.model_state_digest
        and control.optimizer_state_digest==sham.optimizer_state_digest
        and control.rng_state_digest==sham.rng_state_digest
    )

def reduce_projection(results:Mapping[str,ArmResult],*,invalid:bool=False)->tuple[str,dict[str,bool]]:
    if invalid or set(results)!=set(ARM_IDS):
        return "INVALID_PROJECTION_COURT",{}
    try:
        for arm in ARM_IDS: validate_arm(results[arm])
    except (TypeError,ValueError):
        return "INVALID_PROJECTION_COURT",{}
    if any(results[arm].nonfinite_events!=0 for arm in ARM_IDS):
        return "INVALID_PROJECTION_COURT",{}
    passed={arm:arm_pass(results[arm]) for arm in ARM_IDS}
    if world_floor_flags(results[CONTROL_ARM])!=(False,True):
        return "PARENT_ALT_REPRODUCTION_MISMATCH",passed
    if not sham_equivalent(results[CONTROL_ARM],results[SHAM_ARM]):
        return "SHAM_TRAJECTORY_MISMATCH",passed
    if results[PROJECT_ARM].projection_event_count==0:
        return "PROJECTION_NOT_TRIGGERED",passed
    if passed[PROJECT_ARM]:
        return "CONFLICT_PROJECTION_RESCUE",passed
    return "CONFLICT_PROJECTION_NO_RESCUE",passed
