from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping

EXPERIMENT_ID="EXP-331"
SCHEMA_VERSION="EXP331-PAIR-LATTICE-PROJECTION-SAFETY-V1"
FAMILY="iterative-grid-and-maze"
PAIR_MEMBERS={"P01":(0,1),"P02":(0,2),"P03":(0,3),"P12":(1,2),"P13":(1,3),"P23":(2,3)}
PAIR_IDS=tuple(PAIR_MEMBERS)
MODES=("CONTROL","SHAM","PROJECT")
EXPOSURES_PER_WORLD=32
EFFORT_CYCLE=(1,2,4,8)
GRADIENT_CLIP_NORM=1.0
TARGET_NORM_SQUARED_FLOOR=1e-24
TOKEN_FLOOR=.99
FULL_EXACT_FLOOR=.90
APPROVED_PREREGISTRATION_DIGEST="09ef36245d8048bd5bf3810b518c3dbbad5f3e3d84c4bff4786c409515fce167"
PARENT_PAIR_PASS={"P01":True,"P02":False,"P03":True,"P12":True,"P13":True,"P23":True}
AUTHORIZATION_FLAGS={"exp302_implementation_authorized":False,"exp320_implementation_authorized":False,"scale_authorized":False,"authorized_30m":False,"authorized_100m":False}

@dataclass(frozen=True,slots=True)
class PairArmResult:
    pair_id:str
    mode:str
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
    if not isinstance(round_index,int) or not 0<=round_index<EXPOSURES_PER_WORLD:raise ValueError("round index")
    return EFFORT_CYCLE[round_index%len(EFFORT_CYCLE)]

def training_schedule(pair_id:str)->tuple[tuple[int,int,int],...]:
    if pair_id not in PAIR_MEMBERS:raise ValueError("pair")
    return tuple((w,r,effort_for_round(r)) for r in range(EXPOSURES_PER_WORLD) for w in PAIR_MEMBERS[pair_id])

def validate_arm(x:PairArmResult)->None:
    if x.pair_id not in PAIR_IDS or x.mode not in MODES or x.total_optimizer_updates!=64:raise ValueError("arm identity")
    if len(x.world_token_accuracies)!=2 or len(x.world_full_answer_exact)!=2:raise ValueError("metric geometry")
    for v in (*x.world_token_accuracies,*x.world_full_answer_exact):
        if not math.isfinite(v) or not 0<=v<=1:raise ValueError("metric")
    for d in (x.model_state_digest,x.optimizer_state_digest,x.rng_state_digest):
        if not isinstance(d,str) or len(d)!=64:raise ValueError("digest")
    if x.nonfinite_events<0 or not 0<=x.negative_dot_count<=64 or not 0<=x.projection_event_count<=64:raise ValueError("count")
    if x.projection_event_count>x.negative_dot_count:raise ValueError("projection count")
    if x.mode!="PROJECT" and x.projection_event_count!=0:raise ValueError("unexpected projection")
    if x.mode=="CONTROL" and x.negative_dot_count!=0:raise ValueError("control measured dots")

def arm_pass(x:PairArmResult)->bool:
    validate_arm(x)
    return x.nonfinite_events==0 and all(t>=TOKEN_FLOOR and e>=FULL_EXACT_FLOOR for t,e in zip(x.world_token_accuracies,x.world_full_answer_exact))

def sham_equivalent(control:PairArmResult,sham:PairArmResult)->bool:
    validate_arm(control);validate_arm(sham)
    return control.pair_id==sham.pair_id and control.mode=="CONTROL" and sham.mode=="SHAM" and sham.nonfinite_events==0 and control.world_token_accuracies==sham.world_token_accuracies and control.world_full_answer_exact==sham.world_full_answer_exact and control.model_state_digest==sham.model_state_digest and control.optimizer_state_digest==sham.optimizer_state_digest and control.rng_state_digest==sham.rng_state_digest

def expected_keys()->set[str]:
    return {f"{p}:{m}" for p in PAIR_IDS for m in MODES}

def reduce_pair_lattice(results:Mapping[str,PairArmResult],*,parent_reproduced:bool,invalid:bool=False)->tuple[str,dict[str,bool],tuple[str,...]]:
    if invalid or set(results)!=expected_keys():return "INVALID_PAIR_LATTICE_PROJECTION_COURT",{},()
    try:
        for x in results.values():validate_arm(x)
    except (TypeError,ValueError):
        return "INVALID_PAIR_LATTICE_PROJECTION_COURT",{},()
    if any(x.nonfinite_events for x in results.values()):return "INVALID_PAIR_LATTICE_PROJECTION_COURT",{},()
    passed={k:arm_pass(v) for k,v in results.items()}
    if not parent_reproduced:return "PARENT_PAIR_LATTICE_REPRODUCTION_MISMATCH",passed,()
    if any(not sham_equivalent(results[f"{p}:CONTROL"],results[f"{p}:SHAM"]) for p in PAIR_IDS):
        return "SHAM_PAIR_LATTICE_MISMATCH",passed,()
    if results["P02:PROJECT"].projection_event_count==0:return "P02_PROJECTION_NOT_TRIGGERED",passed,()
    regressions=tuple(p for p in PAIR_IDS if PARENT_PAIR_PASS[p] and not passed[f"{p}:PROJECT"])
    if passed["P02:PROJECT"] and not regressions:return "PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION",passed,regressions
    if passed["P02:PROJECT"]:return "P02_RESCUE_WITH_PAIR_REGRESSION",passed,regressions
    return "P02_PROJECTION_NO_RESCUE",passed,regressions
