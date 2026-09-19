from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping,Sequence

EXPERIMENT_ID="EXP-329"
SCHEMA_VERSION="EXP329-P02-CROSS-UPDATE-INTERFERENCE-V1"
FAMILY="iterative-grid-and-maze"
WORLD_INDICES=(0,2)
EXPOSURES_PER_WORLD=32
EFFORT_CYCLE=(1,2,4,8)
PROBE_ROUNDS=(0,1,2,4,8,16,24,31)
CROSS_DAMAGE_EPS=1e-6
SELF_IMPROVEMENT_EPS=1e-6
TOKEN_FLOOR=.99
FULL_EXACT_FLOOR=.90
APPROVED_PREREGISTRATION_DIGEST="3f2e7073ac3c1c39efbcbb7ade65b4381147c90320248870125515091185fae4"
AUTHORIZATION_FLAGS={"exp302_implementation_authorized":False,"exp320_implementation_authorized":False,"scale_authorized":False,"authorized_30m":False,"authorized_100m":False}

@dataclass(frozen=True,slots=True)
class ProbeRecord:
    round_index:int
    effort:int
    world0_loss_pre:float
    world2_loss_pre:float
    gradient_dot:float
    gradient_cosine:float
    world0_gradient_norm:float
    world2_gradient_norm:float
    world0_update_self_delta:float
    world0_update_cross_delta_on_world2:float
    world2_update_self_delta:float
    world2_update_cross_delta_on_world0:float
    world0_direct_damage:bool
    world2_direct_damage:bool
    nonfinite_events:int

def effort_for_round(round_index:int)->int:
    if not isinstance(round_index,int) or not 0<=round_index<EXPOSURES_PER_WORLD:
        raise ValueError("round index")
    return EFFORT_CYCLE[round_index%len(EFFORT_CYCLE)]

def directed_damage(*,self_delta:float,cross_delta:float)->bool:
    if not math.isfinite(self_delta) or not math.isfinite(cross_delta):
        raise ValueError("nonfinite directed deltas")
    return self_delta<=-SELF_IMPROVEMENT_EPS and cross_delta>=CROSS_DAMAGE_EPS

def validate_probe(p:ProbeRecord)->None:
    if p.round_index not in PROBE_ROUNDS or p.effort!=effort_for_round(p.round_index):
        raise ValueError("probe identity")
    vals=(p.world0_loss_pre,p.world2_loss_pre,p.gradient_dot,p.gradient_cosine,p.world0_gradient_norm,p.world2_gradient_norm,p.world0_update_self_delta,p.world0_update_cross_delta_on_world2,p.world2_update_self_delta,p.world2_update_cross_delta_on_world0)
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("nonfinite probe metric")
    if not -1.000001<=p.gradient_cosine<=1.000001 or p.world0_gradient_norm<0 or p.world2_gradient_norm<0:
        raise ValueError("gradient metric range")
    if p.nonfinite_events!=0:
        raise ValueError("probe nonfinite event")
    if p.world0_direct_damage is not directed_damage(self_delta=p.world0_update_self_delta,cross_delta=p.world0_update_cross_delta_on_world2):
        raise ValueError("world0 damage flag mismatch")
    if p.world2_direct_damage is not directed_damage(self_delta=p.world2_update_self_delta,cross_delta=p.world2_update_cross_delta_on_world0):
        raise ValueError("world2 damage flag mismatch")

def simultaneous_floor_pass(token_acc:Sequence[float],full_exact:Sequence[float])->bool:
    if len(token_acc)!=2 or len(full_exact)!=2:
        raise ValueError("final reproduction geometry")
    for value in (*token_acc,*full_exact):
        if not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("final reproduction metric")
    return all(t>=TOKEN_FLOOR and f>=FULL_EXACT_FLOOR for t,f in zip(token_acc,full_exact))

def reduce_cross_update(
    probes:Sequence[ProbeRecord],
    *,
    final_token_accuracies:Sequence[float],
    final_full_answer_exact:Sequence[float],
    invalid:bool=False,
)->tuple[str,tuple[int,...],tuple[int,...]]:
    if invalid:
        return "INVALID_CROSS_UPDATE_COURT",(),()
    rows=tuple(probes)
    try:
        if tuple(x.round_index for x in rows)!=PROBE_ROUNDS:
            return "INVALID_CROSS_UPDATE_COURT",(),()
        for row in rows: validate_probe(row)
        parent_pass=simultaneous_floor_pass(final_token_accuracies,final_full_answer_exact)
    except (TypeError,ValueError):
        return "INVALID_CROSS_UPDATE_COURT",(),()
    if parent_pass:
        return "PARENT_ALT_REPRODUCTION_MISMATCH",(),()
    w0=tuple(x.round_index for x in rows if x.world0_direct_damage)
    w2=tuple(x.round_index for x in rows if x.world2_direct_damage)
    if w0 and w2:
        return "BIDIRECTIONAL_LOCAL_CROSS_DAMAGE",w0,w2
    if w0:
        return "WORLD0_ONLY_LOCAL_CROSS_DAMAGE",w0,w2
    if w2:
        return "WORLD2_ONLY_LOCAL_CROSS_DAMAGE",w0,w2
    return "NO_REGISTERED_LOCAL_CROSS_DAMAGE",w0,w2
