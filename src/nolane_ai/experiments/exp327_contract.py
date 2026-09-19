from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
from typing import Any,Mapping,Sequence

SCHEMA_VERSION="EXP327-ITERATIVE-MINIMAL-FAILING-SUBSET-V1";EXPERIMENT_ID="EXP-327";FAMILY="iterative-grid-and-maze";WORLD_INDICES=(0,1,2,3)
GROUP_MEMBERS={"P01":(0,1),"P02":(0,2),"P03":(0,3),"P12":(1,2),"P13":(1,3),"P23":(2,3),"T012":(0,1,2),"T013":(0,1,3),"T023":(0,2,3),"T123":(1,2,3),"Q0123":(0,1,2,3)}
GROUP_IDS=tuple(GROUP_MEMBERS);PAIR_IDS=("P01","P02","P03","P12","P13","P23");TRIPLE_IDS=("T012","T013","T023","T123");QUARTET_ID="Q0123"
EXPOSURE_CHECKPOINTS=(8,16,32);EXPOSURES_PER_WORLD=32;EFFORT_CYCLE=(1,2,4,8);LEARNING_RATE=5e-5;TOKEN_FLOOR=.99;FULL_EXACT_FLOOR=.90
AUTHORIZATION_FLAGS={"exp302_implementation_authorized":False,"exp320_implementation_authorized":False,"scale_authorized":False,"authorized_30m":False,"authorized_100m":False}
_PREREG_JSON=r'''{"authority":{"parent_exp326_disposition":"QUARTET_LEVEL_BREAK_PRESENT","parent_exp326_execution_digest":"f62c89c9de75b3286c1093a33c17bb6295227247b9fe7892c49044f841d2df50","parent_exp326_final_artifact_id":10573847041,"parent_exp326_final_artifact_zip_digest":"92794024c87ff9fd5d1e7131c2a7404c6d843abc74b825bb96d68d6f952ebdc4","parent_exp326_final_evidence_digest":"e66d51977856219c6ad4124847da4431cb42877f9814cfe41e1b7644c28d2bf0","parent_exp326_final_json_sha256":"b5ccf5bb9e709c8dbc4da33e87a5bbdcaa1164f3408787f1ae423bae7a7003ed","parent_exp326_marker_sha":"fb66defc39bd9861b9bbaf0e6641cda6ae607b1d","parent_exp326_run_id":35409935457,"parent_exp326_source_sha":"937f0500703ec8c904d23dbeb6adcaf7d262380c","parent_q0123_pass":false,"reconstruction_artifact_id":10547681681,"reconstruction_artifact_name":"exp323r-reconstruction-candidate-35345351869-0","reconstruction_artifact_zip_digest":"c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621","reconstruction_checkpoint_sha256":"4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5","reconstruction_receipt_digest":"f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f","reconstruction_run_id":35345351869},"authorization":{"authorized_100m":false,"authorized_30m":false,"exp302_implementation_authorized":false,"exp320_implementation_authorized":false,"scale_authorized":false},"decision_order":["INVALID_SUBSET_COURT","PARENT_QUARTET_REPRODUCTION_MISMATCH","NONMONOTONIC_SUBSET_FIT","PAIR_MINIMAL_FAILURE_PRESENT","TRIPLE_MINIMAL_FAILURE_PRESENT","QUARTET_ONLY_FAILURE_CONFIRMED"],"dispositions":["INVALID_SUBSET_COURT","PARENT_QUARTET_REPRODUCTION_MISMATCH","NONMONOTONIC_SUBSET_FIT","PAIR_MINIMAL_FAILURE_PRESENT","TRIPLE_MINIMAL_FAILURE_PRESENT","QUARTET_ONLY_FAILURE_CONFIRMED"],"experiment_id":"EXP-327","interpretation":{"NONMONOTONIC_SUBSET_FIT":"a larger registered subset fits while at least one of its proper registered subsets fails; subset identity/order effects prevent a monotonic minimal-size claim","PAIR_MINIMAL_FAILURE_PRESENT":"at least one of all six two-world subsets fails; interference can occur with two simultaneously shared iterative worlds","PARENT_QUARTET_REPRODUCTION_MISMATCH":"the exact EXP-326 Q0123 failure did not reproduce under the same registered schedule; no smaller-subset causal claim is allowed","QUARTET_ONLY_FAILURE_CONFIRMED":"all six pairs and all four triples fit while Q0123 fails; the complete four-world interaction is required under this frozen court","TRIPLE_MINIMAL_FAILURE_PRESENT":"all six pairs fit but at least one of all four triples fails; the minimal registered failure size is three worlds"},"model":{"architecture_change":false,"arm":"A_FIXED","device":"cpu","objective_change":false,"resident_trainable_parameters":10000000,"scale_change":false,"starting_cumulative_step":2048,"tokenizer_change":false},"nonclaims":["EXP-327 does not establish a universal interference mechanism outside iterative worlds 0..3","EXP-327 does not establish an architectural capacity ceiling","EXP-327 does not establish that scaling would help","EXP-327 does not authorize EXP-320 or larger model sizes"],"population":{"family":"iterative-grid-and-maze","groups":{"pairs":[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],"quartet":[[0,1,2,3]],"triples":[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]},"independent_clone_per_group":true,"root":0,"stage":"A_SANITY","world_indices":[0,1,2,3]},"role":"post_exp326_iterative_complete_subset_lattice_localization","schema_version":"EXP327-ITERATIVE-MINIMAL-FAILING-SUBSET-V1","thresholds":{"nonfinite_events_max":0,"teacher_forced_full_answer_exact_floor":0.9,"teacher_forced_token_accuracy_floor":0.99},"training":{"effort_assignment":"world-local exposure index; exposure j uses effort_cycle[j mod 4] for every world","effort_cycle":[1,2,4,8],"exposure_checkpoints_per_world":[8,16,32],"exposures_per_world":32,"gradient_clip_norm_decimal":"1.0","inherited_optimizer_state":true,"inherited_rng_state":true,"learning_rate_decimal":"0.00005","optimizer":"AdamW","weight_decay_decimal":"0.01","within_exposure_order":"ascending world index within each registered subset"}}'''

@dataclass(frozen=True,slots=True)
class SubsetSnapshot:
 group_id:str;exposures_per_world:int;total_optimizer_updates:int;world_token_accuracies:tuple[float,...];world_full_answer_exact:tuple[float,...]

def canonical_json_bytes(p:Mapping[str,Any])->bytes:return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def preregistration_payload()->dict[str,Any]:return json.loads(_PREREG_JSON)
def preregistration_digest()->str:return hashlib.sha256(canonical_json_bytes(preregistration_payload())).hexdigest()
def effort_for_exposure(i:int)->int:
 if not isinstance(i,int) or not 0<=i<EXPOSURES_PER_WORLD:raise ValueError("exposure index")
 return EFFORT_CYCLE[i%4]
def training_schedule(gid:str):
 if gid not in GROUP_MEMBERS:raise ValueError("unknown group")
 return tuple((w,e,effort_for_exposure(e)) for e in range(EXPOSURES_PER_WORLD) for w in GROUP_MEMBERS[gid])
def validate_snapshot(s:SubsetSnapshot):
 if s.group_id not in GROUP_MEMBERS or s.exposures_per_world not in EXPOSURE_CHECKPOINTS:raise ValueError("snapshot identity")
 n=len(GROUP_MEMBERS[s.group_id])
 if s.total_optimizer_updates!=n*s.exposures_per_world or len(s.world_token_accuracies)!=n or len(s.world_full_answer_exact)!=n:raise ValueError("snapshot geometry")
 for xs in (s.world_token_accuracies,s.world_full_answer_exact):
  for v in xs:
   if not math.isfinite(v) or not 0<=v<=1:raise ValueError("metric")
def group_floor_pass(s:SubsetSnapshot)->bool:
 validate_snapshot(s);return all(t>=TOKEN_FLOOR and f>=FULL_EXACT_FLOOR for t,f in zip(s.world_token_accuracies,s.world_full_answer_exact))
def proper_subset_edges():
 return tuple((p,c) for c,cm in GROUP_MEMBERS.items() for p,pm in GROUP_MEMBERS.items() if len(pm)>len(cm) and set(cm)<set(pm))
def reduce_subset(records:Mapping[str,Sequence[SubsetSnapshot]],invalid=False):
 if invalid or set(records)!=set(GROUP_IDS):return "INVALID_SUBSET_COURT",{},()
 passed={}
 try:
  for gid in GROUP_IDS:
   rows=tuple(records[gid])
   if tuple(x.exposures_per_world for x in rows)!=EXPOSURE_CHECKPOINTS:return "INVALID_SUBSET_COURT",{},()
   for x in rows:
    validate_snapshot(x)
    if x.group_id!=gid:return "INVALID_SUBSET_COURT",{},()
   passed[gid]=any(group_floor_pass(x) for x in rows)
 except (TypeError,ValueError):return "INVALID_SUBSET_COURT",{},()
 if passed[QUARTET_ID]:return "PARENT_QUARTET_REPRODUCTION_MISMATCH",passed,()
 violations=tuple((p,c) for p,c in proper_subset_edges() if passed[p] and not passed[c])
 if violations:return "NONMONOTONIC_SUBSET_FIT",passed,violations
 if any(not passed[g] for g in PAIR_IDS):return "PAIR_MINIMAL_FAILURE_PRESENT",passed,()
 if any(not passed[g] for g in TRIPLE_IDS):return "TRIPLE_MINIMAL_FAILURE_PRESENT",passed,()
 return "QUARTET_ONLY_FAILURE_CONFIRMED",passed,()
