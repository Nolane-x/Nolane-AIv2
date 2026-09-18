from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "EXP325-SINGLE-WORLD-MEMORIZATION-V1"
EXPERIMENT_ID = "EXP-325"
FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
WORLD_INDICES = tuple(range(8))
WORLD_KEYS = tuple((family, index) for family in FAMILIES for index in WORLD_INDICES)
LOCAL_CHECKPOINTS = (8, 16, 32)
STARTING_STEP = 2048
UPDATES_PER_WORLD = 32
LEARNING_RATE = 5e-5
TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90
DISPOSITIONS = (
    "INVALID_WORLD_ISOLATION",
    "ALL_WORLDS_SINGLE_FIT",
    "SOME_WORLDS_SINGLE_FIT",
    "NO_WORLDS_SINGLE_FIT",
)
AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}
_PREREG_JSON = r'''{"authority":{"parent_exp324_disposition":"NO_FAMILIES_ISOLATED_FIT","parent_exp324_execution_digest":"fbf1e416f3d195a44ca901ed8e2c7377c6850e4e32ec0116bb59d7657513f2f0","parent_exp324_final_artifact_id":10554724648,"parent_exp324_final_artifact_zip_digest":"70fae1e9d8b54df17f2a46740630cbb1861662a5ec3d2df756193e3cd6991965","parent_exp324_final_evidence_digest":"10a493d2f4de15131706b933af4170f39f55e802f069e740f9f98846713ac2ad","parent_exp324_marker_sha":"3bc060f1d3656dbae1de2265a32b145d9cff0dbc","parent_exp324_run_id":35362450626,"reconstruction_artifact_id":10547681681,"reconstruction_artifact_name":"exp323r-reconstruction-candidate-35345351869-0","reconstruction_artifact_zip_digest":"c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621","reconstruction_checkpoint_sha256":"4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5","reconstruction_receipt_digest":"f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f","reconstruction_run_id":35345351869},"authorization":{"authorized_100m":false,"authorized_30m":false,"exp302_implementation_authorized":false,"exp320_implementation_authorized":false,"scale_authorized":false},"decision_order":["INVALID_WORLD_ISOLATION","ALL_WORLDS_SINGLE_FIT","SOME_WORLDS_SINGLE_FIT","NO_WORLDS_SINGLE_FIT"],"dispositions":["INVALID_WORLD_ISOLATION","ALL_WORLDS_SINGLE_FIT","SOME_WORLDS_SINGLE_FIT","NO_WORLDS_SINGLE_FIT"],"experiment_id":"EXP-325","interpretation":{"ALL_WORLDS_SINGLE_FIT":"supports within-family multi-world optimization interference as a contributor under matched exposure; does not identify a unique mechanism","NO_WORLDS_SINGLE_FIT":"shows no registered single world reaches the floor under matched 32-update isolation; cross-world interference is not required for the observed failure","SOME_WORLDS_SINGLE_FIT":"localizes failure to a subset of worlds and leaves both per-world difficulty and residual within-family interference viable"},"model":{"architecture_change":false,"arm":"A_FIXED","device":"cpu","objective_change":false,"resident_trainable_parameters":10000000,"scale_change":false,"starting_cumulative_step":2048,"tokenizer_change":false},"nonclaims":["EXP-325 does not establish a 10M architectural capacity ceiling","EXP-325 does not establish that scaling would help","EXP-325 does not authorize EXP-320 or larger model sizes","single-world success does not uniquely prove a specific interference mechanism","single-world failure at 32 matched updates does not establish failure at larger per-world budgets"],"population":{"families":["iterative-grid-and-maze","algorithmic-sequence-transform","generator-heldout-abstract-transformation","language-sequence-control"],"generator_version":"exp319-worlds-v1","identity_rule":"exact tuple (family, stage=A_SANITY, root=0, index) materialized by exp319-worlds-v1","indices_per_family":[0,1,2,3,4,5,6,7],"root":0,"stage":"A_SANITY","total_worlds":32},"primary_measurements":["single_world_teacher_forced_answer_token_accuracy","single_world_teacher_forced_full_answer_exact"],"role":"post_exp324_single_world_memorization_diagnostic","schema_version":"EXP325-SINGLE-WORLD-MEMORIZATION-V1","secondary_measurements":["single_world_greedy_exact","single_world_answer_only_loss","gradient_norm_preclip","parameter_update_norm_ratio","nonfinite_events"],"thresholds":{"nonfinite_events_max":0,"teacher_forced_full_answer_exact_floor":0.9,"teacher_forced_token_accuracy_floor":0.99},"training":{"gradient_clip_norm_decimal":"1.0","independent_clone_per_world":true,"inherited_optimizer_state":true,"inherited_rng_state":true,"learning_rate_decimal":"0.00005","local_checkpoints":[8,16,32],"optimizer":"AdamW","optimizer_updates_per_world":32,"rationale":"32 updates per world exactly matches each world's exposure count in EXP-324 family isolation and the 1024-step joint control; only within-family/world interference is removed","training_effort_cycle":[1,2,4,8],"weight_decay_decimal":"0.01"}}'''


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    family: str
    world_index: int
    local_update: int
    teacher_forced_answer_token_accuracy: float
    teacher_forced_full_answer_exact: float
    greedy_exact: float
    answer_only_loss: float
    gradient_norm_preclip: float
    parameter_update_norm_ratio: float
    nonfinite_events: int


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def preregistration_payload() -> dict[str, Any]:
    payload = json.loads(_PREREG_JSON)
    if not isinstance(payload, dict):
        raise AssertionError("EXP-325 preregistration root must be object")
    return payload


def preregistration_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(preregistration_payload())).hexdigest()


def validate_snapshot(snapshot: WorldSnapshot) -> None:
    if (snapshot.family, snapshot.world_index) not in WORLD_KEYS:
        raise ValueError("unknown EXP-325 world identity")
    if snapshot.local_update not in LOCAL_CHECKPOINTS:
        raise ValueError("unknown EXP-325 checkpoint")
    for field in (
        "teacher_forced_answer_token_accuracy",
        "teacher_forced_full_answer_exact",
        "greedy_exact",
    ):
        value=getattr(snapshot,field)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite in [0,1]")
    for field in ("answer_only_loss","gradient_norm_preclip","parameter_update_norm_ratio"):
        value=getattr(snapshot,field)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    if snapshot.nonfinite_events != 0:
        raise ValueError("EXP-325 nonfinite event")


def world_floor_pass(snapshot: WorldSnapshot) -> bool:
    return (
        snapshot.teacher_forced_answer_token_accuracy >= TOKEN_FLOOR
        and snapshot.teacher_forced_full_answer_exact >= FULL_EXACT_FLOOR
    )


def reduce_world_isolation(
    records: Mapping[tuple[str,int], Sequence[WorldSnapshot]],
) -> tuple[str, tuple[tuple[str,int], ...]]:
    if set(records) != set(WORLD_KEYS):
        return "INVALID_WORLD_ISOLATION", ()
    passed: list[tuple[str,int]]=[]
    try:
        for key in WORLD_KEYS:
            rows=tuple(records[key])
            if len(rows)!=len(LOCAL_CHECKPOINTS):
                return "INVALID_WORLD_ISOLATION", ()
            if tuple(x.local_update for x in rows)!=LOCAL_CHECKPOINTS:
                return "INVALID_WORLD_ISOLATION", ()
            if any((x.family,x.world_index)!=key for x in rows):
                return "INVALID_WORLD_ISOLATION", ()
            for row in rows: validate_snapshot(row)
            if any(world_floor_pass(row) for row in rows):
                passed.append(key)
    except (TypeError,ValueError):
        return "INVALID_WORLD_ISOLATION", ()
    if len(passed)==len(WORLD_KEYS):
        return "ALL_WORLDS_SINGLE_FIT", tuple(passed)
    if passed:
        return "SOME_WORLDS_SINGLE_FIT", tuple(passed)
    return "NO_WORLDS_SINGLE_FIT", ()
