from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "EXP326-BALANCED-MULTIWORLD-BREAKPOINT-V1"
EXPERIMENT_ID = "EXP-326"
FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
GROUP_MEMBERS = {
    "P01": (0, 1),
    "P23": (2, 3),
    "P45": (4, 5),
    "P67": (6, 7),
    "Q0123": (0, 1, 2, 3),
    "Q4567": (4, 5, 6, 7),
    "O01234567": (0, 1, 2, 3, 4, 5, 6, 7),
}
GROUP_IDS = tuple(GROUP_MEMBERS)
GROUP_KEYS = tuple((family, group_id) for family in FAMILIES for group_id in GROUP_IDS)
PAIR_IDS = ("P01", "P23", "P45", "P67")
QUARTET_IDS = ("Q0123", "Q4567")
OCTET_IDS = ("O01234567",)
EXPOSURE_CHECKPOINTS = (8, 16, 32)
EFFORT_CYCLE = (1, 2, 4, 8)
EXPOSURES_PER_WORLD = 32
LEARNING_RATE = 5e-5
TOKEN_FLOOR = 0.99
FULL_EXACT_FLOOR = 0.90
DISPOSITIONS = (
    "INVALID_BREAKPOINT_COURT",
    "NONMONOTONIC_GROUP_FIT",
    "PAIR_LEVEL_BREAK_PRESENT",
    "QUARTET_LEVEL_BREAK_PRESENT",
    "OCTET_LEVEL_BREAK_PRESENT",
    "NO_BREAK_THROUGH_EIGHT",
)
AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "authorized_30m": False,
    "authorized_100m": False,
}
_PREREG_JSON = r'''{"authority":{"parent_exp325_disposition":"ALL_WORLDS_SINGLE_FIT","parent_exp325_execution_digest":"5c3ebd63682b7678cfd40545ef2c2bd6f98de539a7d931f6e47faa8ed21e7f5e","parent_exp325_final_artifact_id":10572536792,"parent_exp325_final_artifact_zip_digest":"e6b1cda21aa84351ec0d61fd4e8a0bfab84177fc5faa5c8ea169030f5badcd32","parent_exp325_final_evidence_digest":"0ce6a8bd4f96864bf754b22fc7c9b95552f6081225a4c305394d92ed39e7cbc2","parent_exp325_final_json_sha256":"bbf4aca8cae39ff424ad2fa4cea0c122a60b405162198413fb921dd891dcaf3c","parent_exp325_marker_sha":"b24830c842258258f42ffb92e5f00277f8db8e9b","parent_exp325_run_id":35406123945,"parent_exp325_source_sha":"3d1432e78fe85341f45f3de39ddefacdf7fd92bd","reconstruction_artifact_id":10547681681,"reconstruction_artifact_name":"exp323r-reconstruction-candidate-35345351869-0","reconstruction_artifact_zip_digest":"c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621","reconstruction_checkpoint_sha256":"4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5","reconstruction_receipt_digest":"f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f","reconstruction_run_id":35345351869},"authorization":{"authorized_100m":false,"authorized_30m":false,"exp302_implementation_authorized":false,"exp320_implementation_authorized":false,"scale_authorized":false},"decision_order":["INVALID_BREAKPOINT_COURT","NONMONOTONIC_GROUP_FIT","PAIR_LEVEL_BREAK_PRESENT","QUARTET_LEVEL_BREAK_PRESENT","OCTET_LEVEL_BREAK_PRESENT","NO_BREAK_THROUGH_EIGHT"],"dispositions":["INVALID_BREAKPOINT_COURT","NONMONOTONIC_GROUP_FIT","PAIR_LEVEL_BREAK_PRESENT","QUARTET_LEVEL_BREAK_PRESENT","OCTET_LEVEL_BREAK_PRESENT","NO_BREAK_THROUGH_EIGHT"],"experiment_id":"EXP-326","historical_context":{"confound_note":"EXP-324 round-robin used global-step effort assignment; with eight worlds, a world can repeatedly occupy the same effort residue. EXP-326 does not treat EXP-324 k=8 as a clean balanced-effort endpoint.","exp324_disposition":"NO_FAMILIES_ISOLATED_FIT","exp324_evidence_digest":"10a493d2f4de15131706b933af4170f39f55e802f069e740f9f98846713ac2ad","exp324_run_id":35362450626},"interpretation":{"NONMONOTONIC_GROUP_FIT":"a registered larger nested group fits while at least one of its registered child groups fails; group identity/order effects dominate a simple monotonic breakpoint interpretation","NO_BREAK_THROUGH_EIGHT":"all registered balanced-effort groups through eight worlds fit; the earlier family-isolation failure cannot be attributed to group size alone under this corrected schedule","OCTET_LEVEL_BREAK_PRESENT":"all registered pairs and quartets fit, but at least one registered eight-world group fails; the earliest registered monotonic break lies between four and eight worlds","PAIR_LEVEL_BREAK_PRESENT":"at least one registered two-world group fails despite balanced per-world effort/exposure; interference can emerge with only two shared worlds","QUARTET_LEVEL_BREAK_PRESENT":"all registered pairs fit, but at least one registered four-world group fails; the earliest registered monotonic break lies between two and four worlds"},"model":{"architecture_change":false,"arm":"A_FIXED","device":"cpu","objective_change":false,"resident_trainable_parameters":10000000,"scale_change":false,"starting_cumulative_step":2048,"tokenizer_change":false},"nonclaims":["EXP-326 does not establish a universal interference threshold outside the frozen 32 Stage-A worlds","EXP-326 does not establish a 10M architectural capacity ceiling","EXP-326 does not establish that scaling would help","EXP-326 does not identify a unique mechanism if interference is observed","EXP-326 does not authorize EXP-320 or larger model sizes"],"population":{"families":["iterative-grid-and-maze","algorithmic-sequence-transform","generator-heldout-abstract-transformation","language-sequence-control"],"grouping":{"octet":[[0,1,2,3,4,5,6,7]],"pairs":[[0,1],[2,3],[4,5],[6,7]],"quartets":[[0,1,2,3],[4,5,6,7]]},"independent_clone_per_group":true,"indices_per_family":[0,1,2,3,4,5,6,7],"root":0,"stage":"A_SANITY","total_worlds":32},"primary_measurements":["per_world_teacher_forced_answer_token_accuracy","per_world_teacher_forced_full_answer_exact","group_all_worlds_floor_pass"],"role":"post_exp325_balanced_multiworld_interference_breakpoint","schema_version":"EXP326-BALANCED-MULTIWORLD-BREAKPOINT-V1","secondary_measurements":["per_world_greedy_exact","group_teacher_forced_answer_token_accuracy","group_teacher_forced_full_answer_exact","group_answer_only_loss","gradient_norm_preclip","parameter_update_norm_ratio","nonfinite_events"],"thresholds":{"nonfinite_events_max":0,"teacher_forced_full_answer_exact_floor":0.9,"teacher_forced_token_accuracy_floor":0.99},"training":{"effort_assignment":"world-local exposure index; exposure j uses effort_cycle[j mod 4] for every world","effort_cycle":[1,2,4,8],"exposure_checkpoints_per_world":[8,16,32],"exposures_per_world":32,"gradient_clip_norm_decimal":"1.0","inherited_optimizer_state":true,"inherited_rng_state":true,"learning_rate_decimal":"0.00005","optimizer":"AdamW","rationale":"hold per-world exposure count and per-world effort trace equal to EXP-325 while increasing only the number of simultaneously shared worlds","updates_per_octet":256,"updates_per_pair":64,"updates_per_quartet":128,"weight_decay_decimal":"0.01","within_exposure_order":"ascending world index within the registered group"}}'''


@dataclass(frozen=True, slots=True)
class GroupSnapshot:
    family: str
    group_id: str
    exposures_per_world: int
    total_optimizer_updates: int
    world_token_accuracies: tuple[float, ...]
    world_full_answer_exact: tuple[float, ...]


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def preregistration_payload() -> dict[str, Any]:
    payload = json.loads(_PREREG_JSON)
    if not isinstance(payload, dict):
        raise AssertionError("EXP-326 preregistration root must be object")
    return payload


def preregistration_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(preregistration_payload())).hexdigest()


def effort_for_world_exposure(exposure_index: int) -> int:
    if not isinstance(exposure_index, int) or not 0 <= exposure_index < EXPOSURES_PER_WORLD:
        raise ValueError("EXP-326 exposure index out of range")
    return EFFORT_CYCLE[exposure_index % len(EFFORT_CYCLE)]


def expected_effort_trace() -> tuple[int, ...]:
    return tuple(effort_for_world_exposure(i) for i in range(EXPOSURES_PER_WORLD))


def validate_snapshot(snapshot: GroupSnapshot) -> None:
    key = (snapshot.family, snapshot.group_id)
    if key not in GROUP_KEYS:
        raise ValueError("unknown EXP-326 group identity")
    members = GROUP_MEMBERS[snapshot.group_id]
    if snapshot.exposures_per_world not in EXPOSURE_CHECKPOINTS:
        raise ValueError("unknown EXP-326 exposure checkpoint")
    if snapshot.total_optimizer_updates != len(members) * snapshot.exposures_per_world:
        raise ValueError("EXP-326 total update geometry mismatch")
    if len(snapshot.world_token_accuracies) != len(members):
        raise ValueError("EXP-326 token metric cardinality mismatch")
    if len(snapshot.world_full_answer_exact) != len(members):
        raise ValueError("EXP-326 full-exact metric cardinality mismatch")
    for field, values in (
        ("world_token_accuracies", snapshot.world_token_accuracies),
        ("world_full_answer_exact", snapshot.world_full_answer_exact),
    ):
        for value in values:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must contain finite values in [0,1]")


def group_floor_pass(snapshot: GroupSnapshot) -> bool:
    validate_snapshot(snapshot)
    return all(
        token >= TOKEN_FLOOR and full >= FULL_EXACT_FLOOR
        for token, full in zip(
            snapshot.world_token_accuracies,
            snapshot.world_full_answer_exact,
        )
    )


def _validated_pass_map(
    records: Mapping[tuple[str, str], Sequence[GroupSnapshot]],
) -> dict[tuple[str, str], bool] | None:
    if set(records) != set(GROUP_KEYS):
        return None
    passed: dict[tuple[str, str], bool] = {}
    try:
        for key in GROUP_KEYS:
            rows = tuple(records[key])
            if len(rows) != len(EXPOSURE_CHECKPOINTS):
                return None
            if tuple(row.exposures_per_world for row in rows) != EXPOSURE_CHECKPOINTS:
                return None
            if any((row.family, row.group_id) != key for row in rows):
                return None
            for row in rows:
                validate_snapshot(row)
            passed[key] = any(group_floor_pass(row) for row in rows)
    except (TypeError, ValueError):
        return None
    return passed


def monotonicity_violations(
    passed: Mapping[tuple[str, str], bool],
) -> tuple[tuple[str, str, str], ...]:
    violations: list[tuple[str, str, str]] = []
    for family in FAMILIES:
        for parent, children in (
            ("Q0123", ("P01", "P23")),
            ("Q4567", ("P45", "P67")),
            ("O01234567", ("Q0123", "Q4567")),
        ):
            if passed[(family, parent)]:
                for child in children:
                    if not passed[(family, child)]:
                        violations.append((family, parent, child))
    return tuple(violations)


def reduce_breakpoint(
    records: Mapping[tuple[str, str], Sequence[GroupSnapshot]],
) -> tuple[str, dict[tuple[str, str], bool], tuple[tuple[str, str, str], ...]]:
    passed = _validated_pass_map(records)
    if passed is None:
        return "INVALID_BREAKPOINT_COURT", {}, ()
    violations = monotonicity_violations(passed)
    if violations:
        return "NONMONOTONIC_GROUP_FIT", passed, violations
    if any(not passed[(family, group_id)] for family in FAMILIES for group_id in PAIR_IDS):
        return "PAIR_LEVEL_BREAK_PRESENT", passed, ()
    if any(not passed[(family, group_id)] for family in FAMILIES for group_id in QUARTET_IDS):
        return "QUARTET_LEVEL_BREAK_PRESENT", passed, ()
    if any(not passed[(family, group_id)] for family in FAMILIES for group_id in OCTET_IDS):
        return "OCTET_LEVEL_BREAK_PRESENT", passed, ()
    return "NO_BREAK_THROUGH_EIGHT", passed, ()
