from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset, validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp331_runtime import _measured_step
from .exp333_contract import (
    AUTHORIZATION_FLAGS,
    FAMILY,
    MODES,
    PAIR_IDS,
    PAIR_MEMBERS,
    PARENT_ANCHOR_PAIR_IDS,
    PARENT_CONTROL_PASS,
    PARENT_EXP332_EVIDENCE_DIGEST,
    PARENT_EXP332_EXECUTION_DIGEST,
    PairArmResult,
    arm_pass,
    reduce_complete_pair_lattice,
    sham_equivalent,
    training_schedule,
    validate_arm,
)
from .exp333_identity import Exp333ExecutionIdentity, validate_execution_identity

FINAL_SCHEMA = "EXP333-FINAL-EVIDENCE-V1"

def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()

def _metric_equal(left: object, right: object) -> bool:
    return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and tuple(left) == tuple(right)

def validate_parent_exp332(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP332-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-333 parent schema")
    if payload.get("decision") != "PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION":
        raise ValueError("EXP-333 parent disposition")
    if payload.get("evidence_digest") != PARENT_EXP332_EVIDENCE_DIGEST:
        raise ValueError("EXP-333 parent evidence")
    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    if _digest(materialized) != PARENT_EXP332_EVIDENCE_DIGEST:
        raise ValueError("EXP-333 parent canonical")
    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping) or identity.get("exp332_execution_digest") != PARENT_EXP332_EXECUTION_DIGEST:
        raise ValueError("EXP-333 parent execution identity")
    if payload.get("parent_behavior_reproduced") is not True:
        raise ValueError("EXP-333 parent reproduction")
    if payload.get("project_regressions") != []:
        raise ValueError("EXP-333 parent regressions")
    if payload.get("project_pass_pairs") != list(PARENT_ANCHOR_PAIR_IDS):
        raise ValueError("EXP-333 parent project vector")
    if payload.get("pair_sham_match") != {pair: True for pair in PARENT_ANCHOR_PAIR_IDS}:
        raise ValueError("EXP-333 parent sham vector")
    if int(payload.get("p02_projection_event_count", 0)) <= 0:
        raise ValueError("EXP-333 parent P02 projection")
    arm_pass_map = payload.get("arm_pass")
    if not isinstance(arm_pass_map, Mapping):
        raise ValueError("EXP-333 parent arm pass")
    if {pair: bool(arm_pass_map.get(f"{pair}:CONTROL")) for pair in PARENT_ANCHOR_PAIR_IDS} != PARENT_CONTROL_PASS:
        raise ValueError("EXP-333 parent control pass vector")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != len(PARENT_ANCHOR_PAIR_IDS) * 3:
        raise ValueError("EXP-333 parent arm coverage")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-333 parent authorization drift: {key}")

def _parent_control_anchor(parent: Mapping[str, Any], pair: str) -> dict[str, Any]:
    arm = next(
        row for row in parent["arms"]
        if row.get("pair_id") == pair and row.get("mode") == "CONTROL"
    )
    return {
        "members": arm["members"],
        "world_token_accuracies": arm["world_token_accuracies"],
        "world_full_answer_exact": arm["world_full_answer_exact"],
        "rng_state_digest": arm["rng_state_digest"],
        "passed": bool(parent["arm_pass"][f"{pair}:CONTROL"]),
    }

def _world_map() -> dict[int, object]:
    worlds = {int(getattr(world, "index")): world for world in family_worlds(FAMILY)}
    if tuple(sorted(worlds)) != tuple(range(8)):
        raise ValueError("EXP-333 world population")
    return worlds

def _result(record: Mapping[str, Any]) -> PairArmResult:
    return PairArmResult(
        pair_id=str(record["pair_id"]),
        mode=str(record["mode"]),
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

def _finalize(
    pair_id: str,
    mode: str,
    compiled: object,
    optimizer: object,
    worlds: Mapping[int, object],
    nonfinite: int,
    measurements: list[dict[str, Any]],
    invalid_reason: str | None,
) -> dict[str, Any]:
    members = PAIR_MEMBERS[pair_id]
    individual = tuple(_evaluate_subset(compiled, (worlds[index],)) for index in members)
    model = getattr(compiled, "model")
    result = PairArmResult(
        pair_id=pair_id,
        mode=mode,
        total_optimizer_updates=64,
        world_token_accuracies=tuple(x["teacher_forced_answer_token_accuracy"] for x in individual),
        world_full_answer_exact=tuple(x["teacher_forced_full_answer_exact"] for x in individual),
        model_state_digest=model_state_digest(model),
        optimizer_state_digest=optimizer_state_digest(optimizer),
        rng_state_digest=rng_state_digest(),
        nonfinite_events=nonfinite,
        negative_dot_count=sum(bool(row["negative_dot"]) for row in measurements),
        projection_event_count=sum(bool(row["projection_applied"]) for row in measurements),
    )
    validate_arm(result)
    return {
        **asdict(result),
        "members": list(members),
        "measurements": measurements,
        "per_world_greedy_exact": [x["greedy_exact"] for x in individual],
        "per_world_answer_only_loss": [x["answer_only_loss"] for x in individual],
        "invalid_reason": invalid_reason,
    }

def _run_arm(
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
    *,
    pair_id: str,
    mode: str,
) -> dict[str, Any]:
    state = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    compiled, optimizer = state.compiled, state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds = _world_map()
    nonfinite = 0
    invalid_reason = None
    measurements: list[dict[str, Any]] = []
    members = PAIR_MEMBERS[pair_id]
    for source_index, round_index, effort in training_schedule(pair_id):
        if mode == "CONTROL":
            _, _, _, observed = _train_one_step_with_effort(
                compiled,
                worlds[source_index],
                optimizer=optimizer,
                effort=effort,
            )
            nonfinite += observed
        else:
            target_index = members[1] if source_index == members[0] else members[0]
            measurement = _measured_step(
                compiled,
                worlds[source_index],
                worlds[target_index],
                optimizer=optimizer,
                effort=effort,
                project=mode == "PROJECT",
                round_index=round_index,
                source_index=source_index,
                target_index=target_index,
            )
            measurements.append(measurement)
            nonfinite += int(measurement["nonfinite_events"])
        if nonfinite:
            invalid_reason = "NONFINITE_ARM"
            break
    return _finalize(
        pair_id,
        mode,
        compiled,
        optimizer,
        worlds,
        nonfinite,
        measurements,
        invalid_reason,
    )

def _control_matches_parent(record: Mapping[str, Any], anchor: Mapping[str, Any]) -> bool:
    return (
        record.get("mode") == "CONTROL"
        and record.get("members") == anchor["members"]
        and _metric_equal(record.get("world_token_accuracies"), anchor["world_token_accuracies"])
        and _metric_equal(record.get("world_full_answer_exact"), anchor["world_full_answer_exact"])
        and record.get("rng_state_digest") == anchor["rng_state_digest"]
        and record.get("nonfinite_events") == 0
        and arm_pass(_result(record)) is anchor["passed"]
    )

def run_court(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
    parent_exp332_path: str | Path,
    execution_identity: Exp333ExecutionIdentity,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    parent = json.loads(Path(parent_exp332_path).read_text(encoding="utf-8"))
    if not isinstance(parent, Mapping):
        raise ValueError("EXP-333 parent payload")
    validate_parent_exp332(parent)

    authority = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    reconstruction = dict(authority.reconstruction)
    del authority

    arms = [
        _run_arm(
            checkpoint_path,
            receipt_path,
            selection_lock_path,
            pair_id=pair,
            mode=mode,
        )
        for pair in PAIR_IDS
        for mode in MODES
    ]
    results = {f"{arm['pair_id']}:{arm['mode']}": _result(arm) for arm in arms}
    anchors = {pair: _parent_control_anchor(parent, pair) for pair in PARENT_ANCHOR_PAIR_IDS}
    parent_anchor_reproduced = all(
        _control_matches_parent(
            next(
                arm for arm in arms
                if arm["pair_id"] == pair and arm["mode"] == "CONTROL"
            ),
            anchors[pair],
        )
        for pair in PARENT_ANCHOR_PAIR_IDS
    )
    invalid = any(arm.get("invalid_reason") is not None for arm in arms)
    (
        decision,
        passed,
        baseline_failures,
        rescued,
        unresolved,
        regressions,
        no_trigger,
    ) = reduce_complete_pair_lattice(
        results,
        parent_anchor_reproduced=parent_anchor_reproduced,
        invalid=invalid,
    )
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": asdict(execution_identity),
        "reconstruction": reconstruction,
        "decision": decision,
        "parent_exp332_evidence_digest": PARENT_EXP332_EVIDENCE_DIGEST,
        "parent_anchor_reproduced": parent_anchor_reproduced,
        "arms": arms,
        "arm_pass": passed,
        "pair_sham_match": {
            pair: sham_equivalent(results[f"{pair}:CONTROL"], results[f"{pair}:SHAM"])
            for pair in PAIR_IDS
        },
        "control_pass_pairs": [pair for pair in PAIR_IDS if passed.get(f"{pair}:CONTROL", False)],
        "baseline_failures": list(baseline_failures),
        "new_baseline_failures": [pair for pair in baseline_failures if pair not in PARENT_ANCHOR_PAIR_IDS],
        "rescued_baseline_failures": list(rescued),
        "unresolved_baseline_failures": list(unresolved),
        "project_regressions": list(regressions),
        "baseline_failures_without_projection_trigger": list(no_trigger),
        "project_pass_pairs": [pair for pair in PAIR_IDS if passed.get(f"{pair}:PROJECT", False)],
        "projection_event_counts": {
            pair: results[f"{pair}:PROJECT"].projection_event_count
            for pair in PAIR_IDS
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    validate_final_evidence(payload)
    return payload

def validate_final_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != FINAL_SCHEMA:
        raise ValueError("EXP-333 final schema")
    raw = payload.get("execution_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-333 identity")
    validate_execution_identity(Exp333ExecutionIdentity(**raw))
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-333 reconstruction")
    if payload.get("parent_exp332_evidence_digest") != PARENT_EXP332_EVIDENCE_DIGEST:
        raise ValueError("EXP-333 parent authority")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != len(PAIR_IDS) * len(MODES):
        raise ValueError("EXP-333 arm coverage")
    results = {f"{arm.get('pair_id')}:{arm.get('mode')}": _result(arm) for arm in arms}
    invalid = any(arm.get("invalid_reason") is not None for arm in arms)
    (
        decision,
        passed,
        baseline_failures,
        rescued,
        unresolved,
        regressions,
        no_trigger,
    ) = reduce_complete_pair_lattice(
        results,
        parent_anchor_reproduced=payload.get("parent_anchor_reproduced") is True,
        invalid=invalid,
    )
    if payload.get("decision") != decision or payload.get("arm_pass") != passed:
        raise ValueError("EXP-333 reducer decision")
    checks = (
        (payload.get("baseline_failures"), list(baseline_failures)),
        (payload.get("rescued_baseline_failures"), list(rescued)),
        (payload.get("unresolved_baseline_failures"), list(unresolved)),
        (payload.get("project_regressions"), list(regressions)),
        (payload.get("baseline_failures_without_projection_trigger"), list(no_trigger)),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("EXP-333 reducer vectors")
    expected_sham = {
        pair: sham_equivalent(results[f"{pair}:CONTROL"], results[f"{pair}:SHAM"])
        for pair in PAIR_IDS
    }
    if payload.get("pair_sham_match") != expected_sham:
        raise ValueError("EXP-333 sham integrity")
    expected_control_pass = [pair for pair in PAIR_IDS if passed.get(f"{pair}:CONTROL", False)]
    expected_new_failures = [pair for pair in baseline_failures if pair not in PARENT_ANCHOR_PAIR_IDS]
    expected_project_pass = [pair for pair in PAIR_IDS if passed.get(f"{pair}:PROJECT", False)]
    vector_checks = (
        (payload.get("control_pass_pairs"), expected_control_pass),
        (payload.get("new_baseline_failures"), expected_new_failures),
        (payload.get("project_pass_pairs"), expected_project_pass),
    )
    if any(actual != expected for actual, expected in vector_checks):
        raise ValueError("EXP-333 derived vectors")
    expected_counts = {
        pair: results[f"{pair}:PROJECT"].projection_event_count
        for pair in PAIR_IDS
    }
    if payload.get("projection_event_counts") != expected_counts:
        raise ValueError("EXP-333 projection counts")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-333 authorization drift: {key}")
    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or _digest(materialized) != claimed:
        raise ValueError("EXP-333 evidence digest")
