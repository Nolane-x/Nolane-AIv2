from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .exp319_metrics import global_gradient_norm, nonfinite_report, parameter_update_norm_ratio, snapshot_trainable_parameters
from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset, validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp331_runtime import _backward_grads
from .exp334_contract import (
    AUTHORIZATION_FLAGS,
    FAMILY,
    FULL_EXACT_FLOOR,
    GRADIENT_CLIP_NORM,
    GROUP_IDS,
    GROUP_MEMBERS,
    MODES,
    PARENT_CONTROL_PASS,
    PARENT_EXP326_FAMILY_EVIDENCE_DIGEST,
    PARENT_EXP327_EVIDENCE_DIGEST,
    PARENT_EXP333_EVIDENCE_DIGEST,
    PARENT_EXP333_EXECUTION_DIGEST,
    PINV_RTOL,
    TARGET_NORM_SQUARED_FLOOR,
    GroupArmResult,
    arm_pass,
    reduce_higher_order,
    sham_equivalent,
    training_schedule,
    validate_arm,
)
from .exp334_identity import Exp334ExecutionIdentity, validate_execution_identity

FINAL_SCHEMA = "EXP334-FINAL-EVIDENCE-V1"

def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()

def _metric_equal(left: object, right: object) -> bool:
    return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and tuple(left) == tuple(right)

def _world_map() -> dict[int, object]:
    worlds = {int(getattr(world, "index")): world for world in family_worlds(FAMILY)}
    if tuple(sorted(worlds)) != tuple(range(8)):
        raise ValueError("EXP-334 world population")
    return worlds

def validate_exp333_parent(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP333-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-334 EXP333 schema")
    if payload.get("decision") != "COMPLETE_8WORLD_PAIR_LATTICE_RESCUE_NO_REGRESSION":
        raise ValueError("EXP-334 EXP333 decision")
    if payload.get("evidence_digest") != PARENT_EXP333_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 EXP333 evidence")
    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    if _digest(materialized) != PARENT_EXP333_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 EXP333 canonical")
    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping) or identity.get("exp333_execution_digest") != PARENT_EXP333_EXECUTION_DIGEST:
        raise ValueError("EXP-334 EXP333 execution")
    if payload.get("project_regressions") != [] or len(payload.get("project_pass_pairs", [])) != 28:
        raise ValueError("EXP-334 EXP333 project vector")
    if payload.get("baseline_failures") != ["P02", "P06"] or payload.get("rescued_baseline_failures") != ["P02", "P06"]:
        raise ValueError("EXP-334 EXP333 rescue vector")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-334 EXP333 authorization drift: {key}")

def validate_exp327_parent(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP327-FINAL-EVIDENCE-V1" or payload.get("decision") != "PAIR_MINIMAL_FAILURE_PRESENT":
        raise ValueError("EXP-334 EXP327 parent")
    if payload.get("evidence_digest") != PARENT_EXP327_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 EXP327 evidence")
    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    if _digest(materialized) != PARENT_EXP327_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 EXP327 canonical")
    if payload.get("failed_triples") != ["T012", "T023"] or payload.get("failed_pairs") != ["P02"]:
        raise ValueError("EXP-334 EXP327 failure vector")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-334 EXP327 authorization drift: {key}")

def validate_exp326_family(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP326-FAMILY-GROUP-EVIDENCE-V1" or payload.get("family") != FAMILY:
        raise ValueError("EXP-334 EXP326 family")
    if payload.get("family_evidence_digest") != PARENT_EXP326_FAMILY_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 EXP326 evidence")
    materialized = dict(payload)
    materialized.pop("family_evidence_digest", None)
    if _digest(materialized) != PARENT_EXP326_FAMILY_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 EXP326 canonical")
    by = {row.get("group_id"): row for row in payload.get("groups", []) if isinstance(row, Mapping)}
    for group, passed in (("Q0123", False), ("Q4567", True), ("O01234567", False)):
        if group not in by or bool(by[group].get("passed")) is not passed:
            raise ValueError("EXP-334 EXP326 group vector")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-334 EXP326 authorization drift: {key}")

def _final_snapshot(group: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [x for x in group.get("snapshots", []) if x.get("exposures_per_world") == 32]
    if len(rows) != 1:
        raise ValueError("EXP-334 parent final snapshot")
    return rows[0]

def _historical_anchors(exp327: Mapping[str, Any], exp326: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by327 = {row.get("group_id"): row for row in exp327.get("groups", []) if isinstance(row, Mapping)}
    by326 = {row.get("group_id"): row for row in exp326.get("groups", []) if isinstance(row, Mapping)}
    anchors: dict[str, dict[str, Any]] = {}
    for group in ("T012", "T013", "T023", "T123", "Q0123"):
        row = by327[group]
        snap = _final_snapshot(row)
        anchors[group] = {
            "members": row["members"],
            "world_token_accuracies": snap["world_token_accuracies"],
            "world_full_answer_exact": snap["world_full_answer_exact"],
            "rng_state_digest": row["final_state"]["rng_state_digest"],
            "passed": bool(row["passed"]),
        }
    for group in ("Q4567", "O01234567"):
        row = by326[group]
        snap = _final_snapshot(row)
        anchors[group] = {
            "members": row["members"],
            "world_token_accuracies": snap["world_token_accuracies"],
            "world_full_answer_exact": snap["world_full_answer_exact"],
            "rng_state_digest": row["final_state"]["rng_state_digest"],
            "passed": bool(row["passed"]),
        }
    if {group: anchor["passed"] for group, anchor in anchors.items()} != PARENT_CONTROL_PASS:
        raise ValueError("EXP-334 anchor pass vector")
    return anchors

def _result(record: Mapping[str, Any]) -> GroupArmResult:
    return GroupArmResult(
        group_id=str(record["group_id"]),
        mode=str(record["mode"]),
        total_optimizer_updates=int(record["total_optimizer_updates"]),
        world_token_accuracies=tuple(record["world_token_accuracies"]),
        world_full_answer_exact=tuple(record["world_full_answer_exact"]),
        model_state_digest=str(record["model_state_digest"]),
        optimizer_state_digest=str(record["optimizer_state_digest"]),
        rng_state_digest=str(record["rng_state_digest"]),
        nonfinite_events=int(record["nonfinite_events"]),
        negative_target_count=int(record["negative_target_count"]),
        projection_update_count=int(record["projection_update_count"]),
        projected_target_count=int(record["projected_target_count"]),
    )

def _dot(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("EXP-334 gradient geometry")
    value = 0.0
    for x, y in zip(a, b):
        value += float(torch.sum(x * y, dtype=torch.float64).item())
    if not math.isfinite(value):
        raise ValueError("EXP-334 nonfinite gradient dot")
    return value

def _norm_sq(a: tuple[torch.Tensor, ...]) -> float:
    return _dot(a, a)

def _project_source(
    source: tuple[torch.Tensor, ...],
    targets: list[tuple[int, tuple[torch.Tensor, ...]]],
) -> tuple[tuple[torch.Tensor, ...], list[int], dict[int, float], dict[int, float]]:
    raw_dots = {index: _dot(source, grad) for index, grad in targets}
    selected = [
        (index, grad)
        for index, grad in targets
        if raw_dots[index] < 0.0 and _norm_sq(grad) > TARGET_NORM_SQUARED_FLOOR
    ]
    if not selected:
        return tuple(x.clone() for x in source), [], raw_dots, raw_dots.copy()

    n = len(selected)
    gram = torch.empty((n, n), dtype=torch.float64)
    b = torch.empty((n,), dtype=torch.float64)
    for i, (_, gi) in enumerate(selected):
        b[i] = _dot(gi, source)
        for j, (_, gj) in enumerate(selected):
            gram[i, j] = _dot(gi, gj)
    alpha = torch.linalg.pinv(gram, rtol=PINV_RTOL) @ b
    coeffs = [float(x.item()) for x in alpha]
    applied: list[torch.Tensor] = []
    for param_index, s in enumerate(source):
        value = s.clone()
        for coefficient, (_, target) in zip(coeffs, selected):
            value = value - target[param_index] * coefficient
        applied.append(value)
    projected = tuple(applied)
    post_dots = {index: _dot(projected, grad) for index, grad in targets}
    return projected, [index for index, _ in selected], raw_dots, post_dots

def _measured_step(
    compiled: object,
    source_world: object,
    target_worlds: list[tuple[int, object]],
    *,
    optimizer: torch.optim.Optimizer,
    effort: int,
    project: bool,
    exposure_index: int,
    source_index: int,
) -> dict[str, Any]:
    model = getattr(compiled, "model")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    before = snapshot_trainable_parameters(model.parameters())
    model.train()
    rng_start = torch.get_rng_state().clone()

    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_start)
    source_loss, source_grads, source_nonfinite = _backward_grads(compiled, source_world, effort)
    rng_after_source = torch.get_rng_state().clone()

    target_records: list[tuple[int, tuple[torch.Tensor, ...]]] = []
    target_nonfinite = 0
    target_losses: dict[int, float] = {}
    for target_index, target_world in target_worlds:
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_start)
        target_loss, target_grads, observed = _backward_grads(compiled, target_world, effort)
        target_records.append((target_index, target_grads))
        target_losses[target_index] = target_loss
        target_nonfinite += observed

    torch.set_rng_state(rng_after_source)
    optimizer.zero_grad(set_to_none=True)
    projected, selected, raw_dots, post_dots = _project_source(source_grads, target_records)
    applied_grads = projected if project else source_grads
    for parameter, gradient in zip(params, applied_grads):
        parameter.grad = gradient.clone()

    preclip = global_gradient_norm(model.parameters())
    applied_report = nonfinite_report(model.parameters(), loss=source_loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    after = snapshot_trainable_parameters(model.parameters())
    ratio = parameter_update_norm_ratio(before, after)
    after_report = nonfinite_report(model.parameters(), loss=source_loss)
    observed = source_nonfinite + target_nonfinite + applied_report.total + after_report.total

    negative_targets = [index for index, dot in raw_dots.items() if dot < 0.0]
    projected_targets = selected if project else []
    return {
        "exposure_index": exposure_index,
        "source_world": source_index,
        "target_worlds": [index for index, _ in target_worlds],
        "effort": effort,
        "source_loss": source_loss,
        "target_losses": {str(k): v for k, v in target_losses.items()},
        "negative_targets": negative_targets,
        "projected_targets": projected_targets,
        "raw_target_dots": {str(k): v for k, v in raw_dots.items()},
        "post_projection_target_dots": {str(k): v for k, v in post_dots.items()} if project else {},
        "gradient_norm_preclip": preclip,
        "parameter_update_norm_ratio": ratio,
        "nonfinite_events": observed,
    }

def _finalize(
    group_id: str,
    mode: str,
    compiled: object,
    optimizer: object,
    worlds: Mapping[int, object],
    nonfinite: int,
    measurements: list[dict[str, Any]],
    invalid_reason: str | None,
) -> dict[str, Any]:
    members = GROUP_MEMBERS[group_id]
    individual = tuple(_evaluate_subset(compiled, (worlds[index],)) for index in members)
    model = getattr(compiled, "model")
    result = GroupArmResult(
        group_id=group_id,
        mode=mode,
        total_optimizer_updates=len(members) * 32,
        world_token_accuracies=tuple(x["teacher_forced_answer_token_accuracy"] for x in individual),
        world_full_answer_exact=tuple(x["teacher_forced_full_answer_exact"] for x in individual),
        model_state_digest=model_state_digest(model),
        optimizer_state_digest=optimizer_state_digest(optimizer),
        rng_state_digest=rng_state_digest(),
        nonfinite_events=nonfinite,
        negative_target_count=sum(len(row["negative_targets"]) for row in measurements),
        projection_update_count=sum(bool(row["projected_targets"]) for row in measurements),
        projected_target_count=sum(len(row["projected_targets"]) for row in measurements),
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
    group_id: str,
    mode: str,
) -> dict[str, Any]:
    state = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    compiled, optimizer = state.compiled, state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds = _world_map()
    nonfinite = 0
    invalid_reason = None
    measurements: list[dict[str, Any]] = []
    members = GROUP_MEMBERS[group_id]
    for source_index, exposure_index, effort in training_schedule(group_id):
        if mode == "CONTROL":
            _, _, _, observed = _train_one_step_with_effort(
                compiled, worlds[source_index], optimizer=optimizer, effort=effort
            )
            nonfinite += observed
        else:
            targets = [(index, worlds[index]) for index in members if index != source_index]
            measurement = _measured_step(
                compiled,
                worlds[source_index],
                targets,
                optimizer=optimizer,
                effort=effort,
                project=mode == "SUBSPACE_PROJECT",
                exposure_index=exposure_index,
                source_index=source_index,
            )
            measurements.append(measurement)
            nonfinite += int(measurement["nonfinite_events"])
        if nonfinite:
            invalid_reason = "NONFINITE_ARM"
            break
    return _finalize(group_id, mode, compiled, optimizer, worlds, nonfinite, measurements, invalid_reason)

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
    parent_exp333_path: str | Path,
    parent_exp327_path: str | Path,
    parent_exp326_family_path: str | Path,
    execution_identity: Exp334ExecutionIdentity,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    p333 = json.loads(Path(parent_exp333_path).read_text(encoding="utf-8"))
    p327 = json.loads(Path(parent_exp327_path).read_text(encoding="utf-8"))
    p326 = json.loads(Path(parent_exp326_family_path).read_text(encoding="utf-8"))
    for payload in (p333, p327, p326):
        if not isinstance(payload, Mapping):
            raise ValueError("EXP-334 parent payload")
    validate_exp333_parent(p333)
    validate_exp327_parent(p327)
    validate_exp326_family(p326)
    anchors = _historical_anchors(p327, p326)

    authority = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    reconstruction = dict(authority.reconstruction)
    del authority

    arms = [
        _run_arm(checkpoint_path, receipt_path, selection_lock_path, group_id=group, mode=mode)
        for group in GROUP_IDS
        for mode in MODES
    ]
    results = {f"{arm['group_id']}:{arm['mode']}": _result(arm) for arm in arms}
    parent_reproduced = all(
        _control_matches_parent(
            next(arm for arm in arms if arm["group_id"] == group and arm["mode"] == "CONTROL"),
            anchors[group],
        )
        for group in GROUP_IDS
    )
    invalid = any(arm.get("invalid_reason") is not None for arm in arms)
    decision, passed, baseline_failures, rescued, unresolved, regressions, no_trigger = reduce_higher_order(
        results, parent_reproduced=parent_reproduced, invalid=invalid
    )
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": asdict(execution_identity),
        "reconstruction": reconstruction,
        "decision": decision,
        "parent_exp333_evidence_digest": PARENT_EXP333_EVIDENCE_DIGEST,
        "parent_exp327_evidence_digest": PARENT_EXP327_EVIDENCE_DIGEST,
        "parent_exp326_family_evidence_digest": PARENT_EXP326_FAMILY_EVIDENCE_DIGEST,
        "parent_higher_order_anchors": anchors,
        "parent_higher_order_reproduced": parent_reproduced,
        "arms": arms,
        "arm_pass": passed,
        "group_sham_match": {
            group: sham_equivalent(results[f"{group}:CONTROL"], results[f"{group}:SHAM"])
            for group in GROUP_IDS
        },
        "control_pass_groups": [group for group in GROUP_IDS if passed.get(f"{group}:CONTROL", False)],
        "baseline_failures": list(baseline_failures),
        "rescued_baseline_failures": list(rescued),
        "unresolved_baseline_failures": list(unresolved),
        "project_regressions": list(regressions),
        "baseline_failures_without_subspace_trigger": list(no_trigger),
        "project_pass_groups": [group for group in GROUP_IDS if passed.get(f"{group}:SUBSPACE_PROJECT", False)],
        "projection_update_counts": {
            group: results[f"{group}:SUBSPACE_PROJECT"].projection_update_count for group in GROUP_IDS
        },
        "projected_target_counts": {
            group: results[f"{group}:SUBSPACE_PROJECT"].projected_target_count for group in GROUP_IDS
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    validate_final_evidence(payload)
    return payload

def validate_final_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != FINAL_SCHEMA:
        raise ValueError("EXP-334 final schema")
    raw = payload.get("execution_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-334 identity")
    validate_execution_identity(Exp334ExecutionIdentity(**raw))
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-334 reconstruction")
    if payload.get("parent_exp333_evidence_digest") != PARENT_EXP333_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 parent EXP333")
    if payload.get("parent_exp327_evidence_digest") != PARENT_EXP327_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 parent EXP327")
    if payload.get("parent_exp326_family_evidence_digest") != PARENT_EXP326_FAMILY_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 parent EXP326")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != len(GROUP_IDS) * len(MODES):
        raise ValueError("EXP-334 arm coverage")
    results = {f"{arm.get('group_id')}:{arm.get('mode')}": _result(arm) for arm in arms}
    invalid = any(arm.get("invalid_reason") is not None for arm in arms)
    decision, passed, baseline_failures, rescued, unresolved, regressions, no_trigger = reduce_higher_order(
        results,
        parent_reproduced=payload.get("parent_higher_order_reproduced") is True,
        invalid=invalid,
    )
    if payload.get("decision") != decision or payload.get("arm_pass") != passed:
        raise ValueError("EXP-334 reducer")
    vector_checks = (
        (payload.get("baseline_failures"), list(baseline_failures)),
        (payload.get("rescued_baseline_failures"), list(rescued)),
        (payload.get("unresolved_baseline_failures"), list(unresolved)),
        (payload.get("project_regressions"), list(regressions)),
        (payload.get("baseline_failures_without_subspace_trigger"), list(no_trigger)),
        (payload.get("control_pass_groups"), [g for g in GROUP_IDS if passed.get(f"{g}:CONTROL", False)]),
        (payload.get("project_pass_groups"), [g for g in GROUP_IDS if passed.get(f"{g}:SUBSPACE_PROJECT", False)]),
    )
    if any(actual != expected for actual, expected in vector_checks):
        raise ValueError("EXP-334 reducer vectors")
    expected_sham = {
        group: sham_equivalent(results[f"{group}:CONTROL"], results[f"{group}:SHAM"])
        for group in GROUP_IDS
    }
    if payload.get("group_sham_match") != expected_sham:
        raise ValueError("EXP-334 sham integrity")
    expected_projection_updates = {
        group: results[f"{group}:SUBSPACE_PROJECT"].projection_update_count for group in GROUP_IDS
    }
    expected_projected_targets = {
        group: results[f"{group}:SUBSPACE_PROJECT"].projected_target_count for group in GROUP_IDS
    }
    if payload.get("projection_update_counts") != expected_projection_updates:
        raise ValueError("EXP-334 projection updates")
    if payload.get("projected_target_counts") != expected_projected_targets:
        raise ValueError("EXP-334 projected targets")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-334 authorization drift: {key}")
    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or _digest(materialized) != claimed:
        raise ValueError("EXP-334 evidence digest")
