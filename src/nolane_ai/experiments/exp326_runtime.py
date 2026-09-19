from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .exp301_scientific import forward_scientific_arm
from .exp301_training import compute_answer_only_loss
from .exp319_metrics import (
    global_gradient_norm,
    nonfinite_report,
    parameter_update_norm_ratio,
    snapshot_trainable_parameters,
)
from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp322_runtime import _encoded_tensors
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset, validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_contract import (
    AUTHORIZATION_FLAGS,
    EXPOSURE_CHECKPOINTS,
    EXPOSURES_PER_WORLD,
    FAMILIES,
    GROUP_IDS,
    GROUP_KEYS,
    GROUP_MEMBERS,
    LEARNING_RATE,
    GroupSnapshot,
    canonical_json_bytes,
    effort_for_world_exposure,
    group_floor_pass,
    reduce_breakpoint,
    validate_snapshot,
)
from .exp326_identity import (
    PARENT_EXECUTION_DIGEST,
    PARENT_FINAL_EVIDENCE_DIGEST,
    Exp326ExecutionIdentity,
    validate_execution_identity,
)


FAMILY_SCHEMA = "EXP326-FAMILY-GROUP-EVIDENCE-V1"
FINAL_SCHEMA = "EXP326-FINAL-EVIDENCE-V1"


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_parent_final(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP325-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-326 parent final schema mismatch")
    if payload.get("decision") != "ALL_WORLDS_SINGLE_FIT":
        raise ValueError("EXP-326 parent disposition mismatch")
    if payload.get("evidence_digest") != PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-326 parent evidence digest mismatch")
    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    if _digest(materialized) != PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-326 parent canonical evidence mismatch")
    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping) or identity.get("exp325_execution_digest") != PARENT_EXECUTION_DIGEST:
        raise ValueError("EXP-326 parent execution identity mismatch")
    if payload.get("passed_world_count") != 32 or payload.get("failed_worlds") != []:
        raise ValueError("EXP-326 parent world-fit result mismatch")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-326 parent authorization drift: {key}")


def group_training_schedule(group_id: str) -> tuple[tuple[int, int, int], ...]:
    if group_id not in GROUP_MEMBERS:
        raise ValueError("unknown EXP-326 group")
    members = GROUP_MEMBERS[group_id]
    rows: list[tuple[int, int, int]] = []
    for exposure_index in range(EXPOSURES_PER_WORLD):
        effort = effort_for_world_exposure(exposure_index)
        for world_index in members:
            rows.append((world_index, exposure_index, effort))
    return tuple(rows)


def _train_one_step_with_effort(
    compiled: object,
    world: object,
    *,
    optimizer: torch.optim.Optimizer,
    effort: int,
) -> tuple[float, float, float, int]:
    if effort not in (1, 2, 4, 8):
        raise ValueError("EXP-326 effort outside frozen cycle")
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    input_ids, targets, answer_start = _encoded_tensors(world, device=device)
    before = snapshot_trainable_parameters(model.parameters())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = forward_scientific_arm(compiled, input_ids, effort=effort)
    loss = compute_answer_only_loss(logits, targets=targets, answer_start=answer_start)
    loss.backward()
    preclip = global_gradient_norm(model.parameters())
    before_clip = nonfinite_report(model.parameters(), loss=loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    after = snapshot_trainable_parameters(model.parameters())
    update_ratio = parameter_update_norm_ratio(before, after)
    after_step = nonfinite_report(model.parameters(), loss=loss)
    return (
        float(loss.detach().cpu().item()),
        preclip,
        update_ratio,
        before_clip.total + after_step.total,
    )


def _group_worlds(family: str, group_id: str) -> tuple[object, ...]:
    worlds = family_worlds(family)
    members = GROUP_MEMBERS[group_id]
    selected = tuple(worlds[i] for i in members)
    if tuple(int(getattr(w, "index")) for w in selected) != members:
        raise ValueError("EXP-326 group world identity drift")
    return selected


def _snapshot(
    *,
    family: str,
    group_id: str,
    exposures_per_world: int,
    compiled: object,
) -> tuple[GroupSnapshot, dict[str, Any]]:
    worlds = _group_worlds(family, group_id)
    individual = tuple(_evaluate_subset(compiled, (world,)) for world in worlds)
    snap = GroupSnapshot(
        family=family,
        group_id=group_id,
        exposures_per_world=exposures_per_world,
        total_optimizer_updates=len(worlds) * exposures_per_world,
        world_token_accuracies=tuple(x["teacher_forced_answer_token_accuracy"] for x in individual),
        world_full_answer_exact=tuple(x["teacher_forced_full_answer_exact"] for x in individual),
    )
    validate_snapshot(snap)
    aggregate = _evaluate_subset(compiled, worlds)
    return snap, {
        "group_teacher_forced_answer_token_accuracy": aggregate["teacher_forced_answer_token_accuracy"],
        "group_teacher_forced_full_answer_exact": aggregate["teacher_forced_full_answer_exact"],
        "group_greedy_exact": aggregate["greedy_exact"],
        "group_answer_only_loss": aggregate["answer_only_loss"],
        "per_world_greedy_exact": [x["greedy_exact"] for x in individual],
        "per_world_answer_only_loss": [x["answer_only_loss"] for x in individual],
    }


def _run_group(
    *,
    family: str,
    group_id: str,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
) -> dict[str, Any]:
    state = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    compiled, optimizer = state.compiled, state.optimizer
    validate_optimizer_invariants(optimizer)
    worlds_by_index = {int(getattr(w, "index")): w for w in family_worlds(family)}
    snapshots: list[GroupSnapshot] = []
    secondary: dict[str, Any] = {}
    nonfinite = 0
    last_grad = last_update = 0.0
    invalid_reason: str | None = None

    for world_index, exposure_index, effort in group_training_schedule(group_id):
        _, last_grad, last_update, observed = _train_one_step_with_effort(
            compiled,
            worlds_by_index[world_index],
            optimizer=optimizer,
            effort=effort,
        )
        nonfinite += observed
        completed_exposures = exposure_index + 1
        is_end_of_round = world_index == GROUP_MEMBERS[group_id][-1]
        if nonfinite:
            invalid_reason = "NONFINITE_EVENT"
            break
        if is_end_of_round and completed_exposures in EXPOSURE_CHECKPOINTS:
            snap, metrics = _snapshot(
                family=family,
                group_id=group_id,
                exposures_per_world=completed_exposures,
                compiled=compiled,
            )
            snapshots.append(snap)
            secondary[str(completed_exposures)] = {
                **metrics,
                "gradient_norm_preclip": last_grad,
                "parameter_update_norm_ratio": last_update,
                "nonfinite_events": nonfinite,
            }

    model = getattr(compiled, "model")
    return {
        "family": family,
        "group_id": group_id,
        "members": list(GROUP_MEMBERS[group_id]),
        "group_size": len(GROUP_MEMBERS[group_id]),
        "exposures_per_world": EXPOSURES_PER_WORLD,
        "learning_rate": LEARNING_RATE,
        "snapshots": [asdict(x) for x in snapshots],
        "secondary": secondary,
        "passed": any(group_floor_pass(x) for x in snapshots) if invalid_reason is None else False,
        "final_state": {
            "model_state_digest": model_state_digest(model),
            "optimizer_state_digest": optimizer_state_digest(optimizer),
            "rng_state_digest": rng_state_digest(),
        },
        "invalid_reason": invalid_reason,
    }


def _validate_group_record(record: Mapping[str, Any], expected_family: str) -> None:
    key = (record.get("family"), record.get("group_id"))
    if key not in GROUP_KEYS or record.get("family") != expected_family:
        raise ValueError("EXP-326 group record identity mismatch")
    group_id = str(record["group_id"])
    if record.get("members") != list(GROUP_MEMBERS[group_id]):
        raise ValueError("EXP-326 group members mismatch")
    if record.get("group_size") != len(GROUP_MEMBERS[group_id]):
        raise ValueError("EXP-326 group size mismatch")
    if record.get("exposures_per_world") != EXPOSURES_PER_WORLD or record.get("learning_rate") != LEARNING_RATE:
        raise ValueError("EXP-326 training geometry mismatch")
    rows = tuple(GroupSnapshot(**x) for x in record.get("snapshots", ()))
    if record.get("invalid_reason") is None:
        if tuple(x.exposures_per_world for x in rows) != EXPOSURE_CHECKPOINTS:
            raise ValueError("EXP-326 group checkpoint geometry mismatch")
    for row in rows:
        validate_snapshot(row)
        if (row.family, row.group_id) != key:
            raise ValueError("EXP-326 snapshot identity mismatch")
    expected_pass = any(group_floor_pass(x) for x in rows) if record.get("invalid_reason") is None else False
    if record.get("passed") is not expected_pass:
        raise ValueError("EXP-326 group pass flag mismatch")
    final_state = record.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("EXP-326 final state missing")
    for field in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
        value = final_state.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"EXP-326 malformed final digest: {field}")


def build_family_evidence(
    *,
    family: str,
    identity: Exp326ExecutionIdentity,
    reconstruction: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": FAMILY_SCHEMA,
        "family": family,
        "execution_identity": asdict(identity),
        "reconstruction": dict(reconstruction),
        "groups": [dict(x) for x in groups],
        **AUTHORIZATION_FLAGS,
    }
    validate_family_evidence(payload, with_digest=False)
    payload["family_evidence_digest"] = _digest(payload)
    return payload


def validate_family_evidence(payload: Mapping[str, Any], *, with_digest: bool = True) -> None:
    family = payload.get("family")
    if payload.get("schema") != FAMILY_SCHEMA or family not in FAMILIES:
        raise ValueError("EXP-326 family evidence schema/family mismatch")
    raw = payload.get("execution_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-326 execution identity missing")
    validate_execution_identity(Exp326ExecutionIdentity(**raw))
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-326 reconstruction authority mismatch")
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != len(GROUP_IDS):
        raise ValueError("EXP-326 family requires all seven groups")
    seen: set[str] = set()
    for record in groups:
        if not isinstance(record, Mapping):
            raise ValueError("EXP-326 group record must be object")
        _validate_group_record(record, str(family))
        gid = str(record["group_id"])
        if gid in seen:
            raise ValueError("duplicate EXP-326 group record")
        seen.add(gid)
    if seen != set(GROUP_IDS):
        raise ValueError("EXP-326 family group coverage mismatch")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-326 authorization drift: {key}")
    if with_digest:
        materialized = dict(payload)
        claimed = materialized.pop("family_evidence_digest", None)
        if not isinstance(claimed, str) or _digest(materialized) != claimed:
            raise ValueError("EXP-326 family evidence digest mismatch")


def run_family_groups(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
    parent_final_path: str | Path,
    family: str,
    execution_identity: Exp326ExecutionIdentity,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    parent = json.loads(Path(parent_final_path).read_text(encoding="utf-8"))
    if not isinstance(parent, Mapping):
        raise ValueError("EXP-326 parent final must be object")
    validate_parent_final(parent)
    authority = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    reconstruction = dict(authority.reconstruction)
    del authority
    groups = tuple(
        _run_group(
            family=family,
            group_id=group_id,
            checkpoint_path=checkpoint_path,
            receipt_path=receipt_path,
            selection_lock_path=selection_lock_path,
        )
        for group_id in GROUP_IDS
    )
    return build_family_evidence(
        family=family,
        identity=execution_identity,
        reconstruction=reconstruction,
        groups=groups,
    )


def _family_breakpoint(
    family: str,
    passed: Mapping[tuple[str, str], bool],
    violations: Sequence[tuple[str, str, str]],
) -> str:
    if any(v[0] == family for v in violations):
        return "NONMONOTONIC"
    if any(not passed[(family, gid)] for gid in ("P01", "P23", "P45", "P67")):
        return "2"
    if any(not passed[(family, gid)] for gid in ("Q0123", "Q4567")):
        return "4"
    if not passed[(family, "O01234567")]:
        return "8"
    return ">8"


def build_final_evidence(family_arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Mapping[str, Any]] = {}
    for arm in family_arms:
        validate_family_evidence(arm)
        family = str(arm["family"])
        if family in by_family:
            raise ValueError("duplicate EXP-326 family evidence")
        by_family[family] = arm
    if set(by_family) != set(FAMILIES):
        raise ValueError("EXP-326 requires all four family evidence objects")
    identities = {canonical_json_bytes(x["execution_identity"]) for x in by_family.values()}
    reconstructions = {canonical_json_bytes(x["reconstruction"]) for x in by_family.values()}
    if len(identities) != 1 or len(reconstructions) != 1:
        raise ValueError("EXP-326 family authority mismatch")

    records: dict[tuple[str, str], tuple[GroupSnapshot, ...]] = {}
    invalid = False
    for family in FAMILIES:
        for record in by_family[family]["groups"]:
            key = (family, str(record["group_id"]))
            records[key] = tuple(GroupSnapshot(**x) for x in record["snapshots"])
            invalid = invalid or record.get("invalid_reason") is not None

    if invalid:
        decision, passed, violations = "INVALID_BREAKPOINT_COURT", {}, ()
    else:
        decision, passed, violations = reduce_breakpoint(records)

    def count(ids: Sequence[str]) -> int:
        return sum(bool(passed.get((family, gid), False)) for family in FAMILIES for gid in ids)

    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": by_family[FAMILIES[0]]["execution_identity"],
        "reconstruction": by_family[FAMILIES[0]]["reconstruction"],
        "decision": decision,
        "pair_pass_count": count(("P01", "P23", "P45", "P67")),
        "quartet_pass_count": count(("Q0123", "Q4567")),
        "octet_pass_count": count(("O01234567",)),
        "pair_group_total": 16,
        "quartet_group_total": 8,
        "octet_group_total": 4,
        "family_breakpoints": {
            family: _family_breakpoint(family, passed, violations) if passed else "INVALID"
            for family in FAMILIES
        },
        "monotonicity_violations": [
            {"family": f, "parent_group": p, "child_group": c}
            for f, p, c in violations
        ],
        "group_pass": {
            f"{family}:{gid}": bool(passed.get((family, gid), False))
            for family in FAMILIES
            for gid in GROUP_IDS
        },
        "family_evidence_digests": {
            family: by_family[family]["family_evidence_digest"] for family in FAMILIES
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    return payload
