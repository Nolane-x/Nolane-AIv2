from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp319_worlds import materialize_stage_a
from .exp322_runtime import _train_one_step
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset, validate_optimizer_invariants
from .exp325_contract import (
    AUTHORIZATION_FLAGS,
    FAMILIES,
    LEARNING_RATE,
    LOCAL_CHECKPOINTS,
    UPDATES_PER_WORLD,
    WORLD_INDICES,
    WORLD_KEYS,
    WorldSnapshot,
    canonical_json_bytes,
    reduce_world_isolation,
    validate_snapshot,
    world_floor_pass,
)
from .exp325_identity import (
    PARENT_EXECUTION_DIGEST,
    PARENT_FINAL_EVIDENCE_DIGEST,
    Exp325ExecutionIdentity,
    validate_execution_identity,
)


FAMILY_SCHEMA = "EXP325-FAMILY-WORLD-EVIDENCE-V1"
FINAL_SCHEMA = "EXP325-FINAL-EVIDENCE-V1"


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_parent_final(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP324-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-325 parent final schema mismatch")
    if payload.get("decision") != "NO_FAMILIES_ISOLATED_FIT":
        raise ValueError("EXP-325 parent disposition mismatch")
    if payload.get("evidence_digest") != PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-325 parent evidence digest mismatch")
    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    if _digest(materialized) != PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-325 parent canonical evidence mismatch")
    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping) or identity.get("exp324_execution_digest") != PARENT_EXECUTION_DIGEST:
        raise ValueError("EXP-325 parent execution identity mismatch")
    if payload.get("passed_families") != []:
        raise ValueError("EXP-325 parent passed-family set mismatch")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-325 parent authorization drift: {key}")


def family_worlds(family: str) -> tuple[object, ...]:
    if family not in FAMILIES:
        raise ValueError("unknown EXP-325 family")
    worlds = tuple(w for w in materialize_stage_a() if str(getattr(w, "family")) == family)
    if len(worlds) != 8:
        raise ValueError("EXP-325 family must contain exactly eight worlds")
    indices = tuple(int(getattr(w, "index")) for w in worlds)
    if indices != WORLD_INDICES:
        raise ValueError("EXP-325 family world indices drift")
    ids = tuple(str(getattr(w, "content_id")) for w in worlds)
    if len(set(ids)) != 8:
        raise ValueError("EXP-325 family content IDs must be unique")
    return worlds


def _world_record(
    *,
    family: str,
    world: object,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
) -> dict[str, Any]:
    state = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    compiled, optimizer = state.compiled, state.optimizer
    validate_optimizer_invariants(optimizer)

    index = int(getattr(world, "index"))
    content_id = str(getattr(world, "content_id"))
    snapshots: list[WorldSnapshot] = []
    nonfinite = 0
    last_grad = 0.0
    last_update = 0.0
    invalid_reason: str | None = None

    for local_index in range(UPDATES_PER_WORLD):
        global_step = 2048 + local_index
        _, last_grad, last_update, observed = _train_one_step(
            compiled, world, optimizer=optimizer, global_step=global_step
        )
        nonfinite += observed
        completed = local_index + 1
        if nonfinite:
            invalid_reason = "NONFINITE_EVENT"
            break
        if completed in LOCAL_CHECKPOINTS:
            metrics = _evaluate_subset(compiled, (world,))
            snapshots.append(
                WorldSnapshot(
                    family=family,
                    world_index=index,
                    local_update=completed,
                    teacher_forced_answer_token_accuracy=metrics["teacher_forced_answer_token_accuracy"],
                    teacher_forced_full_answer_exact=metrics["teacher_forced_full_answer_exact"],
                    greedy_exact=metrics["greedy_exact"],
                    answer_only_loss=metrics["answer_only_loss"],
                    gradient_norm_preclip=last_grad,
                    parameter_update_norm_ratio=last_update,
                    nonfinite_events=nonfinite,
                )
            )

    model = getattr(compiled, "model")
    return {
        "family": family,
        "world_index": index,
        "content_id": content_id,
        "snapshots": [asdict(x) for x in snapshots],
        "passed": any(world_floor_pass(x) for x in snapshots) if invalid_reason is None else False,
        "final_state": {
            "model_state_digest": model_state_digest(model),
            "optimizer_state_digest": optimizer_state_digest(optimizer),
            "rng_state_digest": rng_state_digest(),
        },
        "invalid_reason": invalid_reason,
    }


def _validate_world_record(record: Mapping[str, Any], expected_family: str) -> None:
    key = (record.get("family"), record.get("world_index"))
    if key not in WORLD_KEYS or record.get("family") != expected_family:
        raise ValueError("EXP-325 world record identity mismatch")
    content_id = record.get("content_id")
    if not isinstance(content_id, str) or not content_id.startswith("exp319:"):
        raise ValueError("EXP-325 content ID malformed")
    rows = tuple(WorldSnapshot(**x) for x in record.get("snapshots", ()))
    if record.get("invalid_reason") is None:
        if tuple(x.local_update for x in rows) != LOCAL_CHECKPOINTS:
            raise ValueError("EXP-325 world checkpoint geometry mismatch")
    for row in rows:
        validate_snapshot(row)
        if (row.family, row.world_index) != key:
            raise ValueError("EXP-325 snapshot identity mismatch")
    expected_pass = any(world_floor_pass(x) for x in rows) if record.get("invalid_reason") is None else False
    if record.get("passed") is not expected_pass:
        raise ValueError("EXP-325 world pass flag mismatch")
    final_state = record.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("EXP-325 final state missing")
    for field in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
        value = final_state.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"EXP-325 malformed final digest: {field}")


def build_family_evidence(
    *,
    family: str,
    identity: Exp325ExecutionIdentity,
    reconstruction: Mapping[str, Any],
    worlds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": FAMILY_SCHEMA,
        "family": family,
        "learning_rate": LEARNING_RATE,
        "updates_per_world": UPDATES_PER_WORLD,
        "execution_identity": asdict(identity),
        "reconstruction": dict(reconstruction),
        "worlds": [dict(x) for x in worlds],
        **AUTHORIZATION_FLAGS,
    }
    validate_family_evidence(payload, with_digest=False)
    payload["family_evidence_digest"] = _digest(payload)
    return payload


def validate_family_evidence(payload: Mapping[str, Any], *, with_digest: bool = True) -> None:
    family = payload.get("family")
    if payload.get("schema") != FAMILY_SCHEMA or family not in FAMILIES:
        raise ValueError("EXP-325 family evidence schema/family mismatch")
    if payload.get("learning_rate") != LEARNING_RATE or payload.get("updates_per_world") != UPDATES_PER_WORLD:
        raise ValueError("EXP-325 family evidence geometry mismatch")
    raw = payload.get("execution_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-325 execution identity missing")
    validate_execution_identity(Exp325ExecutionIdentity(**raw))
    worlds = payload.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != 8:
        raise ValueError("EXP-325 family evidence requires eight worlds")
    seen = set()
    for record in worlds:
        if not isinstance(record, Mapping):
            raise ValueError("EXP-325 world record must be object")
        _validate_world_record(record, str(family))
        key = (str(record["family"]), int(record["world_index"]))
        if key in seen:
            raise ValueError("duplicate EXP-325 world record")
        seen.add(key)
    if seen != {(str(family), i) for i in WORLD_INDICES}:
        raise ValueError("EXP-325 family world coverage mismatch")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-325 authorization drift: {key}")
    if with_digest:
        materialized = dict(payload)
        claimed = materialized.pop("family_evidence_digest", None)
        if not isinstance(claimed, str) or _digest(materialized) != claimed:
            raise ValueError("EXP-325 family evidence digest mismatch")


def run_family_worlds(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
    parent_final_path: str | Path,
    family: str,
    execution_identity: Exp325ExecutionIdentity,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    parent = json.loads(Path(parent_final_path).read_text(encoding="utf-8"))
    if not isinstance(parent, Mapping):
        raise ValueError("EXP-325 parent final must be object")
    validate_parent_final(parent)

    # One validated load fixes the reconstruction receipt included in family evidence.
    authority = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    reconstruction = dict(authority.reconstruction)
    del authority

    records = tuple(
        _world_record(
            family=family,
            world=world,
            checkpoint_path=checkpoint_path,
            receipt_path=receipt_path,
            selection_lock_path=selection_lock_path,
        )
        for world in family_worlds(family)
    )
    return build_family_evidence(
        family=family,
        identity=execution_identity,
        reconstruction=reconstruction,
        worlds=records,
    )


def build_final_evidence(family_arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Mapping[str, Any]] = {}
    for arm in family_arms:
        validate_family_evidence(arm)
        family = str(arm["family"])
        if family in by_family:
            raise ValueError("duplicate EXP-325 family evidence")
        by_family[family] = arm
    if set(by_family) != set(FAMILIES):
        raise ValueError("EXP-325 requires all four family evidence objects")

    identities = {canonical_json_bytes(x["execution_identity"]) for x in by_family.values()}
    reconstructions = {canonical_json_bytes(x["reconstruction"]) for x in by_family.values()}
    if len(identities) != 1 or len(reconstructions) != 1:
        raise ValueError("EXP-325 family authority mismatch")

    records: dict[tuple[str, int], tuple[WorldSnapshot, ...]] = {}
    invalid = False
    content_ids: dict[str, str] = {}
    for family in FAMILIES:
        for record in by_family[family]["worlds"]:
            key = (family, int(record["world_index"]))
            rows = tuple(WorldSnapshot(**x) for x in record["snapshots"])
            records[key] = rows
            content_ids[f"{family}:{key[1]}"] = str(record["content_id"])
            invalid = invalid or record.get("invalid_reason") is not None

    if invalid:
        decision, passed = "INVALID_WORLD_ISOLATION", ()
    else:
        decision, passed = reduce_world_isolation(records)

    passed_set = set(passed)
    family_pass_counts = {
        family: sum((family, i) in passed_set for i in WORLD_INDICES)
        for family in FAMILIES
    }
    failed = [
        {"family": family, "world_index": i, "content_id": content_ids[f"{family}:{i}"]}
        for family, i in WORLD_KEYS
        if (family, i) not in passed_set
    ]
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": by_family[FAMILIES[0]]["execution_identity"],
        "reconstruction": by_family[FAMILIES[0]]["reconstruction"],
        "decision": decision,
        "passed_world_count": len(passed),
        "passed_worlds": [
            {"family": family, "world_index": i, "content_id": content_ids[f"{family}:{i}"]}
            for family, i in passed
        ],
        "failed_worlds": failed,
        "family_pass_counts": family_pass_counts,
        "family_evidence_digests": {
            family: by_family[family]["family_evidence_digest"] for family in FAMILIES
        },
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    return payload
