from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .exp301_scientific import build_scientific_arm
from .exp319_metrics import (
    global_gradient_norm,
    nonfinite_report,
    parameter_update_norm_ratio,
    snapshot_trainable_parameters,
)
from .exp319_training import model_state_digest, optimizer_state_digest, rng_state_digest
from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp324_runtime import _evaluate_subset, validate_optimizer_invariants
from .exp325_runtime import family_worlds
from .exp326_runtime import _train_one_step_with_effort
from .exp331_runtime import _backward_grads
from .exp334_runtime import validate_final_evidence as validate_exp334_final_evidence
from .exp335_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FAMILIES,
    GRADIENT_CLIP_NORM,
    PARENT_EXP334_EVIDENCE_DIGEST,
    PARENT_EXP334_EXECUTION_DIGEST,
    PARENT_EXP334_JSON_SHA256,
    PINV_RTOL,
    RECONSTRUCTION_CHECKPOINT_SHA256,
    RECONSTRUCTION_RECEIPT_SHA256,
    RECONSTRUCTION_ZIP_DIGEST,
    TARGET_NORM_SQUARED_FLOOR,
    UPDATES_PER_CHUNK,
    WORLD_IDS,
    BoundaryResult,
    canonical_digest,
    canonical_json_bytes,
    chunk_schedule,
    data_order_digest,
    reduce_full32,
    sham_equivalent,
    validate_boundary,
)
from .exp335_identity import Exp335ExecutionIdentity, validate_execution_identity
from nolane_ai.protocol.identity import source_tree_digest

CHUNK_CHECKPOINT_SCHEMA = "EXP335-CONTINUATION-CHECKPOINT-V1"
CHUNK_RECEIPT_SCHEMA = "EXP335-CONTINUATION-RECEIPT-V1"
CHUNK_SUMMARY_SCHEMA = "EXP335-CHUNK-SUMMARY-V1"
FINAL_SCHEMA = "EXP335-FINAL-EVIDENCE-V1"
EXPECTED_PARAMETER_COUNT = 10_000_000


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXP-335 JSON object required: {path}")
    return value


def _hex_digest(value: object) -> bool:
    if not isinstance(value, str):
        return False
    raw = value[7:] if value.startswith("sha256:") else value
    if len(raw) != 64:
        return False
    try:
        int(raw, 16)
    except ValueError:
        return False
    return True


def _world_map() -> dict[str, object]:
    worlds: dict[str, object] = {}
    for family in FAMILIES:
        rows = family_worlds(family)
        for index, world in enumerate(rows):
            if int(getattr(world, "index")) != index:
                raise ValueError("EXP-335 world index drift")
            key = f"{family}:{index}"
            if key in worlds:
                raise ValueError("EXP-335 duplicate world")
            worlds[key] = world
    if tuple(worlds) != WORLD_IDS:
        raise ValueError("EXP-335 world population/order drift")
    return worlds


def validate_exp334_parent(payload: Mapping[str, Any]) -> None:
    validate_exp334_final_evidence(payload)
    if payload.get("decision") != "HIGHER_ORDER_SUBSPACE_RESCUE_NO_REGRESSION":
        raise ValueError("EXP-335 EXP-334 disposition")
    if payload.get("evidence_digest") != PARENT_EXP334_EVIDENCE_DIGEST:
        raise ValueError("EXP-335 EXP-334 evidence digest")
    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping) or identity.get("exp334_execution_digest") != PARENT_EXP334_EXECUTION_DIGEST:
        raise ValueError("EXP-335 EXP-334 execution digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-335 parent authorization drift: {key}")


def validate_parent_exp334_file(path: str | Path) -> dict[str, Any]:
    if _file_sha256(path) != PARENT_EXP334_JSON_SHA256:
        raise ValueError("EXP-335 EXP-334 final JSON SHA")
    payload = _read_json(path)
    validate_exp334_parent(payload)
    return payload


def _new_compiled_optimizer() -> tuple[object, torch.optim.Optimizer]:
    torch.manual_seed(0)
    compiled = build_scientific_arm("A_FIXED", device="cpu")
    model = getattr(compiled, "model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("EXP-335 resident parameter count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    return compiled, optimizer


def _state_digests(compiled: object, optimizer: torch.optim.Optimizer) -> dict[str, str]:
    model = getattr(compiled, "model")
    return {
        "model_state_digest": model_state_digest(model),
        "optimizer_state_digest": optimizer_state_digest(optimizer),
        "rng_state_digest": rng_state_digest(),
    }


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    materialized = dict(receipt)
    materialized.pop("receipt_digest", None)
    return canonical_digest(materialized)


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    chunk_index: int,
    checkpoint_path: str | Path,
    identity: Exp335ExecutionIdentity,
) -> None:
    if receipt.get("schema") != CHUNK_RECEIPT_SCHEMA:
        raise ValueError("EXP-335 receipt schema")
    if receipt.get("arm") != arm or receipt.get("chunk_index") != chunk_index:
        raise ValueError("EXP-335 receipt chain position")
    expected_exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    if receipt.get("cumulative_exposure_per_world") != expected_exposure:
        raise ValueError("EXP-335 receipt exposure")
    if receipt.get("cumulative_source_updates") != expected_exposure * len(WORLD_IDS):
        raise ValueError("EXP-335 receipt updates")
    if receipt.get("data_order_digest") != data_order_digest(expected_exposure):
        raise ValueError("EXP-335 receipt data order")
    if receipt.get("source_tree_digest") != identity.source_tree_digest:
        raise ValueError("EXP-335 receipt source tree")
    if receipt.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-335 receipt preregistration")
    if not _hex_digest(receipt.get("parent_artifact_digest")):
        raise ValueError("EXP-335 receipt parent artifact digest")
    if receipt.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("EXP-335 receipt checkpoint sha")
    for key in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
        if not _hex_digest(receipt.get(key)):
            raise ValueError(f"EXP-335 receipt digest: {key}")
    if receipt.get("receipt_digest") != _receipt_digest(receipt):
        raise ValueError("EXP-335 receipt digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"EXP-335 receipt authorization drift: {key}")


def _load_continuation(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    arm: str,
    previous_chunk_index: int,
    identity: Exp335ExecutionIdentity,
) -> tuple[object, torch.optim.Optimizer, dict[str, Any]]:
    receipt = _read_json(receipt_path)
    _validate_receipt(
        receipt,
        arm=arm,
        chunk_index=previous_chunk_index,
        checkpoint_path=checkpoint_path,
        identity=identity,
    )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or raw.get("schema") != CHUNK_CHECKPOINT_SCHEMA:
        raise ValueError("EXP-335 continuation checkpoint schema")
    if raw.get("arm") != arm or raw.get("chunk_index") != previous_chunk_index:
        raise ValueError("EXP-335 continuation checkpoint chain position")
    compiled, optimizer = _new_compiled_optimizer()
    model = getattr(compiled, "model")
    model.load_state_dict(raw["model_state_dict"])
    optimizer.load_state_dict(raw["optimizer_state_dict"])
    torch.set_rng_state(raw["torch_rng_state"])
    validate_optimizer_invariants(optimizer)
    observed = _state_digests(compiled, optimizer)
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise ValueError(f"EXP-335 continuation state mismatch: {key}")
    return compiled, optimizer, receipt


def _dot(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("EXP-335 gradient geometry")
    total = 0.0
    for x, y in zip(a, b):
        total += float(torch.sum(x * y, dtype=torch.float64).item())
    if not math.isfinite(total):
        raise ValueError("EXP-335 nonfinite gradient dot")
    return total


def _norm_sq(a: tuple[torch.Tensor, ...]) -> float:
    return _dot(a, a)


def _project_source(
    source: tuple[torch.Tensor, ...],
    targets: list[tuple[str, tuple[torch.Tensor, ...]]],
) -> tuple[tuple[torch.Tensor, ...], list[str], dict[str, float], dict[str, float]]:
    raw = {world_id: _dot(source, grad) for world_id, grad in targets}
    selected: list[tuple[str, tuple[torch.Tensor, ...]]] = []
    selected_norm_sq: list[float] = []
    for world_id, grad in targets:
        if raw[world_id] >= 0.0:
            continue
        norm_sq = _norm_sq(grad)
        if norm_sq > TARGET_NORM_SQUARED_FLOOR:
            selected.append((world_id, grad))
            selected_norm_sq.append(norm_sq)
    if not selected:
        cloned = tuple(x.clone() for x in source)
        return cloned, [], raw, raw.copy()

    count = len(selected)
    gram = torch.empty((count, count), dtype=torch.float64)
    rhs = torch.empty((count,), dtype=torch.float64)
    for i, (world_id, gi) in enumerate(selected):
        rhs[i] = raw[world_id]
        gram[i, i] = selected_norm_sq[i]
        for j in range(i + 1, count):
            gj = selected[j][1]
            value = _dot(gi, gj)
            gram[i, j] = value
            gram[j, i] = value
    alpha = torch.linalg.pinv(gram, rtol=PINV_RTOL) @ rhs

    result: list[torch.Tensor] = []
    for parameter_index, source_tensor in enumerate(source):
        value = source_tensor.clone()
        for coefficient, (_, target) in zip(alpha, selected):
            value = value - target[parameter_index] * float(coefficient.item())
        result.append(value)
    projected = tuple(result)
    post = {world_id: _dot(projected, grad) for world_id, grad in targets}
    return projected, [world_id for world_id, _ in selected], raw, post


def _measured_step(
    compiled: object,
    source_world: object,
    target_worlds: list[tuple[str, object]],
    *,
    optimizer: torch.optim.Optimizer,
    effort: int,
    project: bool,
    exposure_index: int,
    source_world_id: str,
) -> dict[str, Any]:
    if len(target_worlds) != 31 or source_world_id in {key for key, _ in target_worlds}:
        raise ValueError("EXP-335 requires all other 31 targets")
    model = getattr(compiled, "model")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    before = snapshot_trainable_parameters(model.parameters())
    model.train()
    rng_start = torch.get_rng_state().clone()

    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_start)
    source_loss, source_grads, source_nonfinite = _backward_grads(compiled, source_world, effort)
    rng_after_source = torch.get_rng_state().clone()

    targets: list[tuple[str, tuple[torch.Tensor, ...]]] = []
    target_nonfinite = 0
    target_losses: dict[str, float] = {}
    for target_id, target_world in target_worlds:
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_start)
        target_loss, target_grads, observed = _backward_grads(compiled, target_world, effort)
        targets.append((target_id, target_grads))
        target_losses[target_id] = target_loss
        target_nonfinite += observed

    torch.set_rng_state(rng_after_source)
    optimizer.zero_grad(set_to_none=True)
    if project:
        projected, selected, raw_dots, post_dots = _project_source(source_grads, targets)
        applied = projected
    else:
        raw_dots = {world_id: _dot(source_grads, grad) for world_id, grad in targets}
        selected = []
        post_dots = {}
        applied = source_grads
    for parameter, gradient in zip(params, applied):
        parameter.grad = gradient.clone()

    preclip = global_gradient_norm(model.parameters())
    before_report = nonfinite_report(model.parameters(), loss=source_loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    after = snapshot_trainable_parameters(model.parameters())
    ratio = parameter_update_norm_ratio(before, after)
    after_report = nonfinite_report(model.parameters(), loss=source_loss)
    observed = source_nonfinite + target_nonfinite + before_report.total + after_report.total

    negative = [world_id for world_id, dot in raw_dots.items() if dot < 0.0]
    return {
        "exposure_index": exposure_index,
        "source_world_id": source_world_id,
        "target_world_ids": [world_id for world_id, _ in target_worlds],
        "effort": effort,
        "source_loss": source_loss,
        "target_losses": target_losses,
        "negative_targets": negative,
        "projected_targets": selected if project else [],
        "raw_source_target_dots": raw_dots,
        "post_projection_source_target_dots": post_dots if project else {},
        "gradient_norm_preclip": preclip,
        "parameter_update_norm_ratio": ratio,
        "nonfinite_events": observed,
    }


def _evaluate_worlds(compiled: object, worlds: Mapping[str, object]) -> dict[str, list[float]]:
    rng = torch.get_rng_state().clone()
    try:
        rows = [_evaluate_subset(compiled, (worlds[world_id],)) for world_id in WORLD_IDS]
    finally:
        torch.set_rng_state(rng)
    return {
        "teacher_forced_answer_token_accuracy": [
            float(row["teacher_forced_answer_token_accuracy"]) for row in rows
        ],
        "teacher_forced_full_answer_exact": [
            float(row["teacher_forced_full_answer_exact"]) for row in rows
        ],
        "greedy_exact": [float(row["greedy_exact"]) for row in rows],
        "answer_only_loss": [float(row["answer_only_loss"]) for row in rows],
    }


def _boundary(
    *,
    arm: str,
    chunk_index: int,
    compiled: object,
    optimizer: torch.optim.Optimizer,
    metrics: Mapping[str, list[float]],
    cumulative_nonfinite: int,
    cumulative_negative: int,
    cumulative_projection_updates: int,
    cumulative_projected_targets: int,
) -> BoundaryResult:
    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    digests = _state_digests(compiled, optimizer)
    result = BoundaryResult(
        arm=arm,
        chunk_index=chunk_index,
        cumulative_exposure_per_world=exposure,
        cumulative_source_updates=exposure * len(WORLD_IDS),
        world_token_accuracies=tuple(metrics["teacher_forced_answer_token_accuracy"]),
        world_full_answer_exact=tuple(metrics["teacher_forced_full_answer_exact"]),
        model_state_digest=digests["model_state_digest"],
        optimizer_state_digest=digests["optimizer_state_digest"],
        rng_state_digest=digests["rng_state_digest"],
        nonfinite_events=cumulative_nonfinite,
        negative_target_count=cumulative_negative,
        projection_update_count=cumulative_projection_updates,
        projected_target_count=cumulative_projected_targets,
    )
    validate_boundary(result)
    return result


def _write_checkpoint_and_receipt(
    *,
    output_dir: str | Path,
    arm: str,
    chunk_index: int,
    compiled: object,
    optimizer: torch.optim.Optimizer,
    boundary: BoundaryResult,
    parent_artifact_digest: str,
    identity: Exp335ExecutionIdentity,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = {
        "CONTROL_FULL32": "control",
        "SHAM_MEASURE_FULL32": "sham",
        "SUBSPACE_PROJECT_FULL32": "project",
    }[arm]
    checkpoint_path = output / f"{stem}.pt"
    receipt_path = output / f"{stem}-receipt.json"
    if checkpoint_path.exists() or receipt_path.exists():
        raise FileExistsError("EXP-335 output already exists")
    model = getattr(compiled, "model")
    payload = {
        "schema": CHUNK_CHECKPOINT_SCHEMA,
        "arm": arm,
        "chunk_index": chunk_index,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state().clone(),
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(payload, handle)
    checkpoint_sha = _file_sha256(checkpoint_path)
    receipt = {
        "schema": CHUNK_RECEIPT_SCHEMA,
        "arm": arm,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": boundary.cumulative_exposure_per_world,
        "cumulative_source_updates": boundary.cumulative_source_updates,
        "model_state_digest": boundary.model_state_digest,
        "optimizer_state_digest": boundary.optimizer_state_digest,
        "rng_state_digest": boundary.rng_state_digest,
        "data_order_digest": data_order_digest(boundary.cumulative_exposure_per_world),
        "parent_artifact_digest": parent_artifact_digest,
        "source_tree_digest": identity.source_tree_digest,
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "checkpoint_sha256": checkpoint_sha,
        **AUTHORIZATION_FLAGS,
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "receipt": receipt_path.name,
        "receipt_digest": receipt["receipt_digest"],
    }


def _initial_arm(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    if _file_sha256(checkpoint_path) != RECONSTRUCTION_CHECKPOINT_SHA256:
        raise ValueError("EXP-335 reconstruction checkpoint SHA")
    if _file_sha256(receipt_path) != RECONSTRUCTION_RECEIPT_SHA256:
        raise ValueError("EXP-335 reconstruction receipt SHA")
    state = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    validate_optimizer_invariants(state.optimizer)
    return state.compiled, state.optimizer, {
        "nonfinite_events": 0,
        "negative_target_count": 0,
        "projection_update_count": 0,
        "projected_target_count": 0,
    }


def _continuation_arm(
    *,
    previous_dir: str | Path,
    arm: str,
    chunk_index: int,
    identity: Exp335ExecutionIdentity,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    stem = {
        "CONTROL_FULL32": "control",
        "SHAM_MEASURE_FULL32": "sham",
        "SUBSPACE_PROJECT_FULL32": "project",
    }[arm]
    previous = Path(previous_dir)
    compiled, optimizer, _ = _load_continuation(
        checkpoint_path=previous / f"{stem}.pt",
        receipt_path=previous / f"{stem}-receipt.json",
        arm=arm,
        previous_chunk_index=chunk_index - 1,
        identity=identity,
    )
    summary = validate_chunk_summary(previous / "chunk-summary.json", identity=identity)
    row = summary["arms"][arm]
    return compiled, optimizer, {
        "nonfinite_events": int(row["boundary"]["nonfinite_events"]),
        "negative_target_count": int(row["boundary"]["negative_target_count"]),
        "projection_update_count": int(row["boundary"]["projection_update_count"]),
        "projected_target_count": int(row["boundary"]["projected_target_count"]),
    }


def _run_arm_chunk(
    *,
    arm: str,
    chunk_index: int,
    compiled: object,
    optimizer: torch.optim.Optimizer,
    counters: dict[str, int],
    worlds: Mapping[str, object],
) -> tuple[BoundaryResult, dict[str, list[float]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    for family, index, exposure, effort in chunk_schedule(chunk_index):
        source_id = f"{family}:{index}"
        source_world = worlds[source_id]
        if arm == "CONTROL_FULL32":
            _, _, _, observed = _train_one_step_with_effort(
                compiled, source_world, optimizer=optimizer, effort=effort
            )
            counters["nonfinite_events"] += int(observed)
        else:
            target_worlds = [(target_id, worlds[target_id]) for target_id in WORLD_IDS if target_id != source_id]
            row = _measured_step(
                compiled,
                source_world,
                target_worlds,
                optimizer=optimizer,
                effort=effort,
                project=arm == "SUBSPACE_PROJECT_FULL32",
                exposure_index=exposure,
                source_world_id=source_id,
            )
            measurements.append(row)
            counters["nonfinite_events"] += int(row["nonfinite_events"])
            counters["negative_target_count"] += len(row["negative_targets"])
            if arm == "SUBSPACE_PROJECT_FULL32":
                counters["projection_update_count"] += int(bool(row["projected_targets"]))
                counters["projected_target_count"] += len(row["projected_targets"])
        if counters["nonfinite_events"]:
            raise ValueError(f"EXP-335 nonfinite event in {arm} chunk {chunk_index}")

    metrics = _evaluate_worlds(compiled, worlds)
    boundary = _boundary(
        arm=arm,
        chunk_index=chunk_index,
        compiled=compiled,
        optimizer=optimizer,
        metrics=metrics,
        cumulative_nonfinite=counters["nonfinite_events"],
        cumulative_negative=counters["negative_target_count"],
        cumulative_projection_updates=counters["projection_update_count"],
        cumulative_projected_targets=counters["projected_target_count"],
    )
    return boundary, metrics, measurements


def validate_chunk_summary(
    path: str | Path,
    *,
    identity: Exp335ExecutionIdentity,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != CHUNK_SUMMARY_SCHEMA:
        raise ValueError("EXP-335 chunk summary schema")
    if payload.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-335 chunk preregistration")
    if payload.get("source_tree_digest") != identity.source_tree_digest:
        raise ValueError("EXP-335 chunk source tree")
    if payload.get("execution_digest") != identity.exp335_execution_digest:
        raise ValueError("EXP-335 chunk execution identity")
    chunk_index = payload.get("chunk_index")
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-335 chunk summary index")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("EXP-335 chunk arm coverage")
    boundaries = {}
    for arm in ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping):
            raise ValueError("EXP-335 chunk arm record")
        boundary = BoundaryResult(**row["boundary"])
        validate_boundary(boundary)
        if boundary.arm != arm or boundary.chunk_index != chunk_index:
            raise ValueError("EXP-335 chunk boundary identity")
        boundaries[arm] = boundary
    if not sham_equivalent(boundaries["CONTROL_FULL32"], boundaries["SHAM_MEASURE_FULL32"]):
        raise ValueError("EXP-335 CONTROL/SHAM same-process mismatch")
    materialized = dict(payload)
    claimed = materialized.pop("bundle_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-335 chunk bundle digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-335 chunk authorization drift: {key}")
    return payload


def _arm_state(
    *,
    arm: str,
    chunk_index: int,
    execution_identity: Exp335ExecutionIdentity,
    reconstruction_checkpoint: str | Path | None,
    reconstruction_receipt: str | Path | None,
    selection_lock: str | Path | None,
    previous_dir: str | Path | None,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    if chunk_index == 0:
        if reconstruction_checkpoint is None or reconstruction_receipt is None or selection_lock is None:
            raise ValueError("EXP-335 initial authority inputs")
        return _initial_arm(
            checkpoint_path=reconstruction_checkpoint,
            receipt_path=reconstruction_receipt,
            selection_lock_path=selection_lock,
        )
    if previous_dir is None:
        raise ValueError("EXP-335 continuation requires previous chunk")
    return _continuation_arm(
        previous_dir=previous_dir,
        arm=arm,
        chunk_index=chunk_index,
        identity=execution_identity,
    )


def run_chunk(
    *,
    chunk_index: int,
    execution_identity: Exp335ExecutionIdentity,
    code_root: str | Path,
    parent_artifact_digest: str,
    output_dir: str | Path,
    reconstruction_checkpoint: str | Path | None = None,
    reconstruction_receipt: str | Path | None = None,
    selection_lock: str | Path | None = None,
    parent_exp334: str | Path | None = None,
    previous_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    if source_tree_digest(code_root) != execution_identity.source_tree_digest:
        raise ValueError("EXP-335 source-tree digest mismatch")
    if not _hex_digest(parent_artifact_digest):
        raise ValueError("EXP-335 parent artifact digest malformed")
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-335 chunk index")

    if chunk_index == 0:
        if parent_exp334 is None:
            raise ValueError("EXP-335 chunk 0 parent EXP-334")
        if parent_artifact_digest != RECONSTRUCTION_ZIP_DIGEST:
            raise ValueError("EXP-335 chunk 0 parent artifact")
        validate_parent_exp334_file(parent_exp334)
    else:
        if previous_dir is None:
            raise ValueError("EXP-335 continuation requires previous chunk")
        previous = validate_chunk_summary(Path(previous_dir) / "chunk-summary.json", identity=execution_identity)
        if previous["chunk_index"] != chunk_index - 1:
            raise ValueError("EXP-335 duplicate/skipped chain position")

    worlds = _world_map()
    records: dict[str, Any] = {}
    boundaries: dict[str, BoundaryResult] = {}
    for arm in ARMS:
        compiled, optimizer, counters = _arm_state(
            arm=arm,
            chunk_index=chunk_index,
            execution_identity=execution_identity,
            reconstruction_checkpoint=reconstruction_checkpoint,
            reconstruction_receipt=reconstruction_receipt,
            selection_lock=selection_lock,
            previous_dir=previous_dir,
        )
        boundary, metrics, measurements = _run_arm_chunk(
            arm=arm,
            chunk_index=chunk_index,
            compiled=compiled,
            optimizer=optimizer,
            counters=counters,
            worlds=worlds,
        )
        files = _write_checkpoint_and_receipt(
            output_dir=output_dir,
            arm=arm,
            chunk_index=chunk_index,
            compiled=compiled,
            optimizer=optimizer,
            boundary=boundary,
            parent_artifact_digest=parent_artifact_digest,
            identity=execution_identity,
        )
        records[arm] = {
            "boundary": asdict(boundary),
            "metrics": metrics,
            "measurements": measurements,
            "files": files,
        }
        boundaries[arm] = boundary
        del compiled, optimizer
        if arm == "SHAM_MEASURE_FULL32" and not sham_equivalent(
            boundaries["CONTROL_FULL32"], boundaries["SHAM_MEASURE_FULL32"]
        ):
            raise ValueError("SHAM_FULL32_MISMATCH")

    summary = {
        "schema": CHUNK_SUMMARY_SCHEMA,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": (chunk_index + 1) * EXPOSURES_PER_CHUNK,
        "cumulative_source_updates": (chunk_index + 1) * UPDATES_PER_CHUNK,
        "parent_artifact_digest": parent_artifact_digest,
        "source_tree_digest": execution_identity.source_tree_digest,
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "execution_digest": execution_identity.exp335_execution_digest,
        "control_sham_exact": True,
        "arms": records,
        **AUTHORIZATION_FLAGS,
    }
    summary["bundle_digest"] = canonical_digest(summary)
    path = Path(output_dir) / "chunk-summary.json"
    path.write_bytes(canonical_json_bytes(summary) + b"\n")
    validate_chunk_summary(path, identity=execution_identity)
    return summary


def _boundary_from_summary(summary: Mapping[str, Any], arm: str) -> BoundaryResult:
    return BoundaryResult(**summary["arms"][arm]["boundary"])


def build_final_evidence(
    *,
    chunk_dirs: list[str | Path],
    chunk_artifact_digests: list[str],
    execution_identity: Exp335ExecutionIdentity,
    parent_exp334_path: str | Path,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    validate_parent_exp334_file(parent_exp334_path)
    if len(chunk_dirs) != CHUNK_COUNT or len(chunk_artifact_digests) != CHUNK_COUNT:
        raise ValueError("EXP-335 finalizer requires exactly 8 chunks and 8 artifact digests")
    if not all(_hex_digest(value) for value in chunk_artifact_digests):
        raise ValueError("EXP-335 malformed chunk artifact digest")
    summaries = [
        validate_chunk_summary(Path(path) / "chunk-summary.json", identity=execution_identity)
        for path in chunk_dirs
    ]
    if [row["chunk_index"] for row in summaries] != list(range(CHUNK_COUNT)):
        raise ValueError("EXP-335 chunk chain order")
    if summaries[0]["parent_artifact_digest"] != RECONSTRUCTION_ZIP_DIGEST:
        raise ValueError("EXP-335 root parent artifact chain")
    for index in range(1, CHUNK_COUNT):
        if summaries[index]["parent_artifact_digest"] != chunk_artifact_digests[index - 1]:
            raise ValueError(f"EXP-335 artifact chain mismatch at chunk {index}")

    final = summaries[-1]
    control = _boundary_from_summary(final, "CONTROL_FULL32")
    sham = _boundary_from_summary(final, "SHAM_MEASURE_FULL32")
    project = _boundary_from_summary(final, "SUBSPACE_PROJECT_FULL32")
    decision, vectors = reduce_full32(
        control, sham, project, parent_authority_valid=True, invalid=False
    )
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": asdict(execution_identity),
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "parent_exp334_evidence_digest": PARENT_EXP334_EVIDENCE_DIGEST,
        "reconstruction": expected_reconstruction_payload(),
        "chunk_bundle_digests": [row["bundle_digest"] for row in summaries],
        "chunk_artifact_digests": list(chunk_artifact_digests),
        "chunk_parent_artifact_digests": [row["parent_artifact_digest"] for row in summaries],
        "control_sham_exact_by_chunk": [bool(row["control_sham_exact"]) for row in summaries],
        "final_boundaries": {
            "CONTROL_FULL32": asdict(control),
            "SHAM_MEASURE_FULL32": asdict(sham),
            "SUBSPACE_PROJECT_FULL32": asdict(project),
        },
        "decision": decision,
        **vectors,
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = canonical_digest(payload)
    validate_final_evidence(payload)
    return payload


def validate_final_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != FINAL_SCHEMA:
        raise ValueError("EXP-335 final schema")
    raw_identity = payload.get("execution_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("EXP-335 final identity")
    identity = Exp335ExecutionIdentity(**raw_identity)
    validate_execution_identity(identity)
    if payload.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-335 final preregistration")
    if payload.get("parent_exp334_evidence_digest") != PARENT_EXP334_EVIDENCE_DIGEST:
        raise ValueError("EXP-335 final parent")
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-335 final reconstruction")
    if payload.get("control_sham_exact_by_chunk") != [True] * CHUNK_COUNT:
        raise ValueError("EXP-335 final sham chain")
    if not isinstance(payload.get("chunk_artifact_digests"), list) or len(payload["chunk_artifact_digests"]) != CHUNK_COUNT:
        raise ValueError("EXP-335 final artifact chain")
    if not all(_hex_digest(value) for value in payload["chunk_artifact_digests"]):
        raise ValueError("EXP-335 final artifact digest")
    raw_boundaries = payload.get("final_boundaries")
    if not isinstance(raw_boundaries, Mapping) or set(raw_boundaries) != set(ARMS):
        raise ValueError("EXP-335 final boundary coverage")
    control = BoundaryResult(**raw_boundaries["CONTROL_FULL32"])
    sham = BoundaryResult(**raw_boundaries["SHAM_MEASURE_FULL32"])
    project = BoundaryResult(**raw_boundaries["SUBSPACE_PROJECT_FULL32"])
    decision, vectors = reduce_full32(control, sham, project, parent_authority_valid=True)
    if payload.get("decision") != decision:
        raise ValueError("EXP-335 final reducer")
    for key, expected in vectors.items():
        if payload.get(key) != expected:
            raise ValueError(f"EXP-335 final reducer vector: {key}")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-335 final authorization drift: {key}")
    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-335 final evidence digest")
