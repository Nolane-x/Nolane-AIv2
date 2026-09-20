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
from .exp319_training import (
    evaluate_snapshot,
    load_checkpoint_bundle,
    model_state_digest,
    optimizer_state_digest,
    rng_state_digest,
)
from .exp319_worlds import materialize_stage_a
from .exp324_runtime import _evaluate_subset
from .exp326_runtime import _train_one_step_with_effort
from .exp331_runtime import _backward_grads
from .exp336_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FAMILIES,
    FINAL_CUMULATIVE_STEP,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS,
    PARENT_CNRS_ARTIFACT_NAME,
    PARENT_CNRS_CHECKPOINT_SHA256,
    PARENT_CNRS_MODEL_STATE_DIGEST,
    PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
    PARENT_CNRS_RECEIPT_ARTIFACT_DIGEST,
    PARENT_CNRS_RECEIPT_SHA256,
    PARENT_CNRS_RNG_STATE_DIGEST,
    PARENT_CNRS_RUN_IDENTITY,
    PARENT_CNRS_SELECTION_DIGEST,
    PARENT_CNRS_SOURCE_SHA,
    PARENT_CNRS_SUMMARY_SHA256,
    PARENT_CNRS_ZIP_DIGEST,
    PARENT_SELECTION_AUTHORITY_DIGEST,
    PARENT_SELECTION_JSON_SHA256,
    PINV_RTOL,
    STARTING_CUMULATIVE_STEP,
    TARGET_NORM_SQUARED_FLOOR,
    UPDATES_PER_CHUNK,
    WEIGHT_DECAY,
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
from .exp336_identity import Exp336ExecutionIdentity, validate_execution_identity
from nolane_ai.protocol.identity import source_tree_digest

CHUNK_CHECKPOINT_SCHEMA = "EXP336-CNRS-CONTINUATION-CHECKPOINT-V1"
CHUNK_RECEIPT_SCHEMA = "EXP336-CNRS-CONTINUATION-RECEIPT-V1"
CHUNK_SUMMARY_SCHEMA = "EXP336-CNRS-CHUNK-SUMMARY-V1"
FINAL_SCHEMA = "EXP336-CNRS-FULL32-STAGE-A-FINAL-EVIDENCE-V1"
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
        raise ValueError(f"EXP-336 JSON object required: {path}")
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
    for world in materialize_stage_a():
        family = str(getattr(world, "family"))
        index = int(getattr(world, "index"))
        if family not in FAMILIES or index not in range(8):
            raise ValueError("EXP-336 world identity drift")
        key = f"{family}:{index}"
        if key in worlds:
            raise ValueError("EXP-336 duplicate world")
        worlds[key] = world
    if tuple(worlds) != WORLD_IDS:
        raise ValueError("EXP-336 world population/order drift")
    return worlds


def validate_parent_selection_file(path: str | Path) -> dict[str, Any]:
    if _file_sha256(path) != PARENT_SELECTION_JSON_SHA256:
        raise ValueError("EXP-336 parent selection JSON SHA")
    payload = _read_json(path)
    if payload.get("schema") != "EXP319-STAGE-A-SELECTION-AUTHORITY-V1":
        raise ValueError("EXP-336 parent selection schema")
    if payload.get("authority_digest") != PARENT_SELECTION_AUTHORITY_DIGEST:
        raise ValueError("EXP-336 parent selection authority digest")
    selections = payload.get("selections")
    if not isinstance(selections, list) or len(selections) != 3:
        raise ValueError("EXP-336 parent selection coverage")
    selected = [row for row in selections if isinstance(row, Mapping) and row.get("arm_id") == "C_NRS_CORE"]
    if len(selected) != 1:
        raise ValueError("EXP-336 C_NRS selection cardinality")
    row = selected[0]
    if row.get("passes_floor") is not False:
        raise ValueError("EXP-336 parent C_NRS historical floor drift")
    if float(row.get("selected_learning_rate")) != LEARNING_RATE:
        raise ValueError("EXP-336 parent C_NRS learning rate")
    if row.get("selection_digest") != PARENT_CNRS_SELECTION_DIGEST:
        raise ValueError("EXP-336 parent C_NRS selection digest")
    record = row.get("selected_record")
    if not isinstance(record, Mapping):
        raise ValueError("EXP-336 parent C_NRS selected record")
    fixed = {
        "arm_id": "C_NRS_CORE",
        "root": 0,
        "step": STARTING_CUMULATIVE_STEP,
        "learning_rate": LEARNING_RATE,
        "nonfinite_events": 0,
    }
    for key, expected in fixed.items():
        if record.get(key) != expected:
            raise ValueError(f"EXP-336 parent selection drift: {key}")
    if float(record.get("initial_answer_only_loss")) != ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS:
        raise ValueError("EXP-336 parent initial loss drift")
    return payload


def _validate_parent_summary(path: str | Path) -> dict[str, Any]:
    if _file_sha256(path) != PARENT_CNRS_SUMMARY_SHA256:
        raise ValueError("EXP-336 parent summary SHA")
    payload = _read_json(path)
    if payload.get("schema") != "EXP319-TRAINING-CHUNK-SUMMARY-V1":
        raise ValueError("EXP-336 parent summary schema")
    if payload.get("stage") != "A_SANITY" or payload.get("chunk_index") != 3:
        raise ValueError("EXP-336 parent summary chain position")
    if float(payload.get("initial_answer_only_loss")) != ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS:
        raise ValueError("EXP-336 parent summary initial loss")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 1:
        raise ValueError("EXP-336 parent summary snapshot")
    snap = snapshots[0]
    if not isinstance(snap, Mapping) or snap.get("step") != STARTING_CUMULATIVE_STEP:
        raise ValueError("EXP-336 parent summary step")
    return payload


def _validate_optimizer_invariants(optimizer: torch.optim.Optimizer) -> None:
    if not optimizer.param_groups:
        raise ValueError("EXP-336 optimizer has no parameter groups")
    for group in optimizer.param_groups:
        if float(group.get("lr", math.nan)) != LEARNING_RATE:
            raise ValueError("EXP-336 inherited LR drift")
        if float(group.get("weight_decay", math.nan)) != WEIGHT_DECAY:
            raise ValueError("EXP-336 inherited weight decay drift")


def _parent_arm(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    summary_path: str | Path,
    selection_path: str | Path,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    if _file_sha256(checkpoint_path) != PARENT_CNRS_CHECKPOINT_SHA256:
        raise ValueError("EXP-336 parent checkpoint SHA")
    if _file_sha256(receipt_path) != PARENT_CNRS_RECEIPT_SHA256:
        raise ValueError("EXP-336 parent receipt SHA")
    validate_parent_selection_file(selection_path)
    _validate_parent_summary(summary_path)

    bundle = load_checkpoint_bundle(checkpoint_path, receipt_path)
    receipt = bundle.receipt
    fixed = {
        "schema": "EXP319-TRAINING-CHAIN-RECEIPT-V1",
        "stage": "A_SANITY",
        "run_identity": PARENT_CNRS_RUN_IDENTITY,
        "source_commit_sha": PARENT_CNRS_SOURCE_SHA,
        "arm_id": "C_NRS_CORE",
        "root": 0,
        "learning_rate": LEARNING_RATE,
        "chunk_start_step": 768,
        "chunk_end_step": STARTING_CUMULATIVE_STEP,
        "cumulative_step": STARTING_CUMULATIVE_STEP,
        "artifact_digest": PARENT_CNRS_RECEIPT_ARTIFACT_DIGEST,
        "model_state_digest": PARENT_CNRS_MODEL_STATE_DIGEST,
        "optimizer_state_digest": PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
        "rng_state_digest": PARENT_CNRS_RNG_STATE_DIGEST,
    }
    for key, expected in fixed.items():
        if getattr(receipt, key) != expected:
            raise ValueError(f"EXP-336 parent receipt drift: {key}")

    compiled = build_scientific_arm("C_NRS_CORE", device="cpu")
    model = getattr(compiled, "model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("EXP-336 resident parameter count")
    model.load_state_dict(bundle.model_state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(bundle.optimizer_state_dict)
    _validate_optimizer_invariants(optimizer)
    torch.set_rng_state(bundle.torch_rng_state)

    if model_state_digest(model) != PARENT_CNRS_MODEL_STATE_DIGEST:
        raise ValueError("EXP-336 loaded parent model digest")
    if optimizer_state_digest(optimizer) != PARENT_CNRS_OPTIMIZER_STATE_DIGEST:
        raise ValueError("EXP-336 loaded parent optimizer digest")
    if rng_state_digest() != PARENT_CNRS_RNG_STATE_DIGEST:
        raise ValueError("EXP-336 loaded parent RNG digest")

    return compiled, optimizer, {
        "nonfinite_events": 0,
        "negative_target_count": 0,
        "projection_update_count": 0,
        "projected_target_count": 0,
    }


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
    identity: Exp336ExecutionIdentity,
) -> None:
    if receipt.get("schema") != CHUNK_RECEIPT_SCHEMA:
        raise ValueError("EXP-336 receipt schema")
    if receipt.get("arm") != arm or receipt.get("chunk_index") != chunk_index:
        raise ValueError("EXP-336 receipt chain position")
    expected_exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    expected_updates = expected_exposure * len(WORLD_IDS)
    if receipt.get("cumulative_exposure_per_world") != expected_exposure:
        raise ValueError("EXP-336 receipt exposure")
    if receipt.get("cumulative_source_updates") != expected_updates:
        raise ValueError("EXP-336 receipt updates")
    if receipt.get("cumulative_training_step") != STARTING_CUMULATIVE_STEP + expected_updates:
        raise ValueError("EXP-336 receipt cumulative step")
    if receipt.get("data_order_digest") != data_order_digest(expected_exposure):
        raise ValueError("EXP-336 receipt data order")
    if receipt.get("source_tree_digest") != identity.source_tree_digest:
        raise ValueError("EXP-336 receipt source tree")
    if receipt.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-336 receipt preregistration")
    if not _hex_digest(receipt.get("parent_artifact_digest")):
        raise ValueError("EXP-336 receipt parent artifact digest")
    if receipt.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("EXP-336 receipt checkpoint sha")
    for key in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
        if not _hex_digest(receipt.get(key)):
            raise ValueError(f"EXP-336 receipt digest: {key}")
    if receipt.get("receipt_digest") != _receipt_digest(receipt):
        raise ValueError("EXP-336 receipt digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"EXP-336 receipt authorization drift: {key}")


def _new_cnrs_optimizer() -> tuple[object, torch.optim.Optimizer]:
    torch.manual_seed(0)
    compiled = build_scientific_arm("C_NRS_CORE", device="cpu")
    model = getattr(compiled, "model")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return compiled, optimizer


def _load_continuation(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    arm: str,
    previous_chunk_index: int,
    identity: Exp336ExecutionIdentity,
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
        raise ValueError("EXP-336 continuation checkpoint schema")
    if raw.get("arm") != arm or raw.get("chunk_index") != previous_chunk_index:
        raise ValueError("EXP-336 continuation checkpoint chain position")
    compiled, optimizer = _new_cnrs_optimizer()
    model = getattr(compiled, "model")
    model.load_state_dict(raw["model_state_dict"])
    optimizer.load_state_dict(raw["optimizer_state_dict"])
    _validate_optimizer_invariants(optimizer)
    torch.set_rng_state(raw["torch_rng_state"])
    observed = _state_digests(compiled, optimizer)
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise ValueError(f"EXP-336 continuation state mismatch: {key}")
    return compiled, optimizer, receipt


def _dot(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("EXP-336 gradient geometry")
    total = 0.0
    for x, y in zip(a, b):
        total += float(torch.sum(x * y, dtype=torch.float64).item())
    if not math.isfinite(total):
        raise ValueError("EXP-336 nonfinite gradient dot")
    return total


def _norm_sq(a: tuple[torch.Tensor, ...]) -> float:
    return _dot(a, a)


def _project_source(
    source: tuple[torch.Tensor, ...],
    targets: list[tuple[str, tuple[torch.Tensor, ...]]],
) -> tuple[tuple[torch.Tensor, ...], list[str], dict[str, float], dict[str, float]]:
    raw = {world_id: _dot(source, grad) for world_id, grad in targets}
    selected = [
        (world_id, grad)
        for world_id, grad in targets
        if raw[world_id] < 0.0 and _norm_sq(grad) > TARGET_NORM_SQUARED_FLOOR
    ]
    if not selected:
        cloned = tuple(x.clone() for x in source)
        return cloned, [], raw, raw.copy()

    count = len(selected)
    gram = torch.empty((count, count), dtype=torch.float64)
    rhs = torch.empty((count,), dtype=torch.float64)
    for i, (_, gi) in enumerate(selected):
        rhs[i] = _dot(gi, source)
        for j, (_, gj) in enumerate(selected):
            gram[i, j] = _dot(gi, gj)
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
        raise ValueError("EXP-336 requires all other 31 targets")
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
    projected, selected, raw_dots, post_dots = _project_source(source_grads, targets)
    applied = projected if project else source_grads
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


def _evaluate_aggregate(
    compiled: object,
    *,
    cumulative_training_step: int,
    nonfinite_events: int,
) -> dict[str, float]:
    rng = torch.get_rng_state().clone()
    try:
        snapshot = evaluate_snapshot(
            compiled,
            stage="A_SANITY",
            root=0,
            step=cumulative_training_step,
            gradient_norm_preclip=0.0,
            update_norm_ratio=0.0,
            nonfinite_events=nonfinite_events,
        )
    finally:
        torch.set_rng_state(rng)
    loss = float(snapshot.answer_only_loss)
    return {
        "answer_only_loss": loss,
        "answer_token_accuracy": float(snapshot.answer_token_accuracy),
        "greedy_exact_match": float(snapshot.exact_match),
        "eos_correctness": float(snapshot.eos_correctness),
        "invalid_output_rate": float(snapshot.invalid_output_rate),
        "loss_fraction_of_original_initial": loss / ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS,
    }


def _boundary(
    *,
    arm: str,
    chunk_index: int,
    compiled: object,
    optimizer: torch.optim.Optimizer,
    metrics: Mapping[str, list[float]],
    aggregate: Mapping[str, float],
    cumulative_nonfinite: int,
    cumulative_negative: int,
    cumulative_projection_updates: int,
    cumulative_projected_targets: int,
) -> BoundaryResult:
    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    updates = exposure * len(WORLD_IDS)
    digests = _state_digests(compiled, optimizer)
    result = BoundaryResult(
        arm=arm,
        chunk_index=chunk_index,
        cumulative_exposure_per_world=exposure,
        cumulative_source_updates=updates,
        cumulative_training_step=STARTING_CUMULATIVE_STEP + updates,
        world_token_accuracies=tuple(metrics["teacher_forced_answer_token_accuracy"]),
        world_full_answer_exact=tuple(metrics["teacher_forced_full_answer_exact"]),
        aggregate_answer_only_loss=float(aggregate["answer_only_loss"]),
        aggregate_answer_token_accuracy=float(aggregate["answer_token_accuracy"]),
        aggregate_greedy_exact_match=float(aggregate["greedy_exact_match"]),
        aggregate_eos_correctness=float(aggregate["eos_correctness"]),
        aggregate_invalid_output_rate=float(aggregate["invalid_output_rate"]),
        aggregate_loss_fraction_of_original_initial=float(aggregate["loss_fraction_of_original_initial"]),
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
    identity: Exp336ExecutionIdentity,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = {
        "CONTROL_CNRS_FULL32": "control",
        "SHAM_MEASURE_CNRS_FULL32": "sham",
        "SUBSPACE_PROJECT_CNRS_FULL32": "project",
    }[arm]
    checkpoint_path = output / f"{stem}.pt"
    receipt_path = output / f"{stem}-receipt.json"
    if checkpoint_path.exists() or receipt_path.exists():
        raise FileExistsError("EXP-336 output already exists")

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
        "cumulative_training_step": boundary.cumulative_training_step,
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


def _continuation_arm(
    *,
    previous_dir: str | Path,
    arm: str,
    chunk_index: int,
    identity: Exp336ExecutionIdentity,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    stem = {
        "CONTROL_CNRS_FULL32": "control",
        "SHAM_MEASURE_CNRS_FULL32": "sham",
        "SUBSPACE_PROJECT_CNRS_FULL32": "project",
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
) -> tuple[BoundaryResult, dict[str, list[float]], dict[str, float], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    for family, index, exposure, effort in chunk_schedule(chunk_index):
        source_id = f"{family}:{index}"
        source_world = worlds[source_id]
        if arm == "CONTROL_CNRS_FULL32":
            _, _, _, observed = _train_one_step_with_effort(
                compiled, source_world, optimizer=optimizer, effort=effort
            )
            counters["nonfinite_events"] += int(observed)
        else:
            target_worlds = [
                (target_id, worlds[target_id]) for target_id in WORLD_IDS if target_id != source_id
            ]
            row = _measured_step(
                compiled,
                source_world,
                target_worlds,
                optimizer=optimizer,
                effort=effort,
                project=arm == "SUBSPACE_PROJECT_CNRS_FULL32",
                exposure_index=exposure,
                source_world_id=source_id,
            )
            measurements.append(row)
            counters["nonfinite_events"] += int(row["nonfinite_events"])
            counters["negative_target_count"] += len(row["negative_targets"])
            if arm == "SUBSPACE_PROJECT_CNRS_FULL32":
                counters["projection_update_count"] += int(bool(row["projected_targets"]))
                counters["projected_target_count"] += len(row["projected_targets"])
        if counters["nonfinite_events"]:
            raise ValueError(f"EXP-336 nonfinite event in {arm} chunk {chunk_index}")

    exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    cumulative_step = STARTING_CUMULATIVE_STEP + exposure * len(WORLD_IDS)
    metrics = _evaluate_worlds(compiled, worlds)
    aggregate = _evaluate_aggregate(
        compiled,
        cumulative_training_step=cumulative_step,
        nonfinite_events=counters["nonfinite_events"],
    )
    boundary = _boundary(
        arm=arm,
        chunk_index=chunk_index,
        compiled=compiled,
        optimizer=optimizer,
        metrics=metrics,
        aggregate=aggregate,
        cumulative_nonfinite=counters["nonfinite_events"],
        cumulative_negative=counters["negative_target_count"],
        cumulative_projection_updates=counters["projection_update_count"],
        cumulative_projected_targets=counters["projected_target_count"],
    )
    return boundary, metrics, aggregate, measurements


def validate_chunk_summary(
    path: str | Path,
    *,
    identity: Exp336ExecutionIdentity,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != CHUNK_SUMMARY_SCHEMA:
        raise ValueError("EXP-336 chunk summary schema")
    if payload.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-336 chunk preregistration")
    if payload.get("source_tree_digest") != identity.source_tree_digest:
        raise ValueError("EXP-336 chunk source tree")
    if payload.get("execution_digest") != identity.exp336_execution_digest:
        raise ValueError("EXP-336 chunk execution identity")
    chunk_index = payload.get("chunk_index")
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-336 chunk summary index")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("EXP-336 chunk arm coverage")

    boundaries: dict[str, BoundaryResult] = {}
    for arm in ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping):
            raise ValueError("EXP-336 chunk arm record")
        boundary = BoundaryResult(**row["boundary"])
        validate_boundary(boundary)
        if boundary.arm != arm or boundary.chunk_index != chunk_index:
            raise ValueError("EXP-336 chunk boundary identity")
        boundaries[arm] = boundary

    if not sham_equivalent(
        boundaries["CONTROL_CNRS_FULL32"],
        boundaries["SHAM_MEASURE_CNRS_FULL32"],
    ):
        raise ValueError("SHAM_CNRS_FULL32_MISMATCH")

    materialized = dict(payload)
    claimed = materialized.pop("bundle_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-336 chunk bundle digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-336 chunk authorization drift: {key}")
    return payload


def _arm_state(
    *,
    chunk_index: int,
    arm: str,
    execution_identity: Exp336ExecutionIdentity,
    parent_checkpoint: str | Path | None,
    parent_receipt: str | Path | None,
    parent_summary: str | Path | None,
    parent_selection: str | Path | None,
    previous_dir: str | Path | None,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    if chunk_index == 0:
        if None in (parent_checkpoint, parent_receipt, parent_summary, parent_selection):
            raise ValueError("EXP-336 initial parent authority inputs")
        return _parent_arm(
            checkpoint_path=parent_checkpoint,
            receipt_path=parent_receipt,
            summary_path=parent_summary,
            selection_path=parent_selection,
        )
    if previous_dir is None:
        raise ValueError("EXP-336 continuation requires previous chunk")
    return _continuation_arm(
        previous_dir=previous_dir,
        arm=arm,
        chunk_index=chunk_index,
        identity=execution_identity,
    )


def run_chunk(
    *,
    chunk_index: int,
    execution_identity: Exp336ExecutionIdentity,
    code_root: str | Path,
    parent_artifact_digest: str,
    output_dir: str | Path,
    parent_checkpoint: str | Path | None = None,
    parent_receipt: str | Path | None = None,
    parent_summary: str | Path | None = None,
    parent_selection: str | Path | None = None,
    previous_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    if source_tree_digest(code_root) != execution_identity.source_tree_digest:
        raise ValueError("EXP-336 source-tree digest mismatch")
    if not _hex_digest(parent_artifact_digest):
        raise ValueError("EXP-336 parent artifact digest malformed")
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-336 chunk index")

    if chunk_index == 0:
        if parent_artifact_digest != PARENT_CNRS_ZIP_DIGEST:
            raise ValueError("EXP-336 chunk0 parent artifact digest")
        if None in (parent_checkpoint, parent_receipt, parent_summary, parent_selection):
            raise ValueError("EXP-336 chunk0 parent authority")
        validate_parent_selection_file(parent_selection)
        _validate_parent_summary(parent_summary)
    else:
        if previous_dir is None:
            raise ValueError("EXP-336 continuation requires previous chunk")
        previous = validate_chunk_summary(
            Path(previous_dir) / "chunk-summary.json", identity=execution_identity
        )
        if previous["chunk_index"] != chunk_index - 1:
            raise ValueError("EXP-336 duplicate/skipped chain position")

    worlds = _world_map()
    records: dict[str, Any] = {}
    boundaries: dict[str, BoundaryResult] = {}
    for arm in ARMS:
        compiled, optimizer, counters = _arm_state(
            chunk_index=chunk_index,
            arm=arm,
            execution_identity=execution_identity,
            parent_checkpoint=parent_checkpoint,
            parent_receipt=parent_receipt,
            parent_summary=parent_summary,
            parent_selection=parent_selection,
            previous_dir=previous_dir,
        )
        boundary, metrics, aggregate, measurements = _run_arm_chunk(
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
            "aggregate_metrics": aggregate,
            "measurements": measurements,
            "files": files,
        }
        boundaries[arm] = boundary
        del compiled, optimizer
        if arm == "SHAM_MEASURE_CNRS_FULL32" and not sham_equivalent(
            boundaries["CONTROL_CNRS_FULL32"],
            boundaries["SHAM_MEASURE_CNRS_FULL32"],
        ):
            raise ValueError("SHAM_CNRS_FULL32_MISMATCH")

    summary = {
        "schema": CHUNK_SUMMARY_SCHEMA,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": (chunk_index + 1) * EXPOSURES_PER_CHUNK,
        "cumulative_source_updates": (chunk_index + 1) * UPDATES_PER_CHUNK,
        "cumulative_training_step": STARTING_CUMULATIVE_STEP + (chunk_index + 1) * UPDATES_PER_CHUNK,
        "parent_artifact_digest": parent_artifact_digest,
        "parent_cnrs_artifact_name": PARENT_CNRS_ARTIFACT_NAME if chunk_index == 0 else None,
        "source_tree_digest": execution_identity.source_tree_digest,
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "execution_digest": execution_identity.exp336_execution_digest,
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
    execution_identity: Exp336ExecutionIdentity,
    parent_checkpoint: str | Path,
    parent_receipt: str | Path,
    parent_summary: str | Path,
    parent_selection: str | Path,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    _parent_arm(
        checkpoint_path=parent_checkpoint,
        receipt_path=parent_receipt,
        summary_path=parent_summary,
        selection_path=parent_selection,
    )
    if len(chunk_dirs) != CHUNK_COUNT or len(chunk_artifact_digests) != CHUNK_COUNT:
        raise ValueError("EXP-336 finalizer requires exactly 8 chunks and 8 artifact digests")
    if not all(_hex_digest(value) for value in chunk_artifact_digests):
        raise ValueError("EXP-336 malformed chunk artifact digest")

    summaries = [
        validate_chunk_summary(Path(path) / "chunk-summary.json", identity=execution_identity)
        for path in chunk_dirs
    ]
    if [row["chunk_index"] for row in summaries] != list(range(CHUNK_COUNT)):
        raise ValueError("EXP-336 chunk chain order")
    if summaries[0]["parent_artifact_digest"] != PARENT_CNRS_ZIP_DIGEST:
        raise ValueError("EXP-336 root parent artifact chain")
    for index in range(1, CHUNK_COUNT):
        if summaries[index]["parent_artifact_digest"] != chunk_artifact_digests[index - 1]:
            raise ValueError(f"EXP-336 artifact chain mismatch at chunk {index}")

    final = summaries[-1]
    control = _boundary_from_summary(final, "CONTROL_CNRS_FULL32")
    sham = _boundary_from_summary(final, "SHAM_MEASURE_CNRS_FULL32")
    project = _boundary_from_summary(final, "SUBSPACE_PROJECT_CNRS_FULL32")
    if control.cumulative_training_step != FINAL_CUMULATIVE_STEP:
        raise ValueError("EXP-336 final cumulative step")

    decision, vectors = reduce_full32(
        control, sham, project, parent_authority_valid=True, invalid=False
    )
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": asdict(execution_identity),
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "parent_exp319_run_id": execution_identity.parent_exp319_run_id,
        "parent_selection_authority_digest": PARENT_SELECTION_AUTHORITY_DIGEST,
        "parent_cnrs_zip_digest": PARENT_CNRS_ZIP_DIGEST,
        "parent_cnrs_model_state_digest": PARENT_CNRS_MODEL_STATE_DIGEST,
        "parent_cnrs_optimizer_state_digest": PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
        "parent_cnrs_rng_state_digest": PARENT_CNRS_RNG_STATE_DIGEST,
        "parent_exp335_evidence_digest": execution_identity.parent_exp335_evidence_digest,
        "parent_exp335_audit_report_sha256": execution_identity.parent_exp335_audit_report_sha256,
        "chunk_bundle_digests": [row["bundle_digest"] for row in summaries],
        "chunk_artifact_digests": list(chunk_artifact_digests),
        "chunk_parent_artifact_digests": [row["parent_artifact_digest"] for row in summaries],
        "control_sham_exact_by_chunk": [bool(row["control_sham_exact"]) for row in summaries],
        "final_boundaries": {
            "CONTROL_CNRS_FULL32": asdict(control),
            "SHAM_MEASURE_CNRS_FULL32": asdict(sham),
            "SUBSPACE_PROJECT_CNRS_FULL32": asdict(project),
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
        raise ValueError("EXP-336 final schema")
    raw_identity = payload.get("execution_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("EXP-336 final identity")
    identity = Exp336ExecutionIdentity(**raw_identity)
    validate_execution_identity(identity)

    if payload.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-336 final preregistration")
    if payload.get("parent_selection_authority_digest") != PARENT_SELECTION_AUTHORITY_DIGEST:
        raise ValueError("EXP-336 final parent selection")
    if payload.get("parent_cnrs_zip_digest") != PARENT_CNRS_ZIP_DIGEST:
        raise ValueError("EXP-336 final parent C_NRS artifact")
    if payload.get("parent_cnrs_model_state_digest") != PARENT_CNRS_MODEL_STATE_DIGEST:
        raise ValueError("EXP-336 final parent model")
    if payload.get("parent_cnrs_optimizer_state_digest") != PARENT_CNRS_OPTIMIZER_STATE_DIGEST:
        raise ValueError("EXP-336 final parent optimizer")
    if payload.get("parent_cnrs_rng_state_digest") != PARENT_CNRS_RNG_STATE_DIGEST:
        raise ValueError("EXP-336 final parent RNG")
    if payload.get("control_sham_exact_by_chunk") != [True] * CHUNK_COUNT:
        raise ValueError("EXP-336 final SHAM chain")
    if not isinstance(payload.get("chunk_artifact_digests"), list) or len(payload["chunk_artifact_digests"]) != CHUNK_COUNT:
        raise ValueError("EXP-336 final artifact chain")
    if not all(_hex_digest(value) for value in payload["chunk_artifact_digests"]):
        raise ValueError("EXP-336 final artifact digest")

    raw_boundaries = payload.get("final_boundaries")
    if not isinstance(raw_boundaries, Mapping) or set(raw_boundaries) != set(ARMS):
        raise ValueError("EXP-336 final boundary coverage")
    control = BoundaryResult(**raw_boundaries["CONTROL_CNRS_FULL32"])
    sham = BoundaryResult(**raw_boundaries["SHAM_MEASURE_CNRS_FULL32"])
    project = BoundaryResult(**raw_boundaries["SUBSPACE_PROJECT_CNRS_FULL32"])
    decision, vectors = reduce_full32(control, sham, project, parent_authority_valid=True)
    if payload.get("decision") != decision:
        raise ValueError("EXP-336 final reducer")
    for key, expected in vectors.items():
        if payload.get(key) != expected:
            raise ValueError(f"EXP-336 final reducer vector: {key}")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-336 final authorization drift: {key}")

    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-336 final evidence digest")
