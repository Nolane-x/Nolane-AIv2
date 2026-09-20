from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .exp301_execution import EXP301_MAX_SEQUENCE_TOKENS
from .exp301_scientific import build_scientific_arm, forward_scientific_arm
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss
from .exp319_metrics import (
    global_gradient_norm,
    nonfinite_report,
    parameter_update_norm_ratio,
    snapshot_trainable_parameters,
)
from .exp319_training import (
    evaluate_snapshot,
    model_state_digest,
    optimizer_state_digest,
    rng_state_digest,
)
from .exp319_worlds import materialize_stage_a
from .exp324_runtime import _evaluate_subset
from .exp331_runtime import _backward_grads
from .exp336_runtime import _project_source
from .exp337_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    ARMS,
    AUTHORIZATION_FLAGS,
    CHUNK_COUNT,
    EXPOSURES_PER_CHUNK,
    FAMILIES,
    FINAL_CUMULATIVE_STEP,
    GOLD_LOSS_WEIGHT,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    ORIGINAL_EXP319_INITIAL_ANSWER_ONLY_LOSS,
    PARENT_EXP336_CHUNK7_ARTIFACT_NAME,
    PARENT_EXP336_CHUNK7_BUNDLE_DIGEST,
    PARENT_EXP336_CHUNK7_ZIP_SHA256,
    PARENT_EXP336_EVIDENCE_DIGEST,
    PARENT_EXP336_FINAL_DECISION,
    PARENT_EXP336_PREREGISTRATION_DIGEST,
    PARENT_EXP336_SOURCE_TREE_DIGEST,
    PARENT_PROJECT_CHECKPOINT_SHA256,
    PARENT_PROJECT_MODEL_STATE_DIGEST,
    PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
    PARENT_PROJECT_RECEIPT_DIGEST,
    PARENT_PROJECT_RECEIPT_JSON_SHA256,
    PARENT_PROJECT_RNG_STATE_DIGEST,
    SELF_ROLLIN_LOSS_WEIGHT,
    STARTING_CUMULATIVE_STEP,
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
from .exp337_identity import Exp337ExecutionIdentity, validate_execution_identity
from nolane_ai.protocol.identity import source_tree_digest

CHUNK_CHECKPOINT_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-CHECKPOINT-V1"
CHUNK_RECEIPT_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-RECEIPT-V1"
CHUNK_SUMMARY_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-CHUNK-SUMMARY-V1"
FINAL_SCHEMA = "EXP337-CNRS-SELF-ROLLIN-FINAL-EVIDENCE-V1"
EXPECTED_PARAMETER_COUNT = 10_000_000

_STEMS = {
    "CONTROL_PROJECT_GOLD_PREFIX": "control",
    "SHAM_SELF_ROLLIN_MEASURE_PROJECT": "sham",
    "SELF_ROLLIN_RECOVERY_PROJECT": "recovery",
}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXP-337 JSON object required: {path}")
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
            raise ValueError("EXP-337 world identity drift")
        key = f"{family}:{index}"
        if key in worlds:
            raise ValueError("EXP-337 duplicate world")
        worlds[key] = world
    if tuple(worlds) != WORLD_IDS:
        raise ValueError("EXP-337 world population/order drift")
    return worlds


def _validate_optimizer_invariants(optimizer: torch.optim.Optimizer) -> None:
    if not optimizer.param_groups:
        raise ValueError("EXP-337 optimizer has no parameter groups")
    for group in optimizer.param_groups:
        if float(group.get("lr", math.nan)) != LEARNING_RATE:
            raise ValueError("EXP-337 inherited LR drift")
        if float(group.get("weight_decay", math.nan)) != WEIGHT_DECAY:
            raise ValueError("EXP-337 inherited weight decay drift")


def _state_digests(compiled: object, optimizer: torch.optim.Optimizer) -> dict[str, str]:
    model = getattr(compiled, "model")
    return {
        "model_state_digest": model_state_digest(model),
        "optimizer_state_digest": optimizer_state_digest(optimizer),
        "rng_state_digest": rng_state_digest(),
    }


def _new_cnrs_optimizer() -> tuple[object, torch.optim.Optimizer]:
    torch.manual_seed(0)
    compiled = build_scientific_arm("C_NRS_CORE", device="cpu")
    model = getattr(compiled, "model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("EXP-337 resident parameter count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return compiled, optimizer


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    materialized = dict(receipt)
    materialized.pop("receipt_digest", None)
    return canonical_digest(materialized)


def _validate_parent_receipt(path: str | Path) -> dict[str, Any]:
    if _file_sha256(path) != PARENT_PROJECT_RECEIPT_JSON_SHA256:
        raise ValueError("EXP-337 parent PROJECT receipt raw SHA")
    receipt = _read_json(path)
    fixed = {
        "schema": "EXP336-CNRS-CONTINUATION-RECEIPT-V1",
        "arm": "SUBSPACE_PROJECT_CNRS_FULL32",
        "chunk_index": 7,
        "cumulative_exposure_per_world": 32,
        "cumulative_source_updates": 1024,
        "cumulative_training_step": STARTING_CUMULATIVE_STEP,
        "model_state_digest": PARENT_PROJECT_MODEL_STATE_DIGEST,
        "optimizer_state_digest": PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
        "rng_state_digest": PARENT_PROJECT_RNG_STATE_DIGEST,
        "source_tree_digest": PARENT_EXP336_SOURCE_TREE_DIGEST,
        "preregistration_digest": PARENT_EXP336_PREREGISTRATION_DIGEST,
        "checkpoint_sha256": PARENT_PROJECT_CHECKPOINT_SHA256,
        "receipt_digest": PARENT_PROJECT_RECEIPT_DIGEST,
    }
    for key, expected in fixed.items():
        if receipt.get(key) != expected:
            raise ValueError(f"EXP-337 parent PROJECT receipt drift: {key}")
    materialized = dict(receipt)
    claimed = materialized.pop("receipt_digest")
    if canonical_digest(materialized) != claimed:
        raise ValueError("EXP-337 parent PROJECT receipt canonical digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"EXP-337 parent authorization drift: {key}")
    return receipt


def _parent_arm(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    if _file_sha256(checkpoint_path) != PARENT_PROJECT_CHECKPOINT_SHA256:
        raise ValueError("EXP-337 parent PROJECT checkpoint SHA")
    _validate_parent_receipt(receipt_path)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-337 parent PROJECT checkpoint object")
    if raw.get("schema") != "EXP336-CNRS-CONTINUATION-CHECKPOINT-V1":
        raise ValueError("EXP-337 parent PROJECT checkpoint schema")
    if raw.get("arm") != "SUBSPACE_PROJECT_CNRS_FULL32" or raw.get("chunk_index") != 7:
        raise ValueError("EXP-337 parent PROJECT checkpoint identity")

    compiled, optimizer = _new_cnrs_optimizer()
    model = getattr(compiled, "model")
    model.load_state_dict(raw["model_state_dict"])
    optimizer.load_state_dict(raw["optimizer_state_dict"])
    _validate_optimizer_invariants(optimizer)
    torch.set_rng_state(raw["torch_rng_state"])

    observed = _state_digests(compiled, optimizer)
    expected = {
        "model_state_digest": PARENT_PROJECT_MODEL_STATE_DIGEST,
        "optimizer_state_digest": PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
        "rng_state_digest": PARENT_PROJECT_RNG_STATE_DIGEST,
    }
    if observed != expected:
        raise ValueError(f"EXP-337 loaded parent state mismatch: {observed}")
    return compiled, optimizer, {
        "nonfinite_events": 0,
        "negative_target_count": 0,
        "projection_update_count": 0,
        "projected_target_count": 0,
        "self_rollin_measurement_count": 0,
        "self_rollin_active_update_count": 0,
        "self_rollin_divergent_update_count": 0,
    }


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    chunk_index: int,
    checkpoint_path: str | Path,
    identity: Exp337ExecutionIdentity,
) -> None:
    if receipt.get("schema") != CHUNK_RECEIPT_SCHEMA:
        raise ValueError("EXP-337 receipt schema")
    if receipt.get("arm") != arm or receipt.get("chunk_index") != chunk_index:
        raise ValueError("EXP-337 receipt chain position")
    expected_exposure = (chunk_index + 1) * EXPOSURES_PER_CHUNK
    expected_updates = expected_exposure * len(WORLD_IDS)
    if receipt.get("cumulative_exposure_per_world") != expected_exposure:
        raise ValueError("EXP-337 receipt exposure")
    if receipt.get("cumulative_source_updates") != expected_updates:
        raise ValueError("EXP-337 receipt updates")
    if receipt.get("cumulative_training_step") != STARTING_CUMULATIVE_STEP + expected_updates:
        raise ValueError("EXP-337 receipt cumulative step")
    if receipt.get("data_order_digest") != data_order_digest(expected_exposure):
        raise ValueError("EXP-337 receipt data order")
    if receipt.get("source_tree_digest") != identity.source_tree_digest:
        raise ValueError("EXP-337 receipt source tree")
    if receipt.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-337 receipt preregistration")
    if receipt.get("execution_digest") != identity.exp337_execution_digest:
        raise ValueError("EXP-337 receipt execution digest")
    if not _hex_digest(receipt.get("parent_artifact_digest")):
        raise ValueError("EXP-337 receipt parent artifact digest")
    if receipt.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("EXP-337 receipt checkpoint sha")
    for key in ("model_state_digest", "optimizer_state_digest", "rng_state_digest"):
        if not _hex_digest(receipt.get(key)):
            raise ValueError(f"EXP-337 receipt digest: {key}")
    if receipt.get("receipt_digest") != _receipt_digest(receipt):
        raise ValueError("EXP-337 receipt canonical digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"EXP-337 receipt authorization drift: {key}")


def _load_continuation(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    arm: str,
    previous_chunk_index: int,
    identity: Exp337ExecutionIdentity,
) -> tuple[object, torch.optim.Optimizer]:
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
        raise ValueError("EXP-337 continuation checkpoint schema")
    if raw.get("arm") != arm or raw.get("chunk_index") != previous_chunk_index:
        raise ValueError("EXP-337 continuation checkpoint chain position")
    compiled, optimizer = _new_cnrs_optimizer()
    model = getattr(compiled, "model")
    model.load_state_dict(raw["model_state_dict"])
    optimizer.load_state_dict(raw["optimizer_state_dict"])
    _validate_optimizer_invariants(optimizer)
    torch.set_rng_state(raw["torch_rng_state"])
    observed = _state_digests(compiled, optimizer)
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise ValueError(f"EXP-337 continuation state mismatch: {key}")
    return compiled, optimizer


def _self_rollin_tensors(
    compiled: object,
    world: object,
    *,
    effort: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, int | None]:
    tokenizer = Exp301ByteTokenizer()
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode_text(str(getattr(world, "model_input")))
    answer_ids = tokenizer.encode_text(str(getattr(world, "canonical_answer")))
    prefix = (tokenizer.bos_id, *prompt_ids, tokenizer.separator_id)
    if len(prefix) + len(answer_ids) > EXP301_MAX_SEQUENCE_TOKENS:
        raise ValueError("EXP-337 roll-in sequence exceeds frozen max length")

    generated: list[int] = []
    first_error_position: int | None = None
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            tokens = torch.tensor([prefix], dtype=torch.long, device=device)
            for index, canonical_id in enumerate(answer_ids):
                logits = forward_scientific_arm(compiled, tokens, effort=effort)
                next_id = int(torch.argmax(logits[0, -1]).item())
                generated.append(next_id)
                if first_error_position is None and next_id != canonical_id:
                    first_error_position = index
                tokens = torch.cat(
                    (tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1
                )
    finally:
        model.train(was_training)

    input_tokens = (*prefix, *generated)
    target_tokens = (*prefix[1:], *answer_ids, tokenizer.eos_id)
    if len(input_tokens) != len(target_tokens):
        raise AssertionError("EXP-337 self-roll-in shift geometry")
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    targets = torch.tensor([target_tokens], dtype=torch.long, device=device)
    answer_start = len(prefix)
    return input_ids, targets, answer_start, first_error_position is not None, first_error_position


def _self_rollin_backward_grads(
    compiled: object,
    world: object,
    effort: int,
) -> tuple[float, tuple[torch.Tensor, ...], int, bool, int | None]:
    model = getattr(compiled, "model")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    input_ids, targets, answer_start, diverged, first_error = _self_rollin_tensors(
        compiled, world, effort=effort
    )
    model.train()
    logits = forward_scientific_arm(compiled, input_ids, effort=effort)
    loss = compute_answer_only_loss(logits, targets=targets, answer_start=answer_start)
    loss.backward()
    report = nonfinite_report(model.parameters(), loss=loss)
    grads = tuple(
        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params
    )
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise ValueError("EXP-337 nonfinite self-roll-in loss")
    return value, grads, int(report.total), diverged, first_error


def _combine_grads(
    gold: tuple[torch.Tensor, ...],
    rollin: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if len(gold) != len(rollin):
        raise ValueError("EXP-337 source gradient geometry")
    return tuple(
        g * GOLD_LOSS_WEIGHT + r * SELF_ROLLIN_LOSS_WEIGHT for g, r in zip(gold, rollin)
    )


def _projected_step(
    compiled: object,
    source_world: object,
    target_worlds: list[tuple[str, object]],
    *,
    optimizer: torch.optim.Optimizer,
    effort: int,
    arm: str,
    exposure_index: int,
    source_world_id: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("EXP-337 step arm")
    if len(target_worlds) != 31 or source_world_id in {key for key, _ in target_worlds}:
        raise ValueError("EXP-337 requires all other 31 targets")

    model = getattr(compiled, "model")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    before = snapshot_trainable_parameters(model.parameters())
    model.train()
    rng_start = torch.get_rng_state().clone()

    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_start)
    gold_loss, gold_grads, gold_nonfinite = _backward_grads(compiled, source_world, effort)
    rng_after_gold = torch.get_rng_state().clone()

    self_loss: float | None = None
    self_grads: tuple[torch.Tensor, ...] | None = None
    self_nonfinite = 0
    self_diverged = False
    first_error_position: int | None = None
    if arm != "CONTROL_PROJECT_GOLD_PREFIX":
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_start)
        (
            self_loss,
            self_grads,
            self_nonfinite,
            self_diverged,
            first_error_position,
        ) = _self_rollin_backward_grads(compiled, source_world, effort)

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

    if arm == "SELF_ROLLIN_RECOVERY_PROJECT":
        if self_grads is None or self_loss is None:
            raise AssertionError("EXP-337 recovery missing roll-in gradient")
        source_grads = _combine_grads(gold_grads, self_grads)
        source_loss = GOLD_LOSS_WEIGHT * gold_loss + SELF_ROLLIN_LOSS_WEIGHT * self_loss
    else:
        source_grads = gold_grads
        source_loss = gold_loss

    torch.set_rng_state(rng_after_gold)
    optimizer.zero_grad(set_to_none=True)
    projected, selected, raw_dots, post_dots = _project_source(source_grads, targets)
    for parameter, gradient in zip(params, projected):
        parameter.grad = gradient.clone()

    preclip = global_gradient_norm(model.parameters())
    before_report = nonfinite_report(model.parameters(), loss=source_loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    after = snapshot_trainable_parameters(model.parameters())
    ratio = parameter_update_norm_ratio(before, after)
    after_report = nonfinite_report(model.parameters(), loss=source_loss)

    observed = (
        int(gold_nonfinite)
        + int(self_nonfinite)
        + int(target_nonfinite)
        + int(before_report.total)
        + int(after_report.total)
    )
    negative = [world_id for world_id, dot in raw_dots.items() if dot < 0.0]
    return {
        "exposure_index": exposure_index,
        "source_world_id": source_world_id,
        "target_world_ids": [world_id for world_id, _ in target_worlds],
        "effort": effort,
        "arm": arm,
        "gold_source_loss": gold_loss,
        "self_rollin_loss": self_loss,
        "applied_source_loss": source_loss,
        "self_rollin_measured": arm != "CONTROL_PROJECT_GOLD_PREFIX",
        "self_rollin_active": arm == "SELF_ROLLIN_RECOVERY_PROJECT",
        "self_rollin_diverged": self_diverged,
        "first_error_position": first_error_position,
        "target_losses": target_losses,
        "negative_targets": negative,
        "projected_targets": selected,
        "raw_source_target_dots": raw_dots,
        "post_projection_source_target_dots": post_dots,
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
    counters: Mapping[str, int],
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
        nonfinite_events=int(counters["nonfinite_events"]),
        negative_target_count=int(counters["negative_target_count"]),
        projection_update_count=int(counters["projection_update_count"]),
        projected_target_count=int(counters["projected_target_count"]),
        self_rollin_measurement_count=int(counters["self_rollin_measurement_count"]),
        self_rollin_active_update_count=int(counters["self_rollin_active_update_count"]),
        self_rollin_divergent_update_count=int(counters["self_rollin_divergent_update_count"]),
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
    identity: Exp337ExecutionIdentity,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = _STEMS[arm]
    checkpoint_path = output / f"{stem}.pt"
    receipt_path = output / f"{stem}-receipt.json"
    if checkpoint_path.exists() or receipt_path.exists():
        raise FileExistsError("EXP-337 output already exists")

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
        "execution_digest": identity.exp337_execution_digest,
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
    identity: Exp337ExecutionIdentity,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    previous = Path(previous_dir)
    stem = _STEMS[arm]
    compiled, optimizer = _load_continuation(
        checkpoint_path=previous / f"{stem}.pt",
        receipt_path=previous / f"{stem}-receipt.json",
        arm=arm,
        previous_chunk_index=chunk_index - 1,
        identity=identity,
    )
    summary = validate_chunk_summary(previous / "chunk-summary.json", identity=identity)
    row = summary["arms"][arm]["boundary"]
    return compiled, optimizer, {
        "nonfinite_events": int(row["nonfinite_events"]),
        "negative_target_count": int(row["negative_target_count"]),
        "projection_update_count": int(row["projection_update_count"]),
        "projected_target_count": int(row["projected_target_count"]),
        "self_rollin_measurement_count": int(row["self_rollin_measurement_count"]),
        "self_rollin_active_update_count": int(row["self_rollin_active_update_count"]),
        "self_rollin_divergent_update_count": int(row["self_rollin_divergent_update_count"]),
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
        target_worlds = [(target_id, worlds[target_id]) for target_id in WORLD_IDS if target_id != source_id]
        row = _projected_step(
            compiled,
            source_world,
            target_worlds,
            optimizer=optimizer,
            effort=effort,
            arm=arm,
            exposure_index=exposure,
            source_world_id=source_id,
        )
        measurements.append(row)
        counters["nonfinite_events"] += int(row["nonfinite_events"])
        counters["negative_target_count"] += len(row["negative_targets"])
        counters["projection_update_count"] += int(bool(row["projected_targets"]))
        counters["projected_target_count"] += len(row["projected_targets"])
        if bool(row["self_rollin_measured"]):
            counters["self_rollin_measurement_count"] += 1
        if bool(row["self_rollin_active"]):
            counters["self_rollin_active_update_count"] += 1
        if bool(row["self_rollin_diverged"]):
            counters["self_rollin_divergent_update_count"] += 1
        if counters["nonfinite_events"]:
            raise ValueError(f"EXP-337 nonfinite event in {arm} chunk {chunk_index}")

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
        counters=counters,
    )
    return boundary, metrics, aggregate, measurements


def validate_chunk_summary(
    path: str | Path,
    *,
    identity: Exp337ExecutionIdentity,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != CHUNK_SUMMARY_SCHEMA:
        raise ValueError("EXP-337 chunk summary schema")
    if payload.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-337 chunk preregistration")
    if payload.get("source_tree_digest") != identity.source_tree_digest:
        raise ValueError("EXP-337 chunk source tree")
    if payload.get("execution_digest") != identity.exp337_execution_digest:
        raise ValueError("EXP-337 chunk execution identity")
    chunk_index = payload.get("chunk_index")
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-337 chunk summary index")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("EXP-337 chunk arm coverage")

    boundaries: dict[str, BoundaryResult] = {}
    for arm in ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping):
            raise ValueError("EXP-337 chunk arm record")
        boundary = BoundaryResult(**row["boundary"])
        validate_boundary(boundary)
        if boundary.arm != arm or boundary.chunk_index != chunk_index:
            raise ValueError("EXP-337 chunk boundary identity")
        boundaries[arm] = boundary

    if not sham_equivalent(
        boundaries["CONTROL_PROJECT_GOLD_PREFIX"],
        boundaries["SHAM_SELF_ROLLIN_MEASURE_PROJECT"],
    ):
        raise ValueError("EXP337_SHAM_MEASUREMENT_MISMATCH")

    materialized = dict(payload)
    claimed = materialized.pop("bundle_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-337 chunk bundle digest")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-337 chunk authorization drift: {key}")
    return payload


def _arm_state(
    *,
    chunk_index: int,
    arm: str,
    execution_identity: Exp337ExecutionIdentity,
    parent_checkpoint: str | Path | None,
    parent_receipt: str | Path | None,
    previous_dir: str | Path | None,
) -> tuple[object, torch.optim.Optimizer, dict[str, int]]:
    if chunk_index == 0:
        if parent_checkpoint is None or parent_receipt is None:
            raise ValueError("EXP-337 initial parent authority inputs")
        return _parent_arm(checkpoint_path=parent_checkpoint, receipt_path=parent_receipt)
    if previous_dir is None:
        raise ValueError("EXP-337 continuation requires previous chunk")
    return _continuation_arm(
        previous_dir=previous_dir,
        arm=arm,
        chunk_index=chunk_index,
        identity=execution_identity,
    )


def run_chunk(
    *,
    chunk_index: int,
    execution_identity: Exp337ExecutionIdentity,
    code_root: str | Path,
    parent_artifact_digest: str,
    output_dir: str | Path,
    parent_checkpoint: str | Path | None = None,
    parent_receipt: str | Path | None = None,
    previous_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    if source_tree_digest(code_root) != execution_identity.source_tree_digest:
        raise ValueError("EXP-337 source-tree digest mismatch")
    if not _hex_digest(parent_artifact_digest):
        raise ValueError("EXP-337 parent artifact digest malformed")
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("EXP-337 chunk index")

    if chunk_index == 0:
        if parent_artifact_digest != PARENT_EXP336_CHUNK7_ZIP_SHA256:
            raise ValueError("EXP-337 chunk0 parent artifact digest")
        if parent_checkpoint is None or parent_receipt is None:
            raise ValueError("EXP-337 chunk0 parent authority")
        _validate_parent_receipt(parent_receipt)
        if _file_sha256(parent_checkpoint) != PARENT_PROJECT_CHECKPOINT_SHA256:
            raise ValueError("EXP-337 chunk0 parent checkpoint")
    else:
        if previous_dir is None:
            raise ValueError("EXP-337 continuation requires previous chunk")
        previous = validate_chunk_summary(
            Path(previous_dir) / "chunk-summary.json", identity=execution_identity
        )
        if previous["chunk_index"] != chunk_index - 1:
            raise ValueError("EXP-337 duplicate/skipped chain position")

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
        if arm == "SHAM_SELF_ROLLIN_MEASURE_PROJECT" and not sham_equivalent(
            boundaries["CONTROL_PROJECT_GOLD_PREFIX"],
            boundaries["SHAM_SELF_ROLLIN_MEASURE_PROJECT"],
        ):
            raise ValueError("EXP337_SHAM_MEASUREMENT_MISMATCH")

    summary = {
        "schema": CHUNK_SUMMARY_SCHEMA,
        "chunk_index": chunk_index,
        "cumulative_exposure_per_world": (chunk_index + 1) * EXPOSURES_PER_CHUNK,
        "cumulative_source_updates": (chunk_index + 1) * UPDATES_PER_CHUNK,
        "cumulative_training_step": STARTING_CUMULATIVE_STEP + (chunk_index + 1) * UPDATES_PER_CHUNK,
        "parent_artifact_digest": parent_artifact_digest,
        "parent_exp336_artifact_name": PARENT_EXP336_CHUNK7_ARTIFACT_NAME if chunk_index == 0 else None,
        "parent_exp336_bundle_digest": PARENT_EXP336_CHUNK7_BUNDLE_DIGEST if chunk_index == 0 else None,
        "source_tree_digest": execution_identity.source_tree_digest,
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "execution_digest": execution_identity.exp337_execution_digest,
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
    execution_identity: Exp337ExecutionIdentity,
    parent_checkpoint: str | Path,
    parent_receipt: str | Path,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    _parent_arm(checkpoint_path=parent_checkpoint, receipt_path=parent_receipt)
    if len(chunk_dirs) != CHUNK_COUNT or len(chunk_artifact_digests) != CHUNK_COUNT:
        raise ValueError("EXP-337 finalizer requires exactly 8 chunks and 8 artifact digests")
    if not all(_hex_digest(value) for value in chunk_artifact_digests):
        raise ValueError("EXP-337 malformed chunk artifact digest")

    summaries = [
        validate_chunk_summary(Path(path) / "chunk-summary.json", identity=execution_identity)
        for path in chunk_dirs
    ]
    if [row["chunk_index"] for row in summaries] != list(range(CHUNK_COUNT)):
        raise ValueError("EXP-337 chunk chain order")
    if summaries[0]["parent_artifact_digest"] != PARENT_EXP336_CHUNK7_ZIP_SHA256:
        raise ValueError("EXP-337 root parent artifact chain")
    for index in range(1, CHUNK_COUNT):
        if summaries[index]["parent_artifact_digest"] != chunk_artifact_digests[index - 1]:
            raise ValueError(f"EXP-337 artifact chain mismatch at chunk {index}")

    final = summaries[-1]
    control = _boundary_from_summary(final, "CONTROL_PROJECT_GOLD_PREFIX")
    sham = _boundary_from_summary(final, "SHAM_SELF_ROLLIN_MEASURE_PROJECT")
    recovery = _boundary_from_summary(final, "SELF_ROLLIN_RECOVERY_PROJECT")
    if control.cumulative_training_step != FINAL_CUMULATIVE_STEP:
        raise ValueError("EXP-337 final cumulative step")

    decision, vectors = reduce_full32(
        control, sham, recovery, parent_authority_valid=True, invalid=False
    )
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": asdict(execution_identity),
        "preregistration_digest": APPROVED_PREREGISTRATION_DIGEST,
        "parent_exp336_final_decision": PARENT_EXP336_FINAL_DECISION,
        "parent_exp336_evidence_digest": PARENT_EXP336_EVIDENCE_DIGEST,
        "parent_exp336_chunk7_zip_sha256": PARENT_EXP336_CHUNK7_ZIP_SHA256,
        "parent_project_model_state_digest": PARENT_PROJECT_MODEL_STATE_DIGEST,
        "parent_project_optimizer_state_digest": PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
        "parent_project_rng_state_digest": PARENT_PROJECT_RNG_STATE_DIGEST,
        "chunk_bundle_digests": [row["bundle_digest"] for row in summaries],
        "chunk_artifact_digests": list(chunk_artifact_digests),
        "chunk_parent_artifact_digests": [row["parent_artifact_digest"] for row in summaries],
        "control_sham_exact_by_chunk": [bool(row["control_sham_exact"]) for row in summaries],
        "final_boundaries": {
            "CONTROL_PROJECT_GOLD_PREFIX": asdict(control),
            "SHAM_SELF_ROLLIN_MEASURE_PROJECT": asdict(sham),
            "SELF_ROLLIN_RECOVERY_PROJECT": asdict(recovery),
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
        raise ValueError("EXP-337 final schema")
    raw_identity = payload.get("execution_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("EXP-337 final identity")
    identity = Exp337ExecutionIdentity(**raw_identity)
    validate_execution_identity(identity)

    if payload.get("preregistration_digest") != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-337 final preregistration")
    if payload.get("parent_exp336_final_decision") != PARENT_EXP336_FINAL_DECISION:
        raise ValueError("EXP-337 final parent decision")
    if payload.get("parent_exp336_evidence_digest") != PARENT_EXP336_EVIDENCE_DIGEST:
        raise ValueError("EXP-337 final parent evidence")
    if payload.get("parent_exp336_chunk7_zip_sha256") != PARENT_EXP336_CHUNK7_ZIP_SHA256:
        raise ValueError("EXP-337 final parent artifact")
    if payload.get("control_sham_exact_by_chunk") != [True] * CHUNK_COUNT:
        raise ValueError("EXP-337 final SHAM chain")
    if not isinstance(payload.get("chunk_artifact_digests"), list) or len(payload["chunk_artifact_digests"]) != CHUNK_COUNT:
        raise ValueError("EXP-337 final artifact chain")
    if not all(_hex_digest(value) for value in payload["chunk_artifact_digests"]):
        raise ValueError("EXP-337 final artifact digest")

    raw_boundaries = payload.get("final_boundaries")
    if not isinstance(raw_boundaries, Mapping) or set(raw_boundaries) != set(ARMS):
        raise ValueError("EXP-337 final boundary coverage")
    control = BoundaryResult(**raw_boundaries["CONTROL_PROJECT_GOLD_PREFIX"])
    sham = BoundaryResult(**raw_boundaries["SHAM_SELF_ROLLIN_MEASURE_PROJECT"])
    recovery = BoundaryResult(**raw_boundaries["SELF_ROLLIN_RECOVERY_PROJECT"])
    decision, vectors = reduce_full32(control, sham, recovery, parent_authority_valid=True)
    if payload.get("decision") != decision:
        raise ValueError("EXP-337 final reducer")
    for key, expected in vectors.items():
        if payload.get(key) != expected:
            raise ValueError(f"EXP-337 final reducer vector: {key}")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-337 final authorization drift: {key}")

    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or canonical_digest(materialized) != claimed:
        raise ValueError("EXP-337 final evidence digest")
