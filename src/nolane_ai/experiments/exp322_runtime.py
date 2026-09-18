from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

import torch

from .exp301_scientific import build_scientific_arm, forward_scientific_arm
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss, effort_for_training_step
from .exp319_metrics import (
    global_gradient_norm,
    nonfinite_report,
    parameter_update_norm_ratio,
    snapshot_trainable_parameters,
)
from .exp319_training import (
    _diagnostic_generate,
    load_checkpoint_bundle,
    model_state_digest,
    optimizer_state_digest,
    rng_state_digest,
)
from .exp319_worlds import materialize_stage_a, verify_world_answer
from .exp322_contract import (
    ARM_LEARNING_RATES,
    ARMS,
    CHECKPOINTS,
    PARENT_MODEL_STATE_DIGEST,
    PARENT_RECEIPT_ARTIFACT_DIGEST,
    STARTING_STEP,
    InterventionSnapshot,
)
from .exp322_evidence import build_arm_evidence


EXPECTED_PARAMETER_COUNT = 10_000_000
PARENT_ARM = "A_FIXED"
PARENT_STAGE = "A_SANITY"
PARENT_ROOT = 0
PARENT_LEARNING_RATE = 1e-4


def validate_parent_receipt(receipt: object) -> None:
    expected = {
        "stage": PARENT_STAGE,
        "arm_id": PARENT_ARM,
        "root": PARENT_ROOT,
        "learning_rate": PARENT_LEARNING_RATE,
        "cumulative_step": STARTING_STEP,
        "artifact_digest": PARENT_RECEIPT_ARTIFACT_DIGEST,
        "model_state_digest": PARENT_MODEL_STATE_DIGEST,
    }
    for attribute, value in expected.items():
        if getattr(receipt, attribute, None) != value:
            raise ValueError(f"EXP-322 parent receipt mismatch: {attribute}")


def apply_intervention_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
) -> None:
    if arm not in ARMS:
        raise ValueError("unknown EXP-322 arm")
    groups = optimizer.param_groups
    if not groups:
        raise ValueError("EXP-322 optimizer has no parameter groups")
    for group in groups:
        if float(group.get("lr", math.nan)) != PARENT_LEARNING_RATE:
            raise ValueError("EXP-322 parent optimizer LR is not the sealed 1e-4")
        if float(group.get("weight_decay", math.nan)) != 0.01:
            raise ValueError("EXP-322 parent optimizer weight decay drift")
    target = ARM_LEARNING_RATES[arm]
    if arm == "DECAY_5E5":
        for group in groups:
            group["lr"] = target
    for group in groups:
        if float(group["lr"]) != target:
            raise ValueError("EXP-322 intervention learning rate was not applied exactly")


def _encoded_tensors(world: object, *, device: torch.device):
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example(
        str(getattr(world, "model_input")),
        str(getattr(world, "canonical_answer")),
    )
    input_ids = torch.tensor([encoded.token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long, device=device)
    return input_ids, targets, encoded.answer_start


def _train_one_step(
    compiled: object,
    world: object,
    *,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> tuple[float, float, float, int]:
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    input_ids, targets, answer_start = _encoded_tensors(world, device=device)
    before = snapshot_trainable_parameters(model.parameters())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = forward_scientific_arm(
        compiled,
        input_ids,
        effort=effort_for_training_step(global_step),
    )
    loss = compute_answer_only_loss(
        logits,
        targets=targets,
        answer_start=answer_start,
    )
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


def _evaluate(
    compiled: object,
    *,
    arm: str,
    step: int,
    gradient_norm_preclip: float,
    parameter_update_norm_ratio_value: float,
    nonfinite_events: int,
) -> tuple[InterventionSnapshot, dict[str, Any]]:
    worlds = tuple(materialize_stage_a())
    if len(worlds) != 32:
        raise ValueError("EXP-322 requires exactly 32 Stage-A worlds")
    identities = tuple(
        (str(getattr(world, "family")), str(getattr(world, "content_id")))
        for world in worlds
    )
    if len(set(identities)) != 32:
        raise ValueError("EXP-322 Stage-A world identities must be unique")

    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    tokenizer = Exp301ByteTokenizer()
    correct_tokens = 0
    target_tokens = 0
    full_exact = 0
    weighted_loss = 0.0
    greedy_exact = 0
    eos_correct = 0
    invalid_count = 0
    families: dict[str, dict[str, int]] = {}

    model.eval()
    with torch.inference_mode():
        for world in worlds:
            input_ids, targets, answer_start = _encoded_tensors(world, device=device)
            logits = forward_scientific_arm(compiled, input_ids, effort=4)
            first = answer_start - 1
            answer_logits = logits[0, first:]
            answer_targets = targets[0, first:]
            top1 = torch.argmax(answer_logits, dim=-1)
            correct = top1.eq(answer_targets)
            count = int(correct.numel())
            hits = int(correct.sum().item())
            is_full = bool(correct.all().item())
            loss = compute_answer_only_loss(
                logits,
                targets=targets,
                answer_start=answer_start,
            )
            weighted_loss += float(loss.item()) * count
            target_tokens += count
            correct_tokens += hits
            full_exact += int(is_full)

            candidate, generated, stopped = _diagnostic_generate(
                compiled,
                prompt=str(getattr(world, "model_input")),
            )
            greedy_exact += int(verify_world_answer(world, candidate))
            eos_correct += int(
                stopped and bool(generated) and generated[-1] == tokenizer.eos_id
            )
            invalid = any(
                token != tokenizer.eos_id
                and not tokenizer.byte_offset <= token < tokenizer.byte_offset + 256
                for token in generated
            )
            invalid_count += int(invalid)

            family = str(getattr(world, "family"))
            bucket = families.setdefault(
                family,
                {"worlds": 0, "targets": 0, "correct": 0, "full_exact": 0},
            )
            bucket["worlds"] += 1
            bucket["targets"] += count
            bucket["correct"] += hits
            bucket["full_exact"] += int(is_full)

    if target_tokens <= 0:
        raise RuntimeError("EXP-322 evaluation found no answer targets")

    snapshot = InterventionSnapshot(
        arm=arm,
        step=step,
        teacher_forced_answer_token_accuracy=correct_tokens / target_tokens,
        teacher_forced_full_answer_exact=full_exact / len(worlds),
        greedy_exact=greedy_exact / len(worlds),
        eos_correctness=eos_correct / len(worlds),
        invalid_output_rate=invalid_count / len(worlds),
        answer_only_loss=weighted_loss / target_tokens,
        gradient_norm_preclip=gradient_norm_preclip,
        parameter_update_norm_ratio=parameter_update_norm_ratio_value,
        nonfinite_events=nonfinite_events,
    )
    family_summary = {
        family: {
            "world_count": bucket["worlds"],
            "teacher_forced_answer_token_accuracy": (
                bucket["correct"] / bucket["targets"]
            ),
            "teacher_forced_full_answer_exact": (
                bucket["full_exact"] / bucket["worlds"]
            ),
        }
        for family, bucket in sorted(families.items())
    }
    return snapshot, family_summary


def run_intervention_arm(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    arm: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown EXP-322 arm")
    bundle = load_checkpoint_bundle(checkpoint_path, receipt_path)
    validate_parent_receipt(bundle.receipt)

    torch.manual_seed(bundle.receipt.model_init_seed)
    compiled = build_scientific_arm(PARENT_ARM, device="cpu")
    model = getattr(compiled, "model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("EXP-322 resident trainable-parameter count drift")
    model.load_state_dict(bundle.model_state_dict)
    if model_state_digest(model) != PARENT_MODEL_STATE_DIGEST:
        raise ValueError("EXP-322 loaded model-state digest mismatch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PARENT_LEARNING_RATE,
        weight_decay=0.01,
    )
    optimizer.load_state_dict(bundle.optimizer_state_dict)
    if optimizer_state_digest(optimizer) != bundle.receipt.optimizer_state_digest:
        raise ValueError("EXP-322 loaded optimizer-state digest mismatch")
    apply_intervention_learning_rate(optimizer, arm=arm)

    torch.set_rng_state(bundle.torch_rng_state)
    if rng_state_digest() != bundle.receipt.rng_state_digest:
        raise ValueError("EXP-322 loaded RNG-state digest mismatch")

    worlds = tuple(materialize_stage_a())
    snapshots: list[InterventionSnapshot] = []
    family_summaries: dict[str, Any] = {}
    cumulative_nonfinite = 0
    last_grad = 0.0
    last_update = 0.0
    completed_step = STARTING_STEP
    invalid_reason: str | None = None

    for global_step in range(STARTING_STEP, CHECKPOINTS[-1]):
        world = worlds[global_step % len(worlds)]
        _, last_grad, last_update, nonfinite = _train_one_step(
            compiled,
            world,
            optimizer=optimizer,
            global_step=global_step,
        )
        cumulative_nonfinite += nonfinite
        completed_step = global_step + 1
        if cumulative_nonfinite:
            invalid_reason = "NONFINITE_EVENT"
            break
        if completed_step in CHECKPOINTS:
            snapshot, family = _evaluate(
                compiled,
                arm=arm,
                step=completed_step,
                gradient_norm_preclip=last_grad,
                parameter_update_norm_ratio_value=last_update,
                nonfinite_events=cumulative_nonfinite,
            )
            snapshots.append(snapshot)
            family_summaries[str(completed_step)] = family

    return build_arm_evidence(
        arm=arm,
        snapshots=tuple(snapshots),
        family_summaries=family_summaries,
        completed_step=completed_step,
        final_model_state_digest=model_state_digest(model),
        final_optimizer_state_digest=optimizer_state_digest(optimizer),
        final_rng_state_digest=rng_state_digest(),
        invalid_reason=invalid_reason,
    )
