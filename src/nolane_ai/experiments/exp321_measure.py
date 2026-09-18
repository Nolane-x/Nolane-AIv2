from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .exp301_scientific import build_scientific_arm, forward_scientific_arm
from .exp301_training import Exp301ByteTokenizer
from .exp319_training import load_checkpoint_bundle, model_state_digest
from .exp319_worlds import materialize_stage_a, verify_world_answer
from .exp321_contract import (
    AUTHORIZATION_FLAGS,
    BYTE_ID_END_INCLUSIVE,
    BYTE_ID_START,
    EFFORT_GRID,
    EOS_ID,
    REPRODUCTION_ANCHORS,
    SELECTED_ARM,
    SELECTED_LEARNING_RATE,
    SELECTED_MODEL_STATE_DIGEST,
    SELECTED_RECEIPT_ARTIFACT_DIGEST,
    SELECTED_STEP,
    UNUSED_ID_END_INCLUSIVE,
    UNUSED_ID_START,
    VOCAB_SIZE,
    canonical_json_bytes,
    preregistration_digest,
)
from .exp321_localization import EffortSummary, reduce_localization


@dataclass(frozen=True, slots=True)
class WorldMeasurement:
    effort: int
    family: str
    content_id: str
    answer_length_tokens: int
    teacher_forced_token_accuracy: float
    teacher_forced_full_answer_exact: bool
    mean_correct_token_probability: float
    mean_correct_token_rank_full: float
    mean_correct_token_rank_legal: float
    mean_legal_probability_mass: float
    mean_unused_probability_mass: float
    masked_wrong_target_recovery: float
    greedy_exact: bool
    greedy_eos_correct: bool
    greedy_invalid: bool
    greedy_first_mismatch: int | None
    greedy_correct_prefix: int
    masked_greedy_exact: bool
    masked_greedy_eos_correct: bool
    masked_greedy_invalid: bool
    masked_greedy_first_mismatch: int | None
    masked_greedy_correct_prefix: int


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _expected_answer_ids(world: object) -> tuple[int, ...]:
    tokenizer = Exp301ByteTokenizer()
    return (*tokenizer.encode_text(getattr(world, "canonical_answer")), tokenizer.eos_id)


def _sequence_metrics(generated: tuple[int, ...], expected: tuple[int, ...]) -> tuple[int | None, int]:
    width = max(len(generated), len(expected))
    prefix = 0
    first: int | None = None
    for index in range(width):
        same = (
            index < len(generated)
            and index < len(expected)
            and generated[index] == expected[index]
        )
        if same and first is None:
            prefix += 1
        elif first is None:
            first = index
    return first, prefix


def _generate(
    compiled: object,
    *,
    prompt: str,
    effort: int,
    legal_mask: bool,
) -> tuple[str, tuple[int, ...], bool, bool]:
    tokenizer = Exp301ByteTokenizer()
    prompt_ids = (tokenizer.bos_id, *tokenizer.encode_text(prompt), tokenizer.separator_id)
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    answer_ids: list[int] = []
    stopped = False
    invalid = False

    allowed = None
    if legal_mask:
        allowed = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
        allowed[EOS_ID] = True
        allowed[BYTE_ID_START : BYTE_ID_END_INCLUSIVE + 1] = True

    model.eval()
    with torch.inference_mode():
        for _ in range(96):
            logits = forward_scientific_arm(compiled, tokens, effort=effort)[0, -1]
            if allowed is not None:
                logits = logits.masked_fill(~allowed, float("-inf"))
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                stopped = True
                break
            if not tokenizer.byte_offset <= next_id < tokenizer.byte_offset + 256:
                invalid = True
                break
            answer_ids.append(next_id)
            tokens = torch.cat(
                (tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)),
                dim=1,
            )
    candidate = "" if invalid else tokenizer.decode(answer_ids)
    return candidate, tuple(generated), stopped, invalid


def _measure_world(compiled: object, world: object, effort: int) -> WorldMeasurement:
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example(
        getattr(world, "model_input"),
        getattr(world, "canonical_answer"),
    )
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded.token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long, device=device)
    first = encoded.answer_start - 1

    model.eval()
    with torch.inference_mode():
        logits = forward_scientific_arm(compiled, input_ids, effort=effort)[0, first:]
        answer_targets = targets[0, first:]
        probs = torch.softmax(logits, dim=-1)
        top1 = torch.argmax(logits, dim=-1)
        correct = top1.eq(answer_targets)

        target_logits = logits.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
        rank_full = logits.gt(target_logits.unsqueeze(1)).sum(dim=1).add(1)

        legal_logits = torch.cat(
            (
                logits[:, EOS_ID : EOS_ID + 1],
                logits[:, BYTE_ID_START : BYTE_ID_END_INCLUSIVE + 1],
            ),
            dim=1,
        )
        legal_target_index = torch.where(
            answer_targets.eq(EOS_ID),
            torch.zeros_like(answer_targets),
            answer_targets - BYTE_ID_START + 1,
        )
        legal_target_logits = legal_logits.gather(
            1, legal_target_index.unsqueeze(1)
        ).squeeze(1)
        rank_legal = legal_logits.gt(legal_target_logits.unsqueeze(1)).sum(dim=1).add(1)

        legal_mass = probs[:, EOS_ID] + probs[
            :, BYTE_ID_START : BYTE_ID_END_INCLUSIVE + 1
        ].sum(dim=1)
        unused_mass = probs[
            :, UNUSED_ID_START : UNUSED_ID_END_INCLUSIVE + 1
        ].sum(dim=1)

        masked_logits = logits.clone()
        mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
        mask[EOS_ID] = False
        mask[BYTE_ID_START : BYTE_ID_END_INCLUSIVE + 1] = False
        masked_logits[:, mask] = float("-inf")
        masked_top1 = torch.argmax(masked_logits, dim=-1)
        wrong = ~correct
        recoveries = wrong & masked_top1.eq(answer_targets)
        recovery = (
            float(recoveries.sum().item()) / float(wrong.sum().item())
            if bool(wrong.any())
            else 0.0
        )

    expected = _expected_answer_ids(world)
    candidate, generated, stopped, invalid = _generate(
        compiled,
        prompt=getattr(world, "model_input"),
        effort=effort,
        legal_mask=False,
    )
    masked_candidate, masked_generated, masked_stopped, masked_invalid = _generate(
        compiled,
        prompt=getattr(world, "model_input"),
        effort=effort,
        legal_mask=True,
    )
    first_mismatch, prefix = _sequence_metrics(generated, expected)
    masked_first_mismatch, masked_prefix = _sequence_metrics(masked_generated, expected)

    return WorldMeasurement(
        effort=effort,
        family=str(getattr(world, "family")),
        content_id=str(getattr(world, "content_id")),
        answer_length_tokens=len(expected),
        teacher_forced_token_accuracy=float(correct.float().mean().item()),
        teacher_forced_full_answer_exact=bool(correct.all().item()),
        mean_correct_token_probability=float(
            probs.gather(1, answer_targets.unsqueeze(1)).mean().item()
        ),
        mean_correct_token_rank_full=float(rank_full.float().mean().item()),
        mean_correct_token_rank_legal=float(rank_legal.float().mean().item()),
        mean_legal_probability_mass=float(legal_mass.mean().item()),
        mean_unused_probability_mass=float(unused_mass.mean().item()),
        masked_wrong_target_recovery=recovery,
        greedy_exact=bool(verify_world_answer(world, candidate)),
        greedy_eos_correct=stopped and bool(generated) and generated[-1] == EOS_ID,
        greedy_invalid=invalid,
        greedy_first_mismatch=first_mismatch,
        greedy_correct_prefix=prefix,
        masked_greedy_exact=bool(verify_world_answer(world, masked_candidate)),
        masked_greedy_eos_correct=(
            masked_stopped
            and bool(masked_generated)
            and masked_generated[-1] == EOS_ID
        ),
        masked_greedy_invalid=masked_invalid,
        masked_greedy_first_mismatch=masked_first_mismatch,
        masked_greedy_correct_prefix=masked_prefix,
    )


def _aggregate(records: tuple[WorldMeasurement, ...], effort: int) -> EffortSummary:
    selected = tuple(item for item in records if item.effort == effort)
    if len(selected) != 32:
        raise ValueError("EXP-321 requires exactly 32 measurements per effort")
    total_targets = sum(item.answer_length_tokens for item in selected)
    token_correct = sum(
        item.teacher_forced_token_accuracy * item.answer_length_tokens
        for item in selected
    )
    wrong_targets = sum(
        (1.0 - item.teacher_forced_token_accuracy) * item.answer_length_tokens
        for item in selected
    )
    recovered = sum(
        item.masked_wrong_target_recovery
        * (1.0 - item.teacher_forced_token_accuracy)
        * item.answer_length_tokens
        for item in selected
    )
    return EffortSummary(
        effort=effort,
        teacher_forced_token_accuracy=token_correct / total_targets,
        teacher_forced_full_answer_exact=sum(
            item.teacher_forced_full_answer_exact for item in selected
        )
        / len(selected),
        greedy_exact=sum(item.greedy_exact for item in selected) / len(selected),
        masked_greedy_exact=sum(item.masked_greedy_exact for item in selected)
        / len(selected),
        masked_wrong_target_recovery=(recovered / wrong_targets if wrong_targets else 0.0),
    )


def _family_exact(records: tuple[WorldMeasurement, ...], effort: int) -> dict[str, float]:
    chosen = tuple(item for item in records if item.effort == effort)
    families = sorted({item.family for item in chosen})
    result: dict[str, float] = {}
    for family in families:
        rows = tuple(item for item in chosen if item.family == family)
        result[family] = sum(item.greedy_exact for item in rows) / len(rows)
    return result


def _reproduction_valid(
    records: tuple[WorldMeasurement, ...],
    summaries: dict[int, EffortSummary],
) -> bool:
    baseline = summaries[REPRODUCTION_ANCHORS.effort]
    if not math_isclose(
        baseline.teacher_forced_token_accuracy,
        REPRODUCTION_ANCHORS.teacher_forced_token_accuracy,
    ):
        return False
    if baseline.greedy_exact != REPRODUCTION_ANCHORS.greedy_exact:
        return False
    chosen = tuple(
        item for item in records if item.effort == REPRODUCTION_ANCHORS.effort
    )
    eos = sum(item.greedy_eos_correct for item in chosen) / len(chosen)
    invalid = sum(item.greedy_invalid for item in chosen) / len(chosen)
    if eos != REPRODUCTION_ANCHORS.eos_correctness:
        return False
    if invalid != REPRODUCTION_ANCHORS.invalid_output_rate:
        return False
    return _family_exact(records, 4) == {
        "algorithmic-sequence-transform": 0.0,
        "generator-heldout-abstract-transformation": 0.625,
        "iterative-grid-and-maze": 0.5,
        "language-sequence-control": 0.0,
    }


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def run_localization(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    bundle = load_checkpoint_bundle(checkpoint_path, receipt_path)
    receipt = bundle.receipt
    if (
        receipt.stage != "A_SANITY"
        or receipt.arm_id != SELECTED_ARM
        or receipt.root != 0
        or receipt.learning_rate != SELECTED_LEARNING_RATE
        or receipt.cumulative_step != SELECTED_STEP
        or receipt.artifact_digest != SELECTED_RECEIPT_ARTIFACT_DIGEST
    ):
        raise ValueError("checkpoint receipt does not match frozen EXP-321 authority")

    torch.manual_seed(receipt.model_init_seed)
    compiled = build_scientific_arm(SELECTED_ARM, device="cpu")
    getattr(compiled, "model").load_state_dict(bundle.model_state_dict)
    actual_model_digest = model_state_digest(getattr(compiled, "model"))
    if actual_model_digest != SELECTED_MODEL_STATE_DIGEST:
        raise ValueError("checkpoint model-state digest does not match EXP-321 authority")

    records = tuple(
        _measure_world(compiled, world, effort)
        for effort in EFFORT_GRID
        for world in materialize_stage_a()
    )
    summaries = {effort: _aggregate(records, effort) for effort in EFFORT_GRID}
    reproduction_valid = _reproduction_valid(records, summaries)
    decision = reduce_localization(summaries, reproduction_valid=reproduction_valid)

    payload: dict[str, Any] = {
        "schema": "EXP321-AFIXED-FAILURE-LOCALIZATION-EVIDENCE-V1",
        "preregistration_digest": preregistration_digest(),
        "source_receipt_artifact_digest": receipt.artifact_digest,
        "model_state_digest": actual_model_digest,
        "reproduction_valid": reproduction_valid,
        "effort_summaries": {
            str(effort): asdict(summary) for effort, summary in summaries.items()
        },
        "family_exact_effort_4": _family_exact(records, 4),
        "records": [asdict(item) for item in records],
        "decision": decision,
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    return payload
