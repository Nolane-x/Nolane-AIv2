from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import torch
from torch import nn

from .exp301_arms import (
    CompiledExp301Arm,
    Exp301ArmId,
    build_nrs_core_arm,
    build_simple_recurrent_arm,
    compile_fixed_frontier_point,
)
from .exp301_execution import (
    EXP301_ARMS,
    EXP301_LR_CANDIDATES,
    EXP301_MAX_GENERATION_TOKENS,
    EXP301_ROOTS,
    scientific_model_init_seed,
)
from .exp301_training import Exp301ByteTokenizer


@dataclass(frozen=True, slots=True)
class ScientificTrialPlan:
    arm_id: str
    root: int
    trial_index: int
    learning_rate: float
    model_init_seed: int


@dataclass(frozen=True, slots=True)
class ScientificTrialResult:
    arm_id: str
    root: int
    learning_rate: float
    model_init_seed: int
    training_steps: int
    development_family_balanced_score: float
    development_family_scores: tuple[tuple[str, float], ...]
    checkpoint_digest: str
    trial_receipt_digest: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    candidate_answer: str
    generated_token_count: int
    stopped_on_eos: bool


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def frozen_trial_plan(*, root: int) -> tuple[ScientificTrialPlan, ...]:
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    plan: list[ScientificTrialPlan] = []
    for arm_id in EXP301_ARMS:
        for trial_index, learning_rate in enumerate(EXP301_LR_CANDIDATES):
            plan.append(
                ScientificTrialPlan(
                    arm_id=arm_id,
                    root=root,
                    trial_index=trial_index,
                    learning_rate=learning_rate,
                    model_init_seed=scientific_model_init_seed(
                        root=root,
                        arm_id=arm_id,
                        trial=trial_index,
                    ),
                )
            )
    return tuple(plan)


def build_scientific_arm(arm_id: str, *, device=None) -> CompiledExp301Arm:
    if arm_id == Exp301ArmId.A_FIXED.value:
        compiled = compile_fixed_frontier_point(loop_budget=1, device=device)
    elif arm_id == Exp301ArmId.B_LOOP_SIMPLE.value:
        compiled = build_simple_recurrent_arm(device=device)
    elif arm_id == Exp301ArmId.C_NRS_CORE.value:
        compiled = build_nrs_core_arm(device=device)
    else:
        raise ValueError(f"arm must be one of {EXP301_ARMS}")
    count = sum(parameter.numel() for parameter in compiled.model.parameters() if parameter.requires_grad)
    if count != 10_000_000:
        raise RuntimeError(f"scientific arm parameter drift: {arm_id} has {count:,}")
    return compiled


def _arm_value(compiled: object) -> str:
    arm = getattr(compiled, "arm_id")
    return arm.value if hasattr(arm, "value") else str(arm)


def forward_scientific_arm(
    compiled: object,
    tokens: torch.Tensor,
    *,
    effort: int,
) -> torch.Tensor:
    if effort not in (1, 2, 4, 8, 12, 16):
        raise ValueError("effort must be one of (1, 2, 4, 8, 12, 16)")
    arm_id = _arm_value(compiled)
    model = getattr(compiled, "model")
    if arm_id == Exp301ArmId.A_FIXED.value:
        return model(tokens, restarts=effort)
    if arm_id in (Exp301ArmId.B_LOOP_SIMPLE.value, Exp301ArmId.C_NRS_CORE.value):
        return model(tokens, loops=effort)
    raise ValueError(f"unknown scientific arm: {arm_id}")


def model_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        materialized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(materialized.dtype).encode("ascii"))
        digest.update(json.dumps(list(materialized.shape), separators=(",", ":")).encode("ascii"))
        try:
            raw = materialized.numpy().tobytes(order="C")
        except Exception:
            # numpy is normally present with the model stack; this fallback is
            # intentionally slower but keeps the digest definition total.
            raw = bytes(materialized.view(torch.uint8).reshape(-1).tolist())
        digest.update(raw)
    return digest.hexdigest()


def greedy_generate(
    compiled: object,
    *,
    prompt: str,
    effort: int,
    max_new_tokens: int = EXP301_MAX_GENERATION_TOKENS,
) -> GenerationResult:
    if max_new_tokens <= 0 or max_new_tokens > EXP301_MAX_GENERATION_TOKENS:
        raise ValueError(f"max_new_tokens must be in [1, {EXP301_MAX_GENERATION_TOKENS}]")
    tokenizer = Exp301ByteTokenizer()
    prompt_ids = (tokenizer.bos_id, *tokenizer.encode_text(prompt), tokenizer.separator_id)
    model = getattr(compiled, "model")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    stopped_on_eos = False

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = forward_scientific_arm(compiled, tokens, effort=effort)
            next_id = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_id)
            tokens = torch.cat(
                (tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)),
                dim=1,
            )
            if next_id == tokenizer.eos_id:
                stopped_on_eos = True
                break

    answer_ids: list[int] = []
    for token_id in generated:
        if token_id == tokenizer.eos_id:
            break
        if tokenizer.byte_offset <= token_id < tokenizer.byte_offset + 256:
            answer_ids.append(token_id)
        else:
            # Tokens outside the byte alphabet are invalid output, not a
            # hidden remapping into a valid answer.
            return GenerationResult(
                candidate_answer="",
                generated_token_count=len(generated),
                stopped_on_eos=stopped_on_eos,
            )
    return GenerationResult(
        candidate_answer=tokenizer.decode(answer_ids),
        generated_token_count=len(generated),
        stopped_on_eos=stopped_on_eos,
    )


def validate_trial_result(result: ScientificTrialResult) -> None:
    if result.arm_id not in EXP301_ARMS:
        raise ValueError("trial arm mismatch")
    if result.root not in EXP301_ROOTS:
        raise ValueError("trial root mismatch")
    if result.learning_rate not in EXP301_LR_CANDIDATES:
        raise ValueError("trial learning-rate mismatch")
    if result.training_steps <= 0:
        raise ValueError("trial training steps must be positive")
    if not 0.0 <= result.development_family_balanced_score <= 1.0:
        raise ValueError("development score must be in [0,1]")
    for label, value in (("checkpoint_digest", result.checkpoint_digest), ("trial_receipt_digest", result.trial_receipt_digest)):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{label} must be 64 lowercase hex characters")


def select_development_trial(results: Iterable[ScientificTrialResult]) -> ScientificTrialResult:
    materialized = tuple(results)
    if len(materialized) != len(EXP301_LR_CANDIDATES):
        raise ValueError("development selection requires exactly two frozen LR trials")
    for result in materialized:
        validate_trial_result(result)
    if len({result.arm_id for result in materialized}) != 1 or len({result.root for result in materialized}) != 1:
        raise ValueError("development selection trials must share arm and root")
    if {result.learning_rate for result in materialized} != set(EXP301_LR_CANDIDATES):
        raise ValueError("development selection must contain the frozen LR candidates")
    return max(
        materialized,
        key=lambda result: (result.development_family_balanced_score, -result.learning_rate),
    )


def trial_result_digest_payload(result: ScientificTrialResult) -> str:
    payload = asdict(result)
    payload.pop("trial_receipt_digest", None)
    return _canonical_digest(payload)
