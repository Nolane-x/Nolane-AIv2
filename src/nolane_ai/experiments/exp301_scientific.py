from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
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
    EXP301_MAX_SEQUENCE_TOKENS,
    EXP301_ROOTS,
    EXP301_TRAINING_LOOPS,
    development_worlds_for_root,
    scientific_model_init_seed,
    training_worlds_for_root,
)
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss, effort_for_training_step
from .exp301_worlds import Exp301WorldInstance, verify_world_answer


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


def _validate_plan(plan: ScientificTrialPlan) -> None:
    if plan.root not in EXP301_ROOTS or plan.arm_id not in EXP301_ARMS:
        raise ValueError("scientific trial plan root/arm mismatch")
    expected = {
        (item.arm_id, item.trial_index): item
        for item in frozen_trial_plan(root=plan.root)
    }.get((plan.arm_id, plan.trial_index))
    if expected != plan:
        raise ValueError("scientific trial plan does not match frozen root/arm/LR/seed authority")


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


def forward_scientific_arm(compiled: object, tokens: torch.Tensor, *, effort: int) -> torch.Tensor:
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
    if len(prompt_ids) + max_new_tokens > EXP301_MAX_SEQUENCE_TOKENS:
        raise ValueError("prompt plus frozen generation cap exceeds max sequence tokens")
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
            tokens = torch.cat((tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)
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
            return GenerationResult("", len(generated), stopped_on_eos)
    return GenerationResult(tokenizer.decode(answer_ids), len(generated), stopped_on_eos)


def family_balanced_development_score(
    success_counts: dict[str, tuple[int, int]],
) -> tuple[float, tuple[tuple[str, float], ...]]:
    if not success_counts:
        raise ValueError("development family counts cannot be empty")
    scores: list[tuple[str, float]] = []
    for family in sorted(success_counts):
        successes, total = success_counts[family]
        if total <= 0 or successes < 0 or successes > total:
            raise ValueError("invalid development success counts")
        scores.append((family, successes / total))
    return sum(score for _, score in scores) / len(scores), tuple(scores)


def scientific_train_step(
    compiled: object,
    world: Exp301WorldInstance,
    *,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> float:
    if world.split != "train":
        raise ValueError("scientific training accepts train worlds only")
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example(world.model_input, world.canonical_answer)
    if len(encoded.token_ids) > EXP301_MAX_SEQUENCE_TOKENS:
        raise ValueError("training example exceeds frozen max sequence tokens")
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded.token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = forward_scientific_arm(compiled, input_ids, effort=effort_for_training_step(step))
    loss = compute_answer_only_loss(logits, targets=targets, answer_start=encoded.answer_start)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def _evaluate_development(compiled: object, *, root: int) -> tuple[float, tuple[tuple[str, float], ...]]:
    counts: dict[str, list[int]] = {}
    for world in development_worlds_for_root(root):
        bucket = counts.setdefault(world.family, [0, 0])
        for effort in EXP301_TRAINING_LOOPS:
            generation = greedy_generate(compiled, prompt=world.model_input, effort=effort)
            bucket[0] += int(verify_world_answer(world, generation.candidate_answer))
            bucket[1] += 1
    return family_balanced_development_score(
        {family: (values[0], values[1]) for family, values in counts.items()}
    )


def _checkpoint_write_once(path: Path, compiled: object, result: ScientificTrialResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(
            {
                "schema": "EXP301-SCIENTIFIC-CHECKPOINT-V1",
                "trial_result": asdict(result),
                "state_dict": getattr(compiled, "model").state_dict(),
            },
            handle,
        )


def run_scientific_trial(
    plan: ScientificTrialPlan,
    *,
    device: str | torch.device,
    checkpoint_path: str | Path,
) -> ScientificTrialResult:
    _validate_plan(plan)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)

    torch.manual_seed(plan.model_init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(plan.model_init_seed)
    compiled = build_scientific_arm(plan.arm_id, device=device)
    optimizer = torch.optim.AdamW(
        compiled.model.parameters(),
        lr=plan.learning_rate,
        weight_decay=0.01,
    )
    worlds = training_worlds_for_root(plan.root)
    if len(worlds) != 512:
        raise RuntimeError(f"frozen training world count drift: {len(worlds)}")
    for step, world in enumerate(worlds):
        scientific_train_step(compiled, world, optimizer=optimizer, step=step)

    dev_score, family_scores = _evaluate_development(compiled, root=plan.root)
    checkpoint_digest = model_state_digest(compiled.model)
    provisional = ScientificTrialResult(
        arm_id=plan.arm_id,
        root=plan.root,
        learning_rate=plan.learning_rate,
        model_init_seed=plan.model_init_seed,
        training_steps=len(worlds),
        development_family_balanced_score=dev_score,
        development_family_scores=family_scores,
        checkpoint_digest=checkpoint_digest,
        trial_receipt_digest="",
    )
    receipt_digest = trial_result_digest_payload(provisional)
    result = ScientificTrialResult(**{**asdict(provisional), "trial_receipt_digest": receipt_digest})
    _checkpoint_write_once(checkpoint_path, compiled, result)
    return result


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
    return max(materialized, key=lambda result: (result.development_family_balanced_score, -result.learning_rate))


def trial_result_digest_payload(result: ScientificTrialResult) -> str:
    payload = asdict(result)
    payload.pop("trial_receipt_digest", None)
    return _canonical_digest(payload)
