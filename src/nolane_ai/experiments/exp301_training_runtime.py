from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence

import torch
from torch import nn

from nolane_ai.experiments.exp301_training import (
    EXP301_ARMS,
    EXP301_LR_CANDIDATES,
    EXP301_ROOTS,
    Exp301ByteTokenizer,
    build_training_contract,
    compute_answer_only_loss,
    effort_for_training_step,
    shared_training_contract_digest,
)
from nolane_ai.experiments.exp301_worlds import (
    EXP301_TASK_FAMILIES,
    Exp301WorldInstance,
    generate_world_instance,
)
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


DECISION_TIE_BREAK_RULE = "highest development verified success; ties -> lower LR"
TRAINING_RUNTIME_VERSION = "exp301-training-runtime-v1"


@dataclass(frozen=True, slots=True)
class Exp301TrialPlan:
    arm_id: str
    root: int
    learning_rate: float
    trial_index: int
    max_steps: int
    token_budget: int
    test_only: bool
    scientific_evidence_eligible: bool
    shared_contract_digest: str
    runtime_version: str


@dataclass(frozen=True, slots=True)
class Exp301CheckpointReceipt:
    arm_id: str
    root: int
    trial_index: int
    learning_rate: float
    completed_steps: int
    consumed_tokens: int
    effort_schedule: tuple[int, ...]
    training_order_digest: str
    shared_contract_digest: str
    checkpoint_digest: str
    scientific_evidence_eligible: bool
    runtime_version: str


@dataclass(frozen=True, slots=True)
class Exp301TrialResult:
    plan: Exp301TrialPlan
    receipt: Exp301CheckpointReceipt
    mean_training_loss: float
    development_verified_success: float | None

    @classmethod
    def synthetic_for_selection(
        cls,
        *,
        plan: Exp301TrialPlan,
        development_verified_success: float,
    ) -> "Exp301TrialResult":
        if not 0.0 <= development_verified_success <= 1.0:
            raise ValueError("development_verified_success must be in [0,1]")
        receipt = Exp301CheckpointReceipt(
            arm_id=plan.arm_id,
            root=plan.root,
            trial_index=plan.trial_index,
            learning_rate=plan.learning_rate,
            completed_steps=0,
            consumed_tokens=0,
            effort_schedule=tuple(),
            training_order_digest="synthetic-selection-only",
            shared_contract_digest=plan.shared_contract_digest,
            checkpoint_digest="synthetic-selection-only",
            scientific_evidence_eligible=False,
            runtime_version=plan.runtime_version,
        )
        return cls(
            plan=plan,
            receipt=receipt,
            mean_training_loss=float("nan"),
            development_verified_success=development_verified_success,
        )


@dataclass(frozen=True, slots=True)
class Exp301SelectedTrial:
    plan: Exp301TrialPlan
    receipt: Exp301CheckpointReceipt
    development_verified_success: float
    selection_rule: str


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_trial_plans(
    arm_id: str,
    *,
    root: int,
    max_steps: int,
    token_budget: int,
    test_only: bool,
) -> tuple[Exp301TrialPlan, ...]:
    if arm_id not in EXP301_ARMS:
        raise ValueError(f"arm must be one of {EXP301_ARMS}")
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    if max_steps <= 0 or token_budget <= 0:
        raise ValueError("max_steps and token_budget must be positive")

    contract_digest = shared_training_contract_digest()
    return tuple(
        Exp301TrialPlan(
            arm_id=arm_id,
            root=root,
            learning_rate=learning_rate,
            trial_index=trial_index,
            max_steps=max_steps,
            token_budget=token_budget,
            test_only=bool(test_only),
            scientific_evidence_eligible=not bool(test_only),
            shared_contract_digest=contract_digest,
            runtime_version=TRAINING_RUNTIME_VERSION,
        )
        for trial_index, learning_rate in enumerate(EXP301_LR_CANDIDATES)
    )


def _training_order_digest(examples: Sequence[Exp301WorldInstance]) -> str:
    return _canonical_digest(
        {
            "runtime_version": TRAINING_RUNTIME_VERSION,
            "ordering": "family-then-content-id",
            "content_ids": [example.content_id for example in examples],
        }
    )


def deterministic_training_examples(
    *,
    root: int,
    per_family: int,
) -> tuple[tuple[Exp301WorldInstance, ...], str]:
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    if per_family <= 0:
        raise ValueError("per_family must be positive")

    examples = [
        generate_world_instance(
            family=family,
            root=root,
            split="train",
            index=index,
        )
        for family in EXP301_TASK_FAMILIES
        for index in range(per_family)
    ]
    examples.sort(key=lambda example: (example.family, example.content_id))
    frozen = tuple(examples)
    return frozen, _training_order_digest(frozen)


def checkpoint_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(f"{TRAINING_RUNTIME_VERSION}|byteorder={tensor_byteorder()}\n".encode("utf-8"))
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        metadata = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_raw_bytes(tensor))
        digest.update(b"\0")
    return digest.hexdigest()


def _forward_for_arm(
    model: nn.Module,
    *,
    arm_id: str,
    tokens: torch.Tensor,
    effort: int,
) -> torch.Tensor:
    if arm_id == "A_FIXED":
        return model(tokens, restarts=effort)
    if arm_id in ("B_LOOP_SIMPLE", "C_NRS_CORE"):
        return model(tokens, loops=effort)
    raise ValueError(f"arm must be one of {EXP301_ARMS}")


def run_training_trial(
    model: nn.Module,
    *,
    plan: Exp301TrialPlan,
    training_examples: Iterable[Exp301WorldInstance],
    training_order_digest: str,
) -> Exp301TrialResult:
    if plan.arm_id not in EXP301_ARMS or plan.root not in EXP301_ROOTS:
        raise ValueError("trial plan arm/root is invalid")
    if plan.shared_contract_digest != shared_training_contract_digest():
        raise ValueError("trial plan training contract digest drifted")

    examples = tuple(training_examples)
    if not examples:
        raise ValueError("training_examples must be non-empty")
    if any(example.root != plan.root or example.split != "train" for example in examples):
        raise ValueError("training examples must belong to the plan root and train split")
    expected_order_digest = _training_order_digest(examples)
    if training_order_digest != expected_order_digest:
        raise ValueError("training order digest mismatch")

    contract = build_training_contract()
    tokenizer = Exp301ByteTokenizer()
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must have trainable parameters") from exc

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=plan.learning_rate,
        weight_decay=contract.weight_decay,
    )
    model.train()
    losses: list[float] = []
    consumed_tokens = 0
    efforts: list[int] = []

    for step in range(plan.max_steps):
        example = examples[step % len(examples)]
        encoded = tokenizer.encode_example(example.model_input, example.canonical_answer)
        full = torch.tensor(encoded.token_ids, dtype=torch.long, device=device).unsqueeze(0)
        inputs = full[:, :-1]
        targets = full[:, 1:]
        step_tokens = int(inputs.numel())
        if consumed_tokens + step_tokens > plan.token_budget:
            break

        effort = effort_for_training_step(step)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_for_arm(
            model,
            arm_id=plan.arm_id,
            tokens=inputs,
            effort=effort,
        )
        loss = compute_answer_only_loss(
            logits,
            targets=targets,
            answer_start=encoded.answer_start,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite EXP-301 training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), contract.gradient_clip_norm)
        optimizer.step()

        consumed_tokens += step_tokens
        efforts.append(effort)
        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise ValueError("token budget was too small for one training step")

    receipt = Exp301CheckpointReceipt(
        arm_id=plan.arm_id,
        root=plan.root,
        trial_index=plan.trial_index,
        learning_rate=plan.learning_rate,
        completed_steps=len(losses),
        consumed_tokens=consumed_tokens,
        effort_schedule=tuple(efforts),
        training_order_digest=training_order_digest,
        shared_contract_digest=plan.shared_contract_digest,
        checkpoint_digest=checkpoint_state_digest(model),
        scientific_evidence_eligible=plan.scientific_evidence_eligible,
        runtime_version=TRAINING_RUNTIME_VERSION,
    )
    return Exp301TrialResult(
        plan=plan,
        receipt=receipt,
        mean_training_loss=sum(losses) / len(losses),
        development_verified_success=None,
    )


def select_best_trial(results: Sequence[Exp301TrialResult]) -> Exp301SelectedTrial:
    if not results:
        raise ValueError("trial selection requires results")
    arm_ids = {result.plan.arm_id for result in results}
    roots = {result.plan.root for result in results}
    if len(arm_ids) != 1 or len(roots) != 1:
        raise ValueError("trial selection cannot mix arms or roots")
    if len(results) != len(EXP301_LR_CANDIDATES):
        raise ValueError("trial selection requires the full frozen LR search budget")
    if {result.plan.learning_rate for result in results} != set(EXP301_LR_CANDIDATES):
        raise ValueError("trial selection LR candidates drifted")
    if any(result.development_verified_success is None for result in results):
        raise ValueError("every trial needs a frozen development score before selection")

    selected = min(
        results,
        key=lambda result: (
            -float(result.development_verified_success),
            result.plan.learning_rate,
            result.plan.trial_index,
        ),
    )
    return Exp301SelectedTrial(
        plan=selected.plan,
        receipt=selected.receipt,
        development_verified_success=float(selected.development_verified_success),
        selection_rule=DECISION_TIE_BREAK_RULE,
    )
