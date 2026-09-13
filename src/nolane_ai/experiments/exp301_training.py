from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch.nn import functional as F


EXP301_TRAINING_LOOPS = (1, 2, 4, 8)
EXP301_LR_CANDIDATES = (1e-4, 3e-4)
EXP301_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EXP301_ROOTS = (0, 1, 2, 3)
TOKENIZER_VERSION = "exp301-byte-v1"
TRAINING_ORDER_VERSION = "exp301-data-order-v1"


@dataclass(frozen=True, slots=True)
class Exp301TrainingContract:
    tokenizer_version: str
    vocab_size: int
    optimizer_family: str
    lr_candidates: tuple[float, ...]
    weight_decay: float
    search_trials_per_arm: int
    checkpoint_selection_rule: str
    loop_training_distribution: tuple[int, ...]
    training_roots: tuple[int, ...]
    training_order_version: str
    gradient_clip_norm: float


@dataclass(frozen=True, slots=True)
class Exp301TrainingReceipt:
    arm_id: str
    root: int
    shared_contract_digest: str
    training_data_order_digest: str
    tokenizer_version: str
    search_trials: int


@dataclass(frozen=True, slots=True)
class EncodedExample:
    token_ids: tuple[int, ...]
    answer_start: int


class Exp301ByteTokenizer:
    """Frozen, reversible UTF-8 byte tokenizer shared by every EXP-301 arm.

    IDs 0..3 are control tokens. Bytes 0..255 map to 4..259. The remaining
    vocabulary slots are intentionally unused token IDs, not trainable padding:
    they are rows of the same matched embedding/output matrix in every arm.
    """

    pad_id = 0
    bos_id = 1
    separator_id = 2
    eos_id = 3
    byte_offset = 4
    vocab_size = 4_608
    version = TOKENIZER_VERSION

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(self.byte_offset + byte for byte in text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        raw = bytearray()
        for token_id in token_ids:
            if token_id in (self.pad_id, self.bos_id, self.separator_id, self.eos_id):
                continue
            if not self.byte_offset <= token_id < self.byte_offset + 256:
                raise ValueError(f"token id {token_id} is not a byte token")
            raw.append(token_id - self.byte_offset)
        return bytes(raw).decode("utf-8")

    def encode_example(self, prompt: str, answer: str) -> EncodedExample:
        prompt_ids = self.encode_text(prompt)
        answer_ids = self.encode_text(answer)
        token_ids = (
            self.bos_id,
            *prompt_ids,
            self.separator_id,
            *answer_ids,
            self.eos_id,
        )
        answer_start = 1 + len(prompt_ids) + 1
        return EncodedExample(token_ids=tuple(token_ids), answer_start=answer_start)


def build_training_contract() -> Exp301TrainingContract:
    return Exp301TrainingContract(
        tokenizer_version=TOKENIZER_VERSION,
        vocab_size=4_608,
        optimizer_family="AdamW",
        lr_candidates=EXP301_LR_CANDIDATES,
        weight_decay=0.01,
        search_trials_per_arm=len(EXP301_LR_CANDIDATES),
        checkpoint_selection_rule="highest development verified success; ties -> lower LR",
        loop_training_distribution=EXP301_TRAINING_LOOPS,
        training_roots=EXP301_ROOTS,
        training_order_version=TRAINING_ORDER_VERSION,
        gradient_clip_norm=1.0,
    )


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def shared_training_contract_digest() -> str:
    return _canonical_digest(asdict(build_training_contract()))


def training_receipt_for_arm(arm_id: str, *, root: int) -> Exp301TrainingReceipt:
    if arm_id not in EXP301_ARMS:
        raise ValueError(f"arm must be one of {EXP301_ARMS}")
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    contract = build_training_contract()
    data_order_digest = _canonical_digest(
        {
            "version": contract.training_order_version,
            "root": root,
            "ordering": "family-then-content-id",
            "shared_across_arms": True,
        }
    )
    return Exp301TrainingReceipt(
        arm_id=arm_id,
        root=root,
        shared_contract_digest=shared_training_contract_digest(),
        training_data_order_digest=data_order_digest,
        tokenizer_version=contract.tokenizer_version,
        search_trials=contract.search_trials_per_arm,
    )


def effort_for_training_step(step: int) -> int:
    if step < 0:
        raise ValueError("step must be non-negative")
    return EXP301_TRAINING_LOOPS[step % len(EXP301_TRAINING_LOOPS)]


def compute_answer_only_loss(
    logits: torch.Tensor,
    *,
    targets: torch.Tensor,
    answer_start: int,
) -> torch.Tensor:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be [batch,time,vocab] and targets [batch,time]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits/targets time geometry mismatch")
    if answer_start <= 0:
        raise ValueError("answer_start must be positive")

    # logits[t] predicts full-sequence token t+1. If the first answer token is
    # at full-sequence index answer_start, its prediction lives at target index
    # answer_start-1. EOS is intentionally part of the answer objective.
    first_target_index = answer_start - 1
    if first_target_index >= targets.shape[1]:
        raise ValueError("answer_start leaves no answer targets")
    return F.cross_entropy(
        logits[:, first_target_index:, :].reshape(-1, logits.shape[-1]),
        targets[:, first_target_index:].reshape(-1),
    )
