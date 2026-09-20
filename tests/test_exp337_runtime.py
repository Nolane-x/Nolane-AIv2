from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nolane_ai.experiments.exp301_training import Exp301ByteTokenizer
from nolane_ai.experiments.exp337_runtime import _combine_grads, _self_rollin_tensors


@dataclass
class _World:
    model_input: str
    canonical_answer: str


class _DeterministicNextTokenModel(nn.Module):
    def __init__(self, *, prefix_len: int, generated_ids: list[int]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.prefix_len = int(prefix_len)
        self.generated_ids = list(generated_ids)

    def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
        assert loops in (1, 2, 4, 8)
        batch, time = tokens.shape
        logits = torch.zeros(
            batch,
            time,
            Exp301ByteTokenizer.vocab_size,
            dtype=torch.float32,
            device=tokens.device,
        )
        generated_count = max(0, time - self.prefix_len)
        index = min(generated_count, len(self.generated_ids) - 1)
        logits[:, -1, self.generated_ids[index]] = 10.0 + self.anchor * 0.0
        return logits


@dataclass
class _Compiled:
    model: nn.Module
    arm_id: str = "C_NRS_CORE"


def _prefix_len(prompt: str) -> int:
    tok = Exp301ByteTokenizer()
    return 1 + len(tok.encode_text(prompt)) + 1


def test_self_rollin_tensor_shift_matches_canonical_targets() -> None:
    tok = Exp301ByteTokenizer()
    world = _World(model_input="p", canonical_answer="ab")
    answer_ids = list(tok.encode_text(world.canonical_answer))
    compiled = _Compiled(
        _DeterministicNextTokenModel(
            prefix_len=_prefix_len(world.model_input),
            generated_ids=answer_ids,
        )
    )

    input_ids, targets, answer_start, diverged, first_error = _self_rollin_tensors(
        compiled, world, effort=4
    )

    prefix = (
        tok.bos_id,
        *tok.encode_text(world.model_input),
        tok.separator_id,
    )
    assert input_ids.tolist()[0] == [*prefix, *answer_ids]
    assert targets.tolist()[0] == [*prefix[1:], *answer_ids, tok.eos_id]
    assert answer_start == len(prefix)
    assert diverged is False
    assert first_error is None


def test_self_rollin_does_not_stop_when_model_emits_eos() -> None:
    tok = Exp301ByteTokenizer()
    world = _World(model_input="p", canonical_answer="ab")
    answer_ids = list(tok.encode_text(world.canonical_answer))
    compiled = _Compiled(
        _DeterministicNextTokenModel(
            prefix_len=_prefix_len(world.model_input),
            generated_ids=[tok.eos_id, answer_ids[1]],
        )
    )

    input_ids, targets, answer_start, diverged, first_error = _self_rollin_tensors(
        compiled, world, effort=2
    )

    prefix = (
        tok.bos_id,
        *tok.encode_text(world.model_input),
        tok.separator_id,
    )
    assert input_ids.tolist()[0] == [*prefix, tok.eos_id, answer_ids[1]]
    assert targets.tolist()[0] == [*prefix[1:], *answer_ids, tok.eos_id]
    assert answer_start == len(prefix)
    assert diverged is True
    assert first_error == 0


def test_self_rollin_keeps_nonbyte_special_or_unused_token_ids() -> None:
    tok = Exp301ByteTokenizer()
    world = _World(model_input="q", canonical_answer="xy")
    answer_ids = list(tok.encode_text(world.canonical_answer))
    unused_id = 1000
    assert unused_id >= tok.byte_offset + 256
    compiled = _Compiled(
        _DeterministicNextTokenModel(
            prefix_len=_prefix_len(world.model_input),
            generated_ids=[unused_id, answer_ids[1]],
        )
    )

    input_ids, _, _, diverged, first_error = _self_rollin_tensors(
        compiled, world, effort=1
    )

    assert input_ids.tolist()[0][-2:] == [unused_id, answer_ids[1]]
    assert diverged is True
    assert first_error == 0


def test_combine_grads_is_exact_frozen_half_half_mixture() -> None:
    gold = (
        torch.tensor([2.0, -4.0]),
        torch.tensor([[6.0]]),
    )
    rollin = (
        torch.tensor([4.0, 2.0]),
        torch.tensor([[-2.0]]),
    )

    combined = _combine_grads(gold, rollin)

    assert torch.equal(combined[0], torch.tensor([3.0, -1.0]))
    assert torch.equal(combined[1], torch.tensor([[2.0]]))


def test_rollin_generation_restores_model_training_mode() -> None:
    tok = Exp301ByteTokenizer()
    world = _World(model_input="mode", canonical_answer="a")
    model = _DeterministicNextTokenModel(
        prefix_len=_prefix_len(world.model_input),
        generated_ids=list(tok.encode_text(world.canonical_answer)),
    )
    compiled = _Compiled(model)

    model.train()
    _self_rollin_tensors(compiled, world, effort=8)
    assert model.training is True

    model.eval()
    _self_rollin_tensors(compiled, world, effort=8)
    assert model.training is False
