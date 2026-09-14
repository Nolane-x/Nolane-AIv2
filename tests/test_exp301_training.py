from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp301_training import (
    EXP301_LR_CANDIDATES,
    EXP301_TRAINING_LOOPS,
    Exp301ByteTokenizer,
    build_training_contract,
    compute_answer_only_loss,
    effort_for_training_step,
    training_receipt_for_arm,
)
from nolane_ai.experiments.exp301_worlds import generate_world_instance


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")


def test_training_contract_freezes_same_search_budget_for_all_arms() -> None:
    contract = build_training_contract()

    assert contract.tokenizer_version == "exp301-byte-v1"
    assert contract.optimizer_family == "AdamW"
    assert contract.lr_candidates == EXP301_LR_CANDIDATES == (1e-4, 3e-4)
    assert contract.weight_decay == pytest.approx(0.01)
    assert contract.search_trials_per_arm == len(EXP301_LR_CANDIDATES)
    assert contract.checkpoint_selection_rule == "highest development verified success; ties -> lower LR"
    assert contract.loop_training_distribution == EXP301_TRAINING_LOOPS == (1, 2, 4, 8)
    assert contract.training_roots == (0, 1, 2, 3)


def test_training_receipts_share_every_non_architecture_field() -> None:
    receipts = [training_receipt_for_arm(arm, root=2) for arm in ARMS]
    shared = [receipt.shared_contract_digest for receipt in receipts]
    assert len(set(shared)) == 1

    for receipt in receipts:
        assert receipt.root == 2
        assert receipt.arm_id in ARMS
        assert receipt.training_data_order_digest
        assert receipt.tokenizer_version == "exp301-byte-v1"
        assert receipt.search_trials == 2


def test_training_effort_schedule_is_deterministic_and_cycles_frozen_distribution() -> None:
    assert tuple(effort_for_training_step(step) for step in range(8)) == (1, 2, 4, 8, 1, 2, 4, 8)


def test_byte_tokenizer_is_lossless_for_world_prompt_and_answer_and_within_vocab() -> None:
    tokenizer = Exp301ByteTokenizer()
    world = generate_world_instance(
        family="algorithmic-sequence-transform",
        root=0,
        split="development",
        index=1,
    )

    encoded = tokenizer.encode_example(world.model_input, world.canonical_answer)
    assert max(encoded.token_ids) < 4_608
    assert encoded.answer_start > 0
    assert tokenizer.decode(encoded.token_ids[encoded.answer_start:-1]) == world.canonical_answer


def test_answer_only_loss_masks_prompt_positions() -> None:
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example("prompt", "42")
    vocab = 4_608
    timesteps = len(encoded.token_ids)
    logits = torch.zeros(1, timesteps - 1, vocab, requires_grad=True)
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long)

    loss = compute_answer_only_loss(
        logits,
        targets=targets,
        answer_start=encoded.answer_start,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

    # The first target positions belong to the prompt and must contribute no gradient.
    masked_prefix = max(0, encoded.answer_start - 1)
    if masked_prefix:
        assert torch.count_nonzero(logits.grad[:, :masked_prefix]).item() == 0
    assert torch.count_nonzero(logits.grad[:, masked_prefix:]).item() > 0


def test_training_contract_rejects_unknown_arm_or_root() -> None:
    with pytest.raises(ValueError, match="arm"):
        training_receipt_for_arm("UNKNOWN", root=0)
    with pytest.raises(ValueError, match="root"):
        training_receipt_for_arm("A_FIXED", root=9)
