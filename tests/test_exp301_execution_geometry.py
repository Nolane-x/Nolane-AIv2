from __future__ import annotations

from nolane_ai.experiments.exp301_training import (
    EXP301_CHALLENGE_PER_FAMILY,
    EXP301_DEV_PER_FAMILY,
    EXP301_MAX_GENERATION_TOKENS,
    EXP301_MAX_SEQUENCE_TOKENS,
    EXP301_TRAIN_EPOCHS,
    EXP301_TRAIN_PER_FAMILY,
    build_scientific_execution_contract,
    development_worlds_for_root,
    scientific_execution_contract_digest,
    scientific_model_init_seed,
    training_worlds_for_root,
)
from nolane_ai.experiments.exp301_worlds import EXP301_TASK_FAMILIES


def test_scientific_execution_geometry_is_explicit_and_bounded() -> None:
    contract = build_scientific_execution_contract()
    assert EXP301_TRAIN_PER_FAMILY == 128
    assert EXP301_DEV_PER_FAMILY == 64
    assert EXP301_CHALLENGE_PER_FAMILY == 128
    assert EXP301_TRAIN_EPOCHS == 1
    assert EXP301_MAX_SEQUENCE_TOKENS == 512
    assert EXP301_MAX_GENERATION_TOKENS == 96
    assert contract.training_examples_per_root == 128 * len(EXP301_TASK_FAMILIES)
    assert contract.development_examples_per_root == 64 * len(EXP301_TASK_FAMILIES)
    assert contract.challenge_examples_per_root == 128 * len(EXP301_TASK_FAMILIES)
    assert contract.optimizer_steps_per_trial == contract.training_examples_per_root
    assert contract.search_trials_per_arm == 2
    assert contract.primary_efforts == (1, 2, 4, 8)
    assert contract.challenge_efforts == (1, 2, 4, 8, 12, 16)
    assert "family-balanced" in contract.development_selection_rule
    assert len(scientific_execution_contract_digest()) == 64


def test_train_and_development_world_sets_are_frozen_disjoint_and_complete() -> None:
    train = training_worlds_for_root(2)
    development = development_worlds_for_root(2)

    assert len(train) == 128 * len(EXP301_TASK_FAMILIES)
    assert len(development) == 64 * len(EXP301_TASK_FAMILIES)
    assert {item.split for item in train} == {"train"}
    assert {item.split for item in development} == {"development"}
    assert {item.content_id for item in train}.isdisjoint({item.content_id for item in development})
    assert [item.family for item in train] == sorted(item.family for item in train)
    for family in EXP301_TASK_FAMILIES:
        assert sum(item.family == family for item in train) == 128
        assert sum(item.family == family for item in development) == 64


def test_execution_geometry_never_materializes_challenge_data() -> None:
    train = training_worlds_for_root(0)
    development = development_worlds_for_root(0)
    assert all(item.split != "challenge" for item in (*train, *development))


def test_scientific_model_init_seeds_are_deterministic_and_trial_specific() -> None:
    seeds = {
        scientific_model_init_seed(root=root, arm_id=arm, trial=trial)
        for root in range(4)
        for arm in ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
        for trial in range(2)
    }
    assert len(seeds) == 4 * 3 * 2
    assert scientific_model_init_seed(root=1, arm_id="C_NRS_CORE", trial=0) == scientific_model_init_seed(
        root=1, arm_id="C_NRS_CORE", trial=0
    )
