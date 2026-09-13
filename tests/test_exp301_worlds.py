from __future__ import annotations

import pytest

from nolane_ai.experiments.exp301_worlds import (
    EXP301_TASK_FAMILIES,
    generate_world_instance,
    materialize_world_set,
    metamorphic_variant,
    verify_world_answer,
)


EXPECTED_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)


def test_exp301_task_families_are_frozen() -> None:
    assert EXP301_TASK_FAMILIES == EXPECTED_FAMILIES


def test_world_generation_is_deterministic_and_has_stable_identity() -> None:
    left = generate_world_instance(
        family="algorithmic-sequence-transform",
        root=2,
        split="development",
        index=7,
    )
    right = generate_world_instance(
        family="algorithmic-sequence-transform",
        root=2,
        split="development",
        index=7,
    )

    assert left == right
    assert len(left.content_id) == 64
    assert len(left.generator_identity) == 64
    assert left.generator_version == "exp301-worlds-v1"
    assert left.required_steps >= 1
    assert left.difficulty >= 1


def test_train_development_and_test_only_challenge_are_generator_disjoint() -> None:
    train = materialize_world_set(split="train", roots=(0, 1), per_family=3)
    development = materialize_world_set(split="development", roots=(0, 1), per_family=3)
    challenge = materialize_world_set(
        split="challenge",
        roots=(0, 1),
        per_family=3,
        challenge_nonce="TEST-ONLY-POST-FREEZE-NONCE",
        test_only=True,
    )

    train_ids = {item.content_id for item in train}
    development_ids = {item.content_id for item in development}
    challenge_ids = {item.content_id for item in challenge}
    assert train_ids.isdisjoint(development_ids)
    assert train_ids.isdisjoint(challenge_ids)
    assert development_ids.isdisjoint(challenge_ids)

    train_generators = {item.generator_identity for item in train}
    development_generators = {item.generator_identity for item in development}
    challenge_generators = {item.generator_identity for item in challenge}
    assert train_generators.isdisjoint(development_generators)
    assert train_generators.isdisjoint(challenge_generators)
    assert development_generators.isdisjoint(challenge_generators)


def test_scientific_challenge_requires_post_freeze_nonce() -> None:
    with pytest.raises(ValueError, match="challenge_nonce"):
        materialize_world_set(split="challenge", roots=(0,), per_family=1)


def test_each_family_has_an_exact_verifier() -> None:
    for family in EXPECTED_FAMILIES:
        instance = generate_world_instance(
            family=family,
            root=1,
            split="development",
            index=3,
        )
        assert verify_world_answer(instance, instance.canonical_answer) is True
        assert verify_world_answer(instance, instance.canonical_answer + "__wrong") is False


def test_metamorphic_variant_changes_surface_but_preserves_solution() -> None:
    for family in EXPECTED_FAMILIES:
        original = generate_world_instance(
            family=family,
            root=3,
            split="development",
            index=5,
        )
        variant = metamorphic_variant(original, variant_index=2)

        assert variant.family == original.family
        assert variant.canonical_answer == original.canonical_answer
        assert variant.model_input != original.model_input
        assert variant.content_id != original.content_id
        assert verify_world_answer(variant, original.canonical_answer) is True


def test_model_input_never_exposes_private_generator_metadata() -> None:
    challenge = materialize_world_set(
        split="challenge",
        roots=(0,),
        per_family=2,
        challenge_nonce="TEST-ONLY-SECRET-NONCE",
        test_only=True,
    )

    for item in challenge:
        assert item.generator_identity not in item.model_input
        assert item.template_identity not in item.model_input
        assert item.split not in item.model_input
        assert str(item.root) not in item.model_input
        assert "TEST-ONLY-SECRET-NONCE" not in item.model_input
