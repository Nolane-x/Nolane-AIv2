from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
from typing import Iterable


WORLD_GENERATOR_VERSION = "exp301-worlds-v1"
EXP301_TASK_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
_VALID_SPLITS = ("train", "development", "challenge")


@dataclass(frozen=True, slots=True)
class Exp301WorldInstance:
    family: str
    generator_version: str
    generator_identity: str
    template_identity: str
    content_id: str
    root: int
    split: str
    index: int
    difficulty: int
    required_steps: int
    model_input: str
    canonical_answer: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _split_salt(*, split: str, challenge_nonce: str | None) -> str:
    if split == "challenge":
        if not challenge_nonce:
            raise ValueError("challenge_nonce is required for challenge materialization")
        return _sha256_text(f"challenge|{challenge_nonce}")
    return _sha256_text(f"exp301-public-{split}-generator-salt-v1")


def _seed_for(
    *,
    family: str,
    root: int,
    split: str,
    index: int,
    challenge_nonce: str | None,
) -> int:
    salt = _split_salt(split=split, challenge_nonce=challenge_nonce)
    digest = _sha256_text(
        f"{WORLD_GENERATOR_VERSION}|{family}|{root}|{split}|{index}|{salt}"
    )
    return int(digest[:16], 16)


def _identities(
    *,
    family: str,
    split: str,
    challenge_nonce: str | None,
) -> tuple[str, str]:
    salt = _split_salt(split=split, challenge_nonce=challenge_nonce)
    generator_identity = _sha256_text(
        f"{WORLD_GENERATOR_VERSION}|generator|{family}|{split}|{salt}"
    )
    # Template identity is intentionally split-specific. This is stronger than
    # random-example holdout: challenge surface/template families cannot be
    # identical to train/development template authorities.
    template_identity = _sha256_text(
        f"{WORLD_GENERATOR_VERSION}|template|{family}|{split}|{salt}"
    )
    return generator_identity, template_identity


def _maze_world(rng: random.Random, *, difficulty: int) -> tuple[str, str, int]:
    width = 7 + 2 * difficulty
    start = rng.randrange(1, width - 1)
    steps = 2 + difficulty
    position = start
    moves: list[str] = []
    for _ in range(steps):
        candidates = []
        if position > 0:
            candidates.append(("L", -1))
        if position < width - 1:
            candidates.append(("R", 1))
        move, delta = rng.choice(candidates)
        moves.append(move)
        position += delta
    prompt = (
        f"A token is on a one-dimensional corridor with cells 0 through {width - 1}. "
        f"It starts at cell {start}. Apply these moves in order, L=-1 and R=+1: "
        f"{' '.join(moves)}. Return only the final cell number."
    )
    return prompt, str(position), steps


def _sequence_world(rng: random.Random, *, difficulty: int) -> tuple[str, str, int]:
    length = 4 + difficulty
    values = [rng.randrange(0, 10) for _ in range(length)]
    shift = 1 + rng.randrange(max(1, length - 1))
    add = rng.randrange(1, 5)
    transformed = list(reversed(values))
    shift %= length
    transformed = transformed[shift:] + transformed[:shift]
    transformed = [(value + add) % 10 for value in transformed]
    prompt = (
        "Transform the digit sequence exactly: first reverse it; then rotate left by "
        f"{shift}; then add {add} modulo 10 to every digit. Input: "
        + " ".join(map(str, values))
        + ". Return digits separated by single spaces."
    )
    return prompt, " ".join(map(str, transformed)), 3


def _abstract_world(
    rng: random.Random,
    *,
    difficulty: int,
    split: str,
) -> tuple[str, str, int]:
    # Split-specific operation families provide generator-level rather than
    # random-example-level holdout. The rule remains fully inferable from
    # demonstrations in each instance; no private split ID is exposed.
    split_offset = {"train": 1, "development": 2, "challenge": 3}[split]
    modulus = 7
    multiplier = (2 + split_offset) % modulus
    if multiplier == 0:
        multiplier = 2
    bias = rng.randrange(1, modulus)
    demos: list[str] = []
    for value in range(3):
        answer = (multiplier * value + bias) % modulus
        demos.append(f"{value}->{answer}")
    query = rng.randrange(3, 3 + difficulty + 2)
    canonical = (multiplier * query + bias) % modulus
    prompt = (
        "Infer the deterministic transformation from the examples and apply it to the query. "
        f"Examples: {', '.join(demos)}. Query: {query}->?. Return only the missing integer."
    )
    return prompt, str(canonical), 2 + difficulty


def _language_control_world(rng: random.Random, *, difficulty: int) -> tuple[str, str, int]:
    words = ["amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "harbor"]
    count = min(len(words), 3 + difficulty)
    chosen = rng.sample(words, count)
    mode = rng.choice(("alphabetical", "reverse"))
    if mode == "alphabetical":
        answer_tokens = sorted(chosen)
        instruction = "Sort the words alphabetically"
    else:
        answer_tokens = list(reversed(chosen))
        instruction = "Reverse the word order"
    prompt = (
        f"{instruction}. Words: {' | '.join(chosen)}. "
        "Return only the resulting words separated by a single space."
    )
    return prompt, " ".join(answer_tokens), 1


def generate_world_instance(
    *,
    family: str,
    root: int,
    split: str,
    index: int,
    challenge_nonce: str | None = None,
) -> Exp301WorldInstance:
    if family not in EXP301_TASK_FAMILIES:
        raise ValueError(f"unknown EXP-301 family: {family!r}")
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}")
    if root < 0 or index < 0:
        raise ValueError("root and index must be non-negative")
    if split == "challenge" and not challenge_nonce:
        raise ValueError("challenge_nonce is required for challenge materialization")

    seed = _seed_for(
        family=family,
        root=root,
        split=split,
        index=index,
        challenge_nonce=challenge_nonce,
    )
    rng = random.Random(seed)
    difficulty = 1 + (index + root) % 4

    if family == "iterative-grid-and-maze":
        model_input, answer, required_steps = _maze_world(rng, difficulty=difficulty)
    elif family == "algorithmic-sequence-transform":
        model_input, answer, required_steps = _sequence_world(rng, difficulty=difficulty)
    elif family == "generator-heldout-abstract-transformation":
        model_input, answer, required_steps = _abstract_world(
            rng,
            difficulty=difficulty,
            split=split,
        )
    else:
        model_input, answer, required_steps = _language_control_world(rng, difficulty=difficulty)

    generator_identity, template_identity = _identities(
        family=family,
        split=split,
        challenge_nonce=challenge_nonce,
    )
    content_id = _canonical_digest(
        {
            "version": WORLD_GENERATOR_VERSION,
            "family": family,
            "generator_identity": generator_identity,
            "template_identity": template_identity,
            "root": root,
            "split": split,
            "index": index,
            "difficulty": difficulty,
            "required_steps": required_steps,
            "model_input": model_input,
            "canonical_answer": answer,
        }
    )
    return Exp301WorldInstance(
        family=family,
        generator_version=WORLD_GENERATOR_VERSION,
        generator_identity=generator_identity,
        template_identity=template_identity,
        content_id=content_id,
        root=root,
        split=split,
        index=index,
        difficulty=difficulty,
        required_steps=required_steps,
        model_input=model_input,
        canonical_answer=answer,
    )


def materialize_world_set(
    *,
    split: str,
    roots: Iterable[int],
    per_family: int,
    challenge_nonce: str | None = None,
    test_only: bool = False,
) -> tuple[Exp301WorldInstance, ...]:
    del test_only  # eligibility is recorded by the runner; generation stays identical.
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    if split == "challenge" and not challenge_nonce:
        raise ValueError("challenge_nonce is required for challenge materialization")

    instances = []
    for root in tuple(roots):
        for family in EXP301_TASK_FAMILIES:
            for index in range(per_family):
                instances.append(
                    generate_world_instance(
                        family=family,
                        root=root,
                        split=split,
                        index=index,
                        challenge_nonce=challenge_nonce,
                    )
                )
    return tuple(instances)


def verify_world_answer(instance: Exp301WorldInstance, candidate: str) -> bool:
    # Exact verifier: no fuzzy grading or model-side authority.
    return candidate.strip() == instance.canonical_answer


def metamorphic_variant(
    instance: Exp301WorldInstance,
    *,
    variant_index: int,
) -> Exp301WorldInstance:
    if variant_index < 0:
        raise ValueError("variant_index must be non-negative")
    surface_tag = _sha256_text(
        f"metamorphic|{instance.content_id}|{variant_index}"
    )[:8]
    model_input = (
        instance.model_input
        + f" Surface-note {surface_tag}: formatting noise only; solve the original task."
    )
    content_id = _canonical_digest(
        {
            "source_content_id": instance.content_id,
            "variant_index": variant_index,
            "model_input": model_input,
            "canonical_answer": instance.canonical_answer,
        }
    )
    return replace(instance, content_id=content_id, model_input=model_input)
