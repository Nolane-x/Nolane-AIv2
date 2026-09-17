from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterable, Mapping


WORLD_GENERATOR_VERSION = "exp319-worlds-v1"
EXP319_TASK_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
VALID_STAGES = ("A_SANITY", "B_TRAIN", "B_IID", "C_HELDOUT")
_STAGE_B_ROOTS = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class Exp319WorldInstance:
    family: str
    generator_version: str
    generator_identity: str
    template_identity: str
    content_id: str
    stage: str
    root: int
    index: int
    complexity: int
    required_steps: int
    model_input: str
    canonical_answer: str
    challenge_beacon: str | None = None
    stage_b_selection_digest: str | None = None
    operation_trace: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "generator_version": self.generator_version,
            "generator_identity": self.generator_identity,
            "template_identity": self.template_identity,
            "content_id": self.content_id,
            "stage": self.stage,
            "root": self.root,
            "index": self.index,
            "complexity": self.complexity,
            "required_steps": self.required_steps,
            "model_input": self.model_input,
            "canonical_answer": self.canonical_answer,
            "challenge_beacon": self.challenge_beacon,
            "stage_b_selection_digest": self.stage_b_selection_digest,
            "operation_trace": list(self.operation_trace),
            "tags": list(self.tags),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "Exp319WorldInstance":
        return cls(
            family=str(payload["family"]),
            generator_version=str(payload["generator_version"]),
            generator_identity=str(payload["generator_identity"]),
            template_identity=str(payload["template_identity"]),
            content_id=str(payload["content_id"]),
            stage=str(payload["stage"]),
            root=int(payload["root"]),
            index=int(payload["index"]),
            complexity=int(payload["complexity"]),
            required_steps=int(payload["required_steps"]),
            model_input=str(payload["model_input"]),
            canonical_answer=str(payload["canonical_answer"]),
            challenge_beacon=(None if payload.get("challenge_beacon") is None else str(payload["challenge_beacon"])),
            stage_b_selection_digest=(None if payload.get("stage_b_selection_digest") is None else str(payload["stage_b_selection_digest"])),
            operation_trace=tuple(str(item) for item in payload.get("operation_trace", [])),
            tags=tuple(str(item) for item in payload.get("tags", [])),
        )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(payload: str | bytes) -> str:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(raw).hexdigest()


def _validate_stage_authority(*, stage: str, root: int, index: int, challenge_beacon: str | None, stage_b_selection_digest: str | None) -> None:
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}")
    if index < 0:
        raise ValueError("index must be non-negative")
    if stage == "A_SANITY":
        if root != 0:
            raise ValueError("A_SANITY is frozen to root=0")
        if challenge_beacon is not None or stage_b_selection_digest is not None:
            raise ValueError("A_SANITY cannot carry challenge authority")
        return
    if root not in _STAGE_B_ROOTS:
        raise ValueError("Stage B/C roots must be one of (1, 2, 3, 4)")
    if stage in ("B_TRAIN", "B_IID"):
        if challenge_beacon is not None or stage_b_selection_digest is not None:
            raise ValueError("Stage B cannot carry heldout challenge authority")
        return
    if not challenge_beacon:
        raise ValueError("challenge_beacon is required for C_HELDOUT")
    if "exp301" in challenge_beacon.lower():
        raise ValueError("EXP-301 challenge IDs/nonces are forbidden in EXP-319")
    if not stage_b_selection_digest:
        raise ValueError("stage_b_selection_digest is required for C_HELDOUT")
    if len(stage_b_selection_digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in stage_b_selection_digest):
        raise ValueError("stage_b_selection_digest must be a 64-character SHA-256 hex digest")


def _authority_token(*, stage: str, challenge_beacon: str | None, stage_b_selection_digest: str | None) -> str:
    if stage == "C_HELDOUT":
        return _sha256(f"{challenge_beacon}|{stage_b_selection_digest}")
    return _sha256(f"exp319-public|{stage}")


def _identities(*, family: str, stage: str, root: int, challenge_beacon: str | None, stage_b_selection_digest: str | None) -> tuple[str, str]:
    authority = _authority_token(stage=stage, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest)
    generator_identity = _sha256(f"{WORLD_GENERATOR_VERSION}|generator|{family}|{stage}|root={root}|{authority}")
    template_family = {
        "A_SANITY": "sanity-template-v1",
        "B_TRAIN": "iid-template-v1-train",
        "B_IID": "iid-template-v1-development",
        "C_HELDOUT": "heldout-template-v1-shifted",
    }[stage]
    template_identity = _sha256(f"{WORLD_GENERATOR_VERSION}|template|{family}|{template_family}|root={root}|{authority}")
    return generator_identity, template_identity


def _seed_for(*, family: str, stage: str, root: int, index: int, challenge_beacon: str | None, stage_b_selection_digest: str | None) -> int:
    authority = _authority_token(stage=stage, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest)
    digest = _sha256(f"{WORLD_GENERATOR_VERSION}|{family}|{stage}|{root}|{index}|{authority}")
    return int(digest[:16], 16)


def _complexity_for(*, stage: str, root: int, index: int) -> int:
    base = {"A_SANITY": 1, "B_TRAIN": 3, "B_IID": 3, "C_HELDOUT": 5}[stage]
    return base + ((root + index) % 3)


def _iterative_world(rng: random.Random, *, stage: str, complexity: int) -> tuple[str, str, tuple[str, ...]]:
    if stage == "A_SANITY":
        width = 5 + complexity
        steps = 2 + (complexity % 2)
    elif stage in ("B_TRAIN", "B_IID"):
        width = 9 + complexity
        steps = 4 + complexity
    else:
        width = 13 + complexity
        steps = 7 + complexity
    position = rng.randrange(1, width - 1)
    start = position
    moves: list[str] = []
    for _ in range(steps):
        options: list[tuple[str, int]] = []
        if position > 0:
            options.append(("L", -1))
        if position < width - 1:
            options.append(("R", 1))
        move, delta = rng.choice(options)
        moves.append(move)
        position += delta
    if stage == "C_HELDOUT":
        rendered = " ".join("LEFT" if move == "L" else "RIGHT" for move in moves)
        prompt = f"Track a marker on line positions 0..{width - 1}. Initial position={start}. Execute commands {rendered}; LEFT subtracts one and RIGHT adds one. Emit only the terminal position."
    else:
        prompt = f"A marker starts at cell {start} on a corridor numbered 0 through {width - 1}. Apply moves {' '.join(moves)} where L=-1 and R=+1. Return only the final cell."
    return prompt, str(position), tuple(moves)


def _apply_sequence_operation(values: list[int], operation: str) -> list[int]:
    if operation == "reverse":
        return list(reversed(values))
    if operation.startswith("rotate_left:"):
        shift = int(operation.split(":", 1)[1]) % len(values)
        return values[shift:] + values[:shift]
    if operation.startswith("add_mod10:"):
        add = int(operation.split(":", 1)[1])
        return [(value + add) % 10 for value in values]
    raise ValueError(f"unknown sequence operation: {operation}")


def _sequence_world(rng: random.Random, *, stage: str, complexity: int) -> tuple[str, str, tuple[str, ...]]:
    length = 3 + complexity
    values = [rng.randrange(10) for _ in range(length)]
    shift = 1 + rng.randrange(max(1, length - 1))
    add = 1 + rng.randrange(4)
    if stage == "A_SANITY":
        choices = (("reverse",), (f"rotate_left:{shift}",), (f"add_mod10:{add}",))
        operations = choices[rng.randrange(len(choices))]
    elif stage in ("B_TRAIN", "B_IID"):
        templates = (("reverse", f"rotate_left:{shift}"), (f"rotate_left:{shift}", f"add_mod10:{add}"), ("reverse", f"add_mod10:{add}", f"rotate_left:{shift}"))
        operations = templates[rng.randrange(len(templates))]
    else:
        templates = ((f"add_mod10:{add}", f"rotate_left:{shift}", "reverse"), (f"rotate_left:{shift}", "reverse", f"add_mod10:{add}"))
        operations = templates[rng.randrange(len(templates))]
    transformed = list(values)
    for operation in operations:
        transformed = _apply_sequence_operation(transformed, operation)
    if stage == "C_HELDOUT":
        instruction = "; then ".join("reverse sequence" if op == "reverse" else (f"rotate left {op.split(':', 1)[1]}" if op.startswith("rotate_left:") else f"add {op.split(':', 1)[1]} modulo 10") for op in operations)
        prompt = f"Input digits: {' '.join(map(str, values))}. Apply in order: {instruction}. Output digits separated by single spaces only."
    else:
        prompt = f"Transform digits {' '.join(map(str, values))} with operations {', '.join(operations)} in that order. Return digits separated by single spaces."
    return prompt, " ".join(map(str, transformed)), tuple(operations)


def _abstract_world(rng: random.Random, *, stage: str, complexity: int) -> tuple[str, str, tuple[str, ...]]:
    if stage == "A_SANITY":
        modulus = 5
        bias = 1 + rng.randrange(3)
        transform = lambda x: (x + bias) % modulus
        trace = (f"add:{bias}", f"mod:{modulus}")
    elif stage in ("B_TRAIN", "B_IID"):
        modulus = 7
        multiplier = 2 + rng.randrange(3)
        bias = 1 + rng.randrange(4)
        transform = lambda x: (multiplier * x + bias) % modulus
        trace = (f"multiply:{multiplier}", f"add:{bias}", f"mod:{modulus}")
    else:
        modulus = 11
        multiplier = 2 + rng.randrange(4)
        bias = 1 + rng.randrange(5)
        transform = lambda x: (multiplier * (x + bias)) % modulus
        trace = (f"add:{bias}", f"multiply:{multiplier}", f"mod:{modulus}")
    demo_count = 3 if stage == "A_SANITY" else 4
    demos = [f"{value}->{transform(value)}" for value in range(demo_count)]
    query = demo_count + rng.randrange(1, 2 + complexity)
    answer = transform(query)
    if stage == "C_HELDOUT":
        prompt = f"Infer one deterministic integer mapping from demonstrations, then solve the query. Demonstrations: {'; '.join(demos)}. Complete {query}->?. Emit only the integer."
    else:
        prompt = f"Infer the deterministic mapping from examples and apply it to the query. Examples: {', '.join(demos)}. Query: {query}->?. Return only the integer."
    return prompt, str(answer), trace


def _language_world(rng: random.Random, *, stage: str, complexity: int) -> tuple[str, str, tuple[str, ...]]:
    vocab_by_stage = {
        "A_SANITY": ("amber", "birch", "cedar", "delta", "ember", "fjord"),
        "B_TRAIN": ("galaxy", "harbor", "islet", "juniper", "kernel", "lagoon", "meadow", "nectar"),
        "B_IID": ("opal", "prairie", "quartz", "raven", "summit", "thicket", "umber", "valley"),
        "C_HELDOUT": ("willow", "xenon", "yarrow", "zephyr", "acorn", "beacon", "canyon", "dune"),
    }
    vocab = vocab_by_stage[stage]
    if stage == "A_SANITY":
        count = min(len(vocab), 3 + (complexity % 2))
    elif stage in ("B_TRAIN", "B_IID"):
        count = min(len(vocab), 5 + (complexity % 2))
    else:
        count = min(len(vocab), 6 + (complexity % 2))
    chosen = rng.sample(list(vocab), count)
    mode = ("alphabetical", "reverse")[rng.randrange(2)]
    answer_tokens = sorted(chosen) if mode == "alphabetical" else list(reversed(chosen))
    if stage == "C_HELDOUT":
        action = "lexicographically ascending" if mode == "alphabetical" else "opposite input order"
        prompt = f"Normalize this token list into {action}: {' / '.join(chosen)}. Return only space-separated tokens."
    else:
        action = "Sort alphabetically" if mode == "alphabetical" else "Reverse the word order"
        prompt = f"{action}. Words: {' | '.join(chosen)}. Return only the resulting words separated by one space."
    return prompt, " ".join(answer_tokens), (mode,)


def generate_world_instance(*, family: str, stage: str, root: int, index: int, challenge_beacon: str | None = None, stage_b_selection_digest: str | None = None) -> Exp319WorldInstance:
    if family not in EXP319_TASK_FAMILIES:
        raise ValueError(f"unknown EXP-319 family: {family!r}")
    _validate_stage_authority(stage=stage, root=root, index=index, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest)
    seed = _seed_for(family=family, stage=stage, root=root, index=index, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest)
    rng = random.Random(seed)
    complexity = _complexity_for(stage=stage, root=root, index=index)
    if family == "iterative-grid-and-maze":
        model_input, answer, operation_trace = _iterative_world(rng, stage=stage, complexity=complexity)
    elif family == "algorithmic-sequence-transform":
        model_input, answer, operation_trace = _sequence_world(rng, stage=stage, complexity=complexity)
    elif family == "generator-heldout-abstract-transformation":
        model_input, answer, operation_trace = _abstract_world(rng, stage=stage, complexity=complexity)
    else:
        model_input, answer, operation_trace = _language_world(rng, stage=stage, complexity=complexity)
    generator_identity, template_identity = _identities(family=family, stage=stage, root=root, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest)
    content_payload = {
        "version": WORLD_GENERATOR_VERSION,
        "stage": stage,
        "root": root,
        "family": family,
        "template_identity": template_identity,
        "generator_identity": generator_identity,
        "index": index,
        "complexity": complexity,
        "required_steps": len(operation_trace),
        "model_input": model_input,
        "canonical_answer": answer,
        "challenge_beacon": challenge_beacon,
        "stage_b_selection_digest": stage_b_selection_digest,
        "operation_trace": list(operation_trace),
    }
    content_id = "exp319:" + _sha256(_canonical_bytes(content_payload))
    return Exp319WorldInstance(
        family=family,
        generator_version=WORLD_GENERATOR_VERSION,
        generator_identity=generator_identity,
        template_identity=template_identity,
        content_id=content_id,
        stage=stage,
        root=root,
        index=index,
        complexity=complexity,
        required_steps=len(operation_trace),
        model_input=model_input,
        canonical_answer=answer,
        challenge_beacon=challenge_beacon,
        stage_b_selection_digest=stage_b_selection_digest,
        operation_trace=tuple(operation_trace),
        tags=("exp319", stage.lower(), family),
    )


def _materialize(*, stage: str, root: int, per_family: int, challenge_beacon: str | None = None, stage_b_selection_digest: str | None = None) -> tuple[Exp319WorldInstance, ...]:
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    return tuple(generate_world_instance(family=family, stage=stage, root=root, index=index, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest) for family in EXP319_TASK_FAMILIES for index in range(per_family))


def materialize_stage_a() -> tuple[Exp319WorldInstance, ...]:
    return _materialize(stage="A_SANITY", root=0, per_family=8)


def materialize_stage_b(*, root: int, split: str) -> tuple[Exp319WorldInstance, ...]:
    if split == "train":
        return _materialize(stage="B_TRAIN", root=root, per_family=128)
    if split == "iid":
        return _materialize(stage="B_IID", root=root, per_family=64)
    raise ValueError("split must be 'train' or 'iid'")


def materialize_stage_c(*, root: int, challenge_beacon: str, stage_b_selection_digest: str) -> tuple[Exp319WorldInstance, ...]:
    return _materialize(stage="C_HELDOUT", root=root, per_family=128, challenge_beacon=challenge_beacon, stage_b_selection_digest=stage_b_selection_digest)


def build_generator_manifest(instances: Iterable[Exp319WorldInstance]) -> dict[str, Any]:
    worlds = tuple(instances)
    if not worlds:
        raise ValueError("generator manifest requires at least one world")
    payload = {
        "schema": "EXP319-GENERATOR-MANIFEST-V1",
        "generator_version": WORLD_GENERATOR_VERSION,
        "world_count": len(worlds),
        "stages": sorted({world.stage for world in worlds}),
        "roots": sorted({world.root for world in worlds}),
        "families": sorted({world.family for world in worlds}),
        "content_ids": sorted(world.content_id for world in worlds),
        "generator_identities": sorted({world.generator_identity for world in worlds}),
        "template_identities": sorted({world.template_identity for world in worlds}),
        "challenge_beacons": sorted({world.challenge_beacon for world in worlds if world.challenge_beacon is not None}),
        "stage_b_selection_digests": sorted({world.stage_b_selection_digest for world in worlds if world.stage_b_selection_digest is not None}),
    }
    payload["digest"] = _sha256(_canonical_bytes(payload))
    return payload


def verify_world_answer(instance: Exp319WorldInstance, candidate: str) -> bool:
    return candidate.strip() == instance.canonical_answer
