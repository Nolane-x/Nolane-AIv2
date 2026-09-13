from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp301_worlds import EXP301_TASK_FAMILIES, materialize_world_set


EXP301_TRAINING_LOOPS = (1, 2, 4, 8)
EXP301_CHALLENGE_LOOPS = (1, 2, 4, 8, 12, 16)
EXP301_LR_CANDIDATES = (1e-4, 3e-4)
EXP301_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EXP301_ROOTS = (0, 1, 2, 3)
SCIENTIFIC_EXECUTION_VERSION = "exp301-scientific-execution-v1"
MODEL_INIT_VERSION = "exp301-model-init-v1"

# Frozen before challenge materialization. This module intentionally has no
# torch dependency so core CI and external auditors can validate the execution
# constitution without importing the model runtime.
EXP301_TRAIN_PER_FAMILY = 128
EXP301_DEV_PER_FAMILY = 64
EXP301_CHALLENGE_PER_FAMILY = 128
EXP301_TRAIN_EPOCHS = 1
EXP301_MAX_SEQUENCE_TOKENS = 512
EXP301_MAX_GENERATION_TOKENS = 96


@dataclass(frozen=True, slots=True)
class Exp301ScientificExecutionContract:
    version: str
    training_examples_per_family: int
    development_examples_per_family: int
    challenge_examples_per_family: int
    training_examples_per_root: int
    development_examples_per_root: int
    challenge_examples_per_root: int
    training_epochs: int
    optimizer_steps_per_trial: int
    search_trials_per_arm: int
    primary_efforts: tuple[int, ...]
    challenge_efforts: tuple[int, ...]
    max_sequence_tokens: int
    max_generation_tokens: int
    batch_size: int
    development_selection_rule: str
    challenge_materialization_policy: str
    model_init_version: str


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_scientific_execution_contract() -> Exp301ScientificExecutionContract:
    family_count = len(EXP301_TASK_FAMILIES)
    training_examples = EXP301_TRAIN_PER_FAMILY * family_count
    development_examples = EXP301_DEV_PER_FAMILY * family_count
    challenge_examples = EXP301_CHALLENGE_PER_FAMILY * family_count
    return Exp301ScientificExecutionContract(
        version=SCIENTIFIC_EXECUTION_VERSION,
        training_examples_per_family=EXP301_TRAIN_PER_FAMILY,
        development_examples_per_family=EXP301_DEV_PER_FAMILY,
        challenge_examples_per_family=EXP301_CHALLENGE_PER_FAMILY,
        training_examples_per_root=training_examples,
        development_examples_per_root=development_examples,
        challenge_examples_per_root=challenge_examples,
        training_epochs=EXP301_TRAIN_EPOCHS,
        optimizer_steps_per_trial=training_examples * EXP301_TRAIN_EPOCHS,
        search_trials_per_arm=len(EXP301_LR_CANDIDATES),
        primary_efforts=EXP301_TRAINING_LOOPS,
        challenge_efforts=EXP301_CHALLENGE_LOOPS,
        max_sequence_tokens=EXP301_MAX_SEQUENCE_TOKENS,
        max_generation_tokens=EXP301_MAX_GENERATION_TOKENS,
        batch_size=1,
        development_selection_rule=(
            "highest family-balanced development verified success across primary efforts; "
            "ties -> lower LR"
        ),
        challenge_materialization_policy=(
            "challenge worlds are not materialized by training code; post-freeze nonce required"
        ),
        model_init_version=MODEL_INIT_VERSION,
    )


def scientific_execution_contract_digest() -> str:
    return _canonical_digest(asdict(build_scientific_execution_contract()))


def _validate_root(root: int) -> None:
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")


def _ordered_worlds(*, root: int, split: str, per_family: int):
    _validate_root(root)
    worlds = materialize_world_set(split=split, roots=(root,), per_family=per_family)
    return tuple(sorted(worlds, key=lambda item: (item.family, item.content_id)))


def training_worlds_for_root(root: int):
    return _ordered_worlds(root=root, split="train", per_family=EXP301_TRAIN_PER_FAMILY)


def development_worlds_for_root(root: int):
    return _ordered_worlds(root=root, split="development", per_family=EXP301_DEV_PER_FAMILY)


def scientific_model_init_seed(*, root: int, arm_id: str, trial: int) -> int:
    _validate_root(root)
    if arm_id not in EXP301_ARMS:
        raise ValueError(f"arm must be one of {EXP301_ARMS}")
    if not 0 <= trial < len(EXP301_LR_CANDIDATES):
        raise ValueError(f"trial must be in [0, {len(EXP301_LR_CANDIDATES) - 1}]")
    digest = hashlib.sha256(
        f"{MODEL_INIT_VERSION}|root={root}|arm={arm_id}|trial={trial}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
