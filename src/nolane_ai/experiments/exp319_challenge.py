from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

from .exp301_training import Exp301ByteTokenizer
from .exp319_contract import COMMON_GATING_EFFORT, CONTRACT, PRIMARY_ARMS, canonical_json_bytes
from .exp319_worlds import (
    EXP319_TASK_FAMILIES,
    build_generator_manifest,
    materialize_stage_c,
    verify_world_answer,
)


STAGE_C_SHARD_COUNT = 16
STAGE_C_WORLDS_PER_SHARD = 32
STAGE_C_WORLD_COUNT = STAGE_C_SHARD_COUNT * STAGE_C_WORLDS_PER_SHARD

_SELECTION_SCHEMA = "EXP319-STAGE-B-SELECTION-SEAL-V1"
_AUTHORITY_SCHEMA = "EXP319-STAGE-B-SELECTION-AUTHORITY-V1"
_MANIFEST_SCHEMA = "EXP319-STAGE-C-MANIFEST-V1"
_COMMITMENT_SCHEMA = "EXP319-STAGE-C-PREDICTION-COMMITMENT-V1"
_SHARD_SCHEMA = "EXP319-STAGE-C-PREDICTION-SHARD-V1"


@dataclass(frozen=True, slots=True)
class StageBSelectionSeal:
    arm_id: str
    root: int
    selected_step: int
    passes_floor: bool
    checkpoint_digest: str
    selection_digest: str


@dataclass(frozen=True, slots=True)
class StageBSelectionAuthority:
    schema: str
    roots: tuple[int, ...]
    primary_arms: tuple[str, ...]
    selections: tuple[StageBSelectionSeal, ...]
    authority_digest: str


@dataclass(frozen=True, slots=True)
class StageCManifest:
    schema: str
    root: int
    challenge_beacon: str
    source_digest: str
    stage_b_selection_digest: str
    selection_roots: tuple[int, ...]
    primary_arms: tuple[str, ...]
    effort: int
    gradient_updates: int
    world_count: int
    content_ids: tuple[str, ...]
    generator_manifest_digest: str
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class StageCPredictionInput:
    root: int
    family: str
    content_id: str
    model_input: str


@dataclass(frozen=True, slots=True)
class StageCPredictionOutput:
    content_id: str
    arm_id: str
    candidate_answer: str
    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool


@dataclass(frozen=True, slots=True)
class StageCPredictionCommitment:
    schema: str
    root: int
    shard_index: int
    content_id: str
    arm_id: str
    effort: int
    candidate_answer: str
    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool
    challenge_manifest_digest: str
    commitment_digest: str


@dataclass(frozen=True, slots=True)
class StageCPredictionShard:
    schema: str
    root: int
    shard_index: int
    world_count: int
    challenge_manifest_digest: str
    commitments: tuple[StageCPredictionCommitment, ...]
    shard_digest: str


@dataclass(frozen=True, slots=True)
class StageCScoredRow:
    root: int
    family: str
    content_id: str
    arm_id: str
    effort: int
    commitment_digest: str
    exact: bool
    answer_token_accuracy: float
    eos_correct: bool
    invalid_output: bool


def _sha256_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _valid_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _require_digest(value: str, *, label: str) -> None:
    if not _valid_digest(value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")


def _selection_payload(seal: StageBSelectionSeal) -> dict[str, object]:
    return {
        "schema": _SELECTION_SCHEMA,
        "arm_id": seal.arm_id,
        "root": seal.root,
        "selected_step": seal.selected_step,
        "passes_floor": seal.passes_floor,
        "checkpoint_digest": seal.checkpoint_digest,
        "selection_digest": seal.selection_digest,
    }


def _validate_selection_seal(seal: StageBSelectionSeal) -> None:
    if seal.arm_id not in PRIMARY_ARMS:
        raise ValueError("Stage B selection seal arm mismatch")
    if seal.root not in CONTRACT.stage_b.roots:
        raise ValueError("Stage B selection seal root mismatch")
    if seal.selected_step not in CONTRACT.stage_b.checkpoints:
        raise ValueError("Stage B selection seal checkpoint mismatch")
    if not isinstance(seal.passes_floor, bool):
        raise ValueError("Stage B selection seal passes_floor must be boolean")
    _require_digest(seal.checkpoint_digest, label="checkpoint_digest")
    _require_digest(seal.selection_digest, label="selection_digest")


def seal_stage_b_selection_authority(
    selections: Iterable[StageBSelectionSeal],
) -> StageBSelectionAuthority:
    materialized = tuple(selections)
    expected_keys = {
        (arm_id, root)
        for arm_id in PRIMARY_ARMS
        for root in CONTRACT.stage_b.roots
    }
    if len(materialized) != len(expected_keys):
        raise ValueError("complete Stage B authority requires all eight sealed selections")
    for seal in materialized:
        _validate_selection_seal(seal)
    actual_keys = {(seal.arm_id, seal.root) for seal in materialized}
    if actual_keys != expected_keys:
        raise ValueError("complete Stage B authority requires exactly one seal per arm/root")

    for arm_id in PRIMARY_ARMS:
        passed = sum(
            seal.passes_floor
            for seal in materialized
            if seal.arm_id == arm_id
        )
        if passed < CONTRACT.stage_b.cross_root_min_passes:
            raise ValueError(f"Stage B cross-root gate not satisfied for {arm_id}")

    ordered = tuple(sorted(materialized, key=lambda item: (item.arm_id, item.root)))
    payload = {
        "schema": _AUTHORITY_SCHEMA,
        "roots": list(CONTRACT.stage_b.roots),
        "primary_arms": list(PRIMARY_ARMS),
        "selections": [_selection_payload(item) for item in ordered],
    }
    return StageBSelectionAuthority(
        schema=_AUTHORITY_SCHEMA,
        roots=tuple(CONTRACT.stage_b.roots),
        primary_arms=tuple(PRIMARY_ARMS),
        selections=ordered,
        authority_digest=_sha256_payload(payload),
    )


def _validate_selection_authority(authority: StageBSelectionAuthority) -> None:
    rebuilt = seal_stage_b_selection_authority(authority.selections)
    if authority != rebuilt:
        raise ValueError("Stage B selection authority digest mismatch")


def _manifest_payload(manifest: StageCManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload.pop("manifest_digest", None)
    return payload


def _validate_manifest(manifest: StageCManifest) -> tuple[object, ...]:
    if manifest.schema != _MANIFEST_SCHEMA:
        raise ValueError("Stage C manifest schema mismatch")
    if manifest.root not in CONTRACT.stage_c.roots:
        raise ValueError("Stage C manifest root mismatch")
    if not manifest.challenge_beacon:
        raise ValueError("Stage C challenge beacon is required")
    _require_digest(manifest.source_digest, label="source_digest")
    _require_digest(manifest.stage_b_selection_digest, label="stage_b_selection_digest")
    if manifest.selection_roots != tuple(CONTRACT.stage_b.roots):
        raise ValueError("Stage C manifest selection roots mismatch")
    if manifest.primary_arms != tuple(PRIMARY_ARMS):
        raise ValueError("Stage C manifest primary arms mismatch")
    if manifest.effort != COMMON_GATING_EFFORT:
        raise ValueError("Stage C manifest effort mismatch")
    if manifest.gradient_updates != 0:
        raise ValueError("Stage C forbids gradient updates")
    if manifest.world_count != STAGE_C_WORLD_COUNT:
        raise ValueError("Stage C manifest world count mismatch")
    if len(manifest.content_ids) != STAGE_C_WORLD_COUNT or len(set(manifest.content_ids)) != STAGE_C_WORLD_COUNT:
        raise ValueError("Stage C manifest content grid mismatch")
    _require_digest(manifest.generator_manifest_digest, label="generator_manifest_digest")
    _require_digest(manifest.manifest_digest, label="manifest_digest")
    expected_manifest_digest = _sha256_payload(_manifest_payload(manifest))
    if manifest.manifest_digest != expected_manifest_digest:
        raise ValueError("Stage C manifest digest mismatch")

    worlds = materialize_stage_c(
        root=manifest.root,
        challenge_beacon=manifest.challenge_beacon,
        stage_b_selection_digest=manifest.stage_b_selection_digest,
    )
    if tuple(world.content_id for world in worlds) != manifest.content_ids:
        raise ValueError("Stage C manifest does not reproduce content IDs")
    generator_manifest = build_generator_manifest(worlds)
    if generator_manifest["digest"] != manifest.generator_manifest_digest:
        raise ValueError("Stage C generator manifest digest mismatch")
    return worlds


def materialize_stage_c_manifest(
    *,
    root: int,
    challenge_beacon: str,
    source_digest: str,
    selection_authority: StageBSelectionAuthority,
) -> StageCManifest:
    if root not in CONTRACT.stage_c.roots:
        raise ValueError("Stage C root mismatch")
    if not challenge_beacon:
        raise ValueError("challenge_beacon is required")
    _require_digest(source_digest, label="source_digest")
    _validate_selection_authority(selection_authority)

    worlds = materialize_stage_c(
        root=root,
        challenge_beacon=challenge_beacon,
        stage_b_selection_digest=selection_authority.authority_digest,
    )
    if len(worlds) != STAGE_C_WORLD_COUNT:
        raise ValueError("Stage C generator did not materialize exactly 512 worlds")
    generator_manifest = build_generator_manifest(worlds)
    without_digest = {
        "schema": _MANIFEST_SCHEMA,
        "root": root,
        "challenge_beacon": challenge_beacon,
        "source_digest": source_digest,
        "stage_b_selection_digest": selection_authority.authority_digest,
        "selection_roots": tuple(CONTRACT.stage_b.roots),
        "primary_arms": tuple(PRIMARY_ARMS),
        "effort": COMMON_GATING_EFFORT,
        "gradient_updates": 0,
        "world_count": len(worlds),
        "content_ids": tuple(world.content_id for world in worlds),
        "generator_manifest_digest": str(generator_manifest["digest"]),
    }
    manifest = StageCManifest(
        **without_digest,
        manifest_digest=_sha256_payload(without_digest),
    )
    _validate_manifest(manifest)
    return manifest


def prediction_inputs_for_shard(
    manifest: StageCManifest,
    *,
    shard_index: int,
) -> tuple[StageCPredictionInput, ...]:
    if not 0 <= shard_index < STAGE_C_SHARD_COUNT:
        raise ValueError("Stage C shard index is outside the frozen range 0..15")
    worlds = _validate_manifest(manifest)
    start = shard_index * STAGE_C_WORLDS_PER_SHARD
    stop = start + STAGE_C_WORLDS_PER_SHARD
    selected = worlds[start:stop]
    if len(selected) != STAGE_C_WORLDS_PER_SHARD:
        raise ValueError("Stage C shard geometry mismatch")
    return tuple(
        StageCPredictionInput(
            root=manifest.root,
            family=world.family,
            content_id=world.content_id,
            model_input=world.model_input,
        )
        for world in selected
    )


def _commitment_payload(commitment: StageCPredictionCommitment) -> dict[str, object]:
    payload = asdict(commitment)
    payload.pop("commitment_digest", None)
    return payload


def _validate_commitment(commitment: StageCPredictionCommitment, manifest: StageCManifest) -> None:
    if commitment.schema != _COMMITMENT_SCHEMA:
        raise ValueError("Stage C commitment schema mismatch")
    if commitment.root != manifest.root:
        raise ValueError("Stage C commitment root mismatch")
    if commitment.arm_id not in PRIMARY_ARMS:
        raise ValueError("Stage C commitment arm mismatch")
    if commitment.effort != COMMON_GATING_EFFORT:
        raise ValueError("Stage C commitment effort mismatch")
    if not 0 <= commitment.shard_index < STAGE_C_SHARD_COUNT:
        raise ValueError("Stage C commitment shard mismatch")
    if commitment.challenge_manifest_digest != manifest.manifest_digest:
        raise ValueError("Stage C commitment challenge manifest mismatch")
    if not isinstance(commitment.candidate_answer, str):
        raise ValueError("Stage C candidate answer must be a string")
    if not isinstance(commitment.generated_token_ids, tuple) or not all(
        isinstance(token, int) and not isinstance(token, bool)
        for token in commitment.generated_token_ids
    ):
        raise ValueError("Stage C generated token IDs must be an integer tuple")
    if not isinstance(commitment.stopped_on_eos, bool):
        raise ValueError("Stage C stopped_on_eos must be boolean")
    _require_digest(commitment.commitment_digest, label="commitment_digest")
    if commitment.commitment_digest != _sha256_payload(_commitment_payload(commitment)):
        raise ValueError("Stage C commitment digest mismatch")


def build_prediction_shard(
    manifest: StageCManifest,
    *,
    shard_index: int,
    outputs: Iterable[StageCPredictionOutput],
) -> StageCPredictionShard:
    inputs = prediction_inputs_for_shard(manifest, shard_index=shard_index)
    expected_keys = {
        (item.content_id, arm_id)
        for item in inputs
        for arm_id in PRIMARY_ARMS
    }
    materialized = tuple(outputs)
    actual_keys = {(item.content_id, item.arm_id) for item in materialized}
    if any(item.arm_id not in PRIMARY_ARMS for item in materialized):
        raise ValueError("Stage C output arm must be one of the frozen primary arms")
    if len(materialized) != len(expected_keys) or actual_keys != expected_keys:
        if len(actual_keys) != len(materialized):
            raise ValueError("duplicate Stage C prediction output in commitment grid")
        raise ValueError("complete Stage C prediction output grid required")

    commitments: list[StageCPredictionCommitment] = []
    for output in materialized:
        if not isinstance(output.candidate_answer, str):
            raise ValueError("Stage C output candidate_answer must be a string")
        token_ids = tuple(output.generated_token_ids)
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids):
            raise ValueError("Stage C output generated_token_ids must contain integers")
        without_digest = {
            "schema": _COMMITMENT_SCHEMA,
            "root": manifest.root,
            "shard_index": shard_index,
            "content_id": output.content_id,
            "arm_id": output.arm_id,
            "effort": COMMON_GATING_EFFORT,
            "candidate_answer": output.candidate_answer,
            "generated_token_ids": token_ids,
            "stopped_on_eos": bool(output.stopped_on_eos),
            "challenge_manifest_digest": manifest.manifest_digest,
        }
        commitment = StageCPredictionCommitment(
            **without_digest,
            commitment_digest=_sha256_payload(without_digest),
        )
        _validate_commitment(commitment, manifest)
        commitments.append(commitment)

    ordered = tuple(
        sorted(
            commitments,
            key=lambda item: (
                manifest.content_ids.index(item.content_id),
                PRIMARY_ARMS.index(item.arm_id),
            ),
        )
    )
    shard_payload = {
        "schema": _SHARD_SCHEMA,
        "root": manifest.root,
        "shard_index": shard_index,
        "world_count": STAGE_C_WORLDS_PER_SHARD,
        "challenge_manifest_digest": manifest.manifest_digest,
        "commitment_digests": [item.commitment_digest for item in ordered],
    }
    return StageCPredictionShard(
        schema=_SHARD_SCHEMA,
        root=manifest.root,
        shard_index=shard_index,
        world_count=STAGE_C_WORLDS_PER_SHARD,
        challenge_manifest_digest=manifest.manifest_digest,
        commitments=ordered,
        shard_digest=_sha256_payload(shard_payload),
    )


def _validate_shard(shard: StageCPredictionShard, manifest: StageCManifest) -> None:
    if shard.schema != _SHARD_SCHEMA:
        raise ValueError("Stage C shard schema mismatch")
    if shard.root != manifest.root:
        raise ValueError("Stage C shard root mismatch")
    if not 0 <= shard.shard_index < STAGE_C_SHARD_COUNT:
        raise ValueError("Stage C shard index mismatch")
    if shard.world_count != STAGE_C_WORLDS_PER_SHARD:
        raise ValueError("Stage C shard world count mismatch")
    if shard.challenge_manifest_digest != manifest.manifest_digest:
        raise ValueError("Stage C shard manifest mismatch")
    if len(shard.commitments) != STAGE_C_WORLDS_PER_SHARD * len(PRIMARY_ARMS):
        raise ValueError("Stage C shard commitment grid incomplete")
    for commitment in shard.commitments:
        _validate_commitment(commitment, manifest)
        if commitment.shard_index != shard.shard_index:
            raise ValueError("Stage C shard/commitment index mismatch")
    payload = {
        "schema": _SHARD_SCHEMA,
        "root": shard.root,
        "shard_index": shard.shard_index,
        "world_count": shard.world_count,
        "challenge_manifest_digest": shard.challenge_manifest_digest,
        "commitment_digests": [item.commitment_digest for item in shard.commitments],
    }
    if shard.shard_digest != _sha256_payload(payload):
        raise ValueError("Stage C shard digest mismatch")


def merge_prediction_shards(
    manifest: StageCManifest,
    shards: Iterable[StageCPredictionShard],
) -> tuple[StageCPredictionCommitment, ...]:
    _validate_manifest(manifest)
    materialized = tuple(shards)
    if len(materialized) != STAGE_C_SHARD_COUNT:
        raise ValueError("complete Stage C merge requires exactly 16 shards")
    indices = tuple(shard.shard_index for shard in materialized)
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate Stage C shard index; complete 16-shard cover required")
    if set(indices) != set(range(STAGE_C_SHARD_COUNT)):
        raise ValueError("complete Stage C merge requires shards 0 through 15")
    for shard in materialized:
        _validate_shard(shard, manifest)

    ordered_shards = sorted(materialized, key=lambda item: item.shard_index)
    commitments = tuple(
        commitment
        for shard in ordered_shards
        for commitment in shard.commitments
    )
    expected_keys = {
        (content_id, arm_id)
        for content_id in manifest.content_ids
        for arm_id in PRIMARY_ARMS
    }
    actual_keys = {(item.content_id, item.arm_id) for item in commitments}
    if len(commitments) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("complete Stage C commitment grid required after shard merge")
    return commitments


def _answer_token_accuracy(generated: Sequence[int], expected: Sequence[int]) -> float:
    if not expected:
        raise ValueError("expected answer token sequence cannot be empty")
    width = max(len(generated), len(expected))
    if width == 0:
        return 0.0
    correct = sum(
        index < len(generated)
        and index < len(expected)
        and generated[index] == expected[index]
        for index in range(width)
    )
    return correct / width


def _invalid_generated_tokens(token_ids: Sequence[int]) -> bool:
    tokenizer = Exp301ByteTokenizer()
    for token_id in token_ids:
        if token_id == tokenizer.eos_id:
            return False
        if not tokenizer.byte_offset <= token_id < tokenizer.byte_offset + 256:
            return True
    return False


def score_stage_c_commitments(
    manifest: StageCManifest,
    commitments: Iterable[StageCPredictionCommitment],
) -> tuple[StageCScoredRow, ...]:
    worlds = _validate_manifest(manifest)
    materialized = tuple(commitments)
    expected_keys = {
        (world.content_id, arm_id)
        for world in worlds
        for arm_id in PRIMARY_ARMS
    }
    actual_keys = {(item.content_id, item.arm_id) for item in materialized}
    if len(materialized) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("complete commitment grid required before Stage C scoring")
    for commitment in materialized:
        _validate_commitment(commitment, manifest)

    tokenizer = Exp301ByteTokenizer()
    worlds_by_id = {world.content_id: world for world in worlds}
    rows: list[StageCScoredRow] = []
    for commitment in materialized:
        world = worlds_by_id[commitment.content_id]
        expected_tokens = (*tokenizer.encode_text(world.canonical_answer), tokenizer.eos_id)
        generated_tokens = tuple(commitment.generated_token_ids)
        eos_correct = (
            commitment.stopped_on_eos
            and bool(generated_tokens)
            and tokenizer.eos_id in generated_tokens
            and generated_tokens.index(tokenizer.eos_id) == len(generated_tokens) - 1
        )
        invalid_output = _invalid_generated_tokens(generated_tokens)
        rows.append(
            StageCScoredRow(
                root=manifest.root,
                family=world.family,
                content_id=world.content_id,
                arm_id=commitment.arm_id,
                effort=commitment.effort,
                commitment_digest=commitment.commitment_digest,
                exact=(not invalid_output and verify_world_answer(world, commitment.candidate_answer)),
                answer_token_accuracy=_answer_token_accuracy(generated_tokens, expected_tokens),
                eos_correct=eos_correct,
                invalid_output=invalid_output,
            )
        )
    return tuple(rows)
