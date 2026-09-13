from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Protocol

from .exp301_execution import (
    EXP301_ARMS,
    EXP301_CHALLENGE_PER_FAMILY,
    EXP301_LR_CANDIDATES,
    EXP301_ROOTS,
)
from .exp301_identity import RuntimeRootIdentity, build_runtime_root_identity
from .exp301_worlds import Exp301WorldInstance, materialize_world_set


_HEX = frozenset("0123456789abcdef")


class TrialEvidence(Protocol):
    arm_id: str
    root: int
    learning_rate: float
    model_init_seed: int
    training_steps: int
    development_family_balanced_score: float
    development_family_scores: tuple[tuple[str, float], ...]
    checkpoint_digest: str
    trial_receipt_digest: str


@dataclass(frozen=True, slots=True)
class ArmSelectionReceipt:
    schema: str
    arm_id: str
    root: int
    selected_learning_rate: float
    selected_checkpoint_digest: str
    selected_trial_receipt_digest: str
    trial_receipt_digests: tuple[str, ...]
    selection_digest: str


@dataclass(frozen=True, slots=True)
class RootSelectionManifest:
    schema: str
    root: int
    arm_selections: tuple[ArmSelectionReceipt, ...]
    selection_manifest_digest: str


@dataclass(frozen=True, slots=True)
class ChallengeMaterialization:
    schema: str
    root: int
    challenge_nonce: str
    worlds: tuple[Exp301WorldInstance, ...]
    materialization_digest: str


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _require_hex(value: str, *, field: str, length: int = 64) -> None:
    if not isinstance(value, str) or len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")


def _validate_trial_evidence(result: TrialEvidence) -> None:
    if result.arm_id not in EXP301_ARMS:
        raise ValueError("trial arm mismatch")
    if result.root not in EXP301_ROOTS:
        raise ValueError("trial root mismatch")
    if result.learning_rate not in EXP301_LR_CANDIDATES:
        raise ValueError("trial learning-rate mismatch")
    if result.training_steps != 512:
        raise ValueError("trial must bind exactly 512 frozen optimizer steps")
    if not 0.0 <= result.development_family_balanced_score <= 1.0:
        raise ValueError("development score must be in [0,1]")
    _require_hex(result.checkpoint_digest, field="checkpoint_digest")
    _require_hex(result.trial_receipt_digest, field="trial_receipt_digest")


def build_arm_selection_receipt(results: Iterable[TrialEvidence]) -> ArmSelectionReceipt:
    materialized = tuple(results)
    if len(materialized) != 2:
        raise ValueError("arm selection requires exactly the two frozen LR trials")
    for result in materialized:
        _validate_trial_evidence(result)
    arm_ids = {result.arm_id for result in materialized}
    roots = {result.root for result in materialized}
    lrs = {result.learning_rate for result in materialized}
    if len(arm_ids) != 1 or len(roots) != 1:
        raise ValueError("arm selection trials must share arm and root")
    if lrs != set(EXP301_LR_CANDIDATES):
        raise ValueError("arm selection must contain exactly the frozen LR candidates")

    ordered_trials = tuple(sorted(materialized, key=lambda result: result.learning_rate))
    selected = max(
        ordered_trials,
        key=lambda result: (result.development_family_balanced_score, -result.learning_rate),
    )
    values = {
        "schema": "EXP301-ARM-SELECTION-V1",
        "arm_id": selected.arm_id,
        "root": selected.root,
        "selected_learning_rate": selected.learning_rate,
        "selected_checkpoint_digest": selected.checkpoint_digest,
        "selected_trial_receipt_digest": selected.trial_receipt_digest,
        "trial_receipt_digests": tuple(result.trial_receipt_digest for result in ordered_trials),
    }
    return ArmSelectionReceipt(**values, selection_digest=_canonical_digest(values))


def build_root_selection_manifest(receipts: Iterable[ArmSelectionReceipt]) -> RootSelectionManifest:
    materialized = tuple(receipts)
    if len(materialized) != len(EXP301_ARMS):
        raise ValueError(f"root selection must contain exactly arms {EXP301_ARMS}")
    by_arm = {receipt.arm_id: receipt for receipt in materialized}
    if set(by_arm) != set(EXP301_ARMS) or len(by_arm) != len(materialized):
        raise ValueError(f"root selection must contain each arm exactly once: {EXP301_ARMS}")
    roots = {receipt.root for receipt in materialized}
    if len(roots) != 1:
        raise ValueError("all arm selections must share one root")
    ordered = tuple(by_arm[arm] for arm in EXP301_ARMS)
    for receipt in ordered:
        _require_hex(receipt.selection_digest, field="selection_digest")
        payload = asdict(receipt)
        digest = payload.pop("selection_digest")
        if _canonical_digest(payload) != digest:
            raise ValueError(f"arm selection digest mismatch for {receipt.arm_id}")
    root = ordered[0].root
    values = {
        "schema": "EXP301-ROOT-SELECTION-MANIFEST-V1",
        "root": root,
        "arm_selections": tuple(asdict(receipt) for receipt in ordered),
    }
    return RootSelectionManifest(
        schema=values["schema"],
        root=root,
        arm_selections=ordered,
        selection_manifest_digest=_canonical_digest(values),
    )


def derive_challenge_nonce(*, beacon: str, frozen_implementation_digest: str) -> str:
    if not isinstance(beacon, str) or not beacon.strip():
        raise ValueError("challenge beacon must be a non-empty string")
    _require_hex(frozen_implementation_digest, field="frozen_implementation_digest")
    return _canonical_digest(
        {
            "schema": "EXP301-POST-FREEZE-CHALLENGE-NONCE-V1",
            "beacon": beacon,
            "frozen_implementation_digest": frozen_implementation_digest,
        }
    )


def _world_identity(world: Exp301WorldInstance) -> dict[str, object]:
    return {
        "content_id": world.content_id,
        "family": world.family,
        "generator_version": world.generator_version,
        "generator_identity": world.generator_identity,
        "template_identity": world.template_identity,
        "root": world.root,
        "split": world.split,
        "index": world.index,
        "difficulty": world.difficulty,
        "required_steps": world.required_steps,
    }


def materialize_root_challenge(
    *,
    root: int,
    beacon: str,
    frozen_implementation_digest: str,
) -> ChallengeMaterialization:
    if root not in EXP301_ROOTS:
        raise ValueError(f"root must be one of {EXP301_ROOTS}")
    nonce = derive_challenge_nonce(
        beacon=beacon,
        frozen_implementation_digest=frozen_implementation_digest,
    )
    worlds = materialize_world_set(
        split="challenge",
        roots=(root,),
        per_family=EXP301_CHALLENGE_PER_FAMILY,
        challenge_nonce=nonce,
    )
    if len(worlds) != 512:
        raise RuntimeError(f"frozen challenge world count drift: {len(worlds)}")
    values = {
        "schema": "EXP301-ROOT-CHALLENGE-MATERIALIZATION-V1",
        "root": root,
        "challenge_nonce": nonce,
        "world_identities": tuple(_world_identity(world) for world in worlds),
    }
    return ChallengeMaterialization(
        schema=values["schema"],
        root=root,
        challenge_nonce=nonce,
        worlds=worlds,
        materialization_digest=_canonical_digest(values),
    )


def build_runtime_identity_for_root(
    *,
    frozen_implementation_digest: str,
    selection_manifest: RootSelectionManifest,
    challenge: ChallengeMaterialization,
    beacon: str,
) -> RuntimeRootIdentity:
    if selection_manifest.root != challenge.root:
        raise ValueError("selection/challenge root mismatch")
    _require_hex(selection_manifest.selection_manifest_digest, field="selection_manifest_digest")
    _require_hex(challenge.materialization_digest, field="challenge_materialization_digest")
    beacon_digest = hashlib.sha256(beacon.encode("utf-8")).hexdigest()
    return build_runtime_root_identity(
        frozen_implementation_digest=frozen_implementation_digest,
        root=selection_manifest.root,
        selected_hyperparameter_receipt_digest=selection_manifest.selection_manifest_digest,
        challenge_beacon_digest=beacon_digest,
        challenge_materialization_digest=challenge.materialization_digest,
        scientific_evidence_eligible=True,
    )
