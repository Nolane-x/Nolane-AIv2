from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .exp301_ceremony import (
    ArmSelectionReceipt,
    RootSelectionManifest,
    build_root_selection_manifest,
    build_runtime_identity_for_root,
    materialize_root_challenge,
)
from .exp301_evaluation import (
    EXP301_ARMS,
    EXP301_EFFORTS,
    Exp301EvaluationRow,
    PredictionCommitment,
    score_committed_prediction,
    validate_prediction_commitment,
)
from .exp301_execution import EXP301_MAX_GENERATION_TOKENS
from .exp301_identity import RuntimeRootIdentity


ROOT_EVIDENCE_SCHEMA = "EXP301-ROOT-SCIENTIFIC-EVIDENCE-V1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class RootScientificEvidenceArtifact:
    schema: str
    frozen_implementation_digest: str
    root: int
    run_identity: str
    selection_manifest: RootSelectionManifest
    challenge_beacon: str
    challenge_nonce: str
    challenge_materialization_digest: str
    commitments: tuple[PredictionCommitment, ...]
    evaluation_rows: tuple[Exp301EvaluationRow, ...]
    scientific_evidence_eligible: bool
    artifact_digest: str


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_hex(value: str, *, field: str, length: int = 64) -> None:
    if not isinstance(value, str) or len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")


def _artifact_payload(artifact: RootScientificEvidenceArtifact) -> dict[str, object]:
    payload = asdict(artifact)
    payload.pop("artifact_digest", None)
    return payload


def canonical_root_artifact_digest(artifact: RootScientificEvidenceArtifact) -> str:
    return _canonical_digest(_artifact_payload(artifact))


def _rebuild_selection(selection: RootSelectionManifest) -> RootSelectionManifest:
    rebuilt = build_root_selection_manifest(selection.arm_selections)
    if rebuilt != selection:
        raise ValueError("root selection manifest digest mismatch")
    return rebuilt


def _complete_grid(
    challenge,
    commitments: tuple[PredictionCommitment, ...],
) -> None:
    expected = {
        (world.content_id, arm_id, effort)
        for world in challenge.worlds
        for arm_id in EXP301_ARMS
        for effort in EXP301_EFFORTS
    }
    actual = {
        (item.content_id, item.arm_id, item.effort_multiplier)
        for item in commitments
    }
    if len(commitments) != len(expected) or actual != expected:
        raise ValueError("root evidence requires the complete commitment grid")


def _recompute_rows(
    challenge,
    commitments: tuple[PredictionCommitment, ...],
) -> tuple[Exp301EvaluationRow, ...]:
    _complete_grid(challenge, commitments)
    worlds_by_id = {world.content_id: world for world in challenge.worlds}
    rows: list[Exp301EvaluationRow] = []
    for commitment in commitments:
        validate_prediction_commitment(commitment)
        if commitment.root != challenge.root:
            raise ValueError("prediction commitment root mismatch")
        if commitment.generation_token_count != EXP301_MAX_GENERATION_TOKENS:
            raise ValueError("scientific root evidence requires the frozen decode budget")
        if commitment.compute_match_status != "VALID_COMPUTE_MATCH":
            raise ValueError("scientific root evidence contains an invalid compute match")
        try:
            world = worlds_by_id[commitment.content_id]
        except KeyError as exc:
            raise ValueError("prediction commitment references an unknown challenge world") from exc
        rows.append(score_committed_prediction(commitment, world))
    return tuple(rows)


def build_root_evidence_artifact(
    *,
    frozen_implementation_digest: str,
    selection_manifest: RootSelectionManifest,
    challenge,
    runtime_identity: RuntimeRootIdentity,
    challenge_beacon: str,
    commitments: Iterable[PredictionCommitment],
    evaluation_rows: Iterable[Exp301EvaluationRow],
) -> RootScientificEvidenceArtifact:
    _require_hex(frozen_implementation_digest, field="frozen_implementation_digest")
    if not isinstance(challenge_beacon, str) or not challenge_beacon.strip():
        raise ValueError("challenge beacon must be non-empty")
    _rebuild_selection(selection_manifest)
    if selection_manifest.root != challenge.root:
        raise ValueError("selection/challenge root mismatch")

    rematerialized = materialize_root_challenge(
        root=challenge.root,
        beacon=challenge_beacon,
        frozen_implementation_digest=frozen_implementation_digest,
    )
    if rematerialized != challenge:
        raise ValueError("challenge materialization does not match frozen beacon identity")

    rebuilt_runtime = build_runtime_identity_for_root(
        frozen_implementation_digest=frozen_implementation_digest,
        selection_manifest=selection_manifest,
        challenge=challenge,
        beacon=challenge_beacon,
    )
    if rebuilt_runtime != runtime_identity:
        raise ValueError("runtime root identity mismatch")
    if not runtime_identity.scientific_evidence_eligible:
        raise ValueError("root scientific evidence must be evidence-eligible")

    materialized_commitments = tuple(commitments)
    materialized_rows = tuple(evaluation_rows)
    recomputed_rows = _recompute_rows(challenge, materialized_commitments)
    if recomputed_rows != materialized_rows:
        raise ValueError("evaluation rows do not match recomputed verifier results")

    values = dict(
        schema=ROOT_EVIDENCE_SCHEMA,
        frozen_implementation_digest=frozen_implementation_digest,
        root=challenge.root,
        run_identity=runtime_identity.run_identity,
        selection_manifest=selection_manifest,
        challenge_beacon=challenge_beacon,
        challenge_nonce=challenge.challenge_nonce,
        challenge_materialization_digest=challenge.materialization_digest,
        commitments=materialized_commitments,
        evaluation_rows=materialized_rows,
        scientific_evidence_eligible=True,
    )
    provisional = RootScientificEvidenceArtifact(**values, artifact_digest="")
    return RootScientificEvidenceArtifact(
        **values,
        artifact_digest=canonical_root_artifact_digest(provisional),
    )


def write_root_evidence_artifact(
    path: str | Path,
    artifact: RootScientificEvidenceArtifact,
) -> None:
    if canonical_root_artifact_digest(artifact) != artifact.artifact_digest:
        raise ValueError("root evidence artifact digest mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(artifact), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _selection_from_raw(raw: object) -> RootSelectionManifest:
    if not isinstance(raw, dict):
        raise ValueError("root evidence selection manifest must be an object")
    arm_items = raw.get("arm_selections")
    if not isinstance(arm_items, (list, tuple)):
        raise ValueError("root evidence arm selections must be a sequence")
    try:
        receipts = tuple(
            ArmSelectionReceipt(
                schema=item["schema"],
                arm_id=item["arm_id"],
                root=item["root"],
                selected_learning_rate=item["selected_learning_rate"],
                selected_checkpoint_digest=item["selected_checkpoint_digest"],
                selected_trial_receipt_digest=item["selected_trial_receipt_digest"],
                trial_receipt_digests=tuple(item["trial_receipt_digests"]),
                selection_digest=item["selection_digest"],
            )
            for item in arm_items
        )
        selection = RootSelectionManifest(
            schema=raw["schema"],
            root=raw["root"],
            arm_selections=receipts,
            selection_manifest_digest=raw["selection_manifest_digest"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("root evidence selection manifest schema mismatch") from exc
    return _rebuild_selection(selection)


def _artifact_from_raw(raw: object) -> RootScientificEvidenceArtifact:
    if not isinstance(raw, dict):
        raise ValueError("root evidence must be a JSON object")
    expected = {item.name for item in fields(RootScientificEvidenceArtifact)}
    if set(raw) != expected:
        raise ValueError("root evidence artifact field set mismatch")
    try:
        selection = _selection_from_raw(raw["selection_manifest"])
        commitments = tuple(PredictionCommitment(**item) for item in raw["commitments"])
        rows = tuple(Exp301EvaluationRow(**item) for item in raw["evaluation_rows"])
        return RootScientificEvidenceArtifact(
            schema=raw["schema"],
            frozen_implementation_digest=raw["frozen_implementation_digest"],
            root=raw["root"],
            run_identity=raw["run_identity"],
            selection_manifest=selection,
            challenge_beacon=raw["challenge_beacon"],
            challenge_nonce=raw["challenge_nonce"],
            challenge_materialization_digest=raw["challenge_materialization_digest"],
            commitments=commitments,
            evaluation_rows=rows,
            scientific_evidence_eligible=raw["scientific_evidence_eligible"],
            artifact_digest=raw["artifact_digest"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("root evidence artifact schema mismatch") from exc


def load_and_audit_root_evidence(path: str | Path) -> RootScientificEvidenceArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact = _artifact_from_raw(raw)
    if artifact.schema != ROOT_EVIDENCE_SCHEMA:
        raise ValueError("root evidence artifact schema mismatch")
    _require_hex(artifact.frozen_implementation_digest, field="frozen_implementation_digest")
    _require_hex(artifact.run_identity, field="run_identity")
    if not artifact.scientific_evidence_eligible:
        raise ValueError("root evidence is not scientific-evidence eligible")
    if canonical_root_artifact_digest(artifact) != artifact.artifact_digest:
        raise ValueError("root evidence artifact digest mismatch")

    challenge = materialize_root_challenge(
        root=artifact.root,
        beacon=artifact.challenge_beacon,
        frozen_implementation_digest=artifact.frozen_implementation_digest,
    )
    if challenge.challenge_nonce != artifact.challenge_nonce:
        raise ValueError("root evidence challenge nonce mismatch")
    if challenge.materialization_digest != artifact.challenge_materialization_digest:
        raise ValueError("root evidence challenge materialization digest mismatch")
    runtime = build_runtime_identity_for_root(
        frozen_implementation_digest=artifact.frozen_implementation_digest,
        selection_manifest=artifact.selection_manifest,
        challenge=challenge,
        beacon=artifact.challenge_beacon,
    )
    if runtime.run_identity != artifact.run_identity:
        raise ValueError("root evidence run identity mismatch")

    recomputed_rows = _recompute_rows(challenge, artifact.commitments)
    if recomputed_rows != artifact.evaluation_rows:
        raise ValueError("root evidence evaluation rows do not match recomputed verifier results")
    return artifact
