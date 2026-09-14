from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import pytest

from nolane_ai.experiments.exp301_ceremony import (
    build_arm_selection_receipt,
    build_root_selection_manifest,
    build_runtime_identity_for_root,
    materialize_root_challenge,
)
from nolane_ai.experiments.exp301_evaluation import (
    EXP301_ARMS,
    EXP301_EFFORTS,
    commit_prediction,
    score_committed_prediction,
)
from nolane_ai.experiments.exp301_evidence import (
    build_root_evidence_artifact,
    load_and_audit_root_evidence,
    write_root_evidence_artifact,
)


FROZEN = "a" * 64
BEACON = "workflow-run-12345"


@dataclass(frozen=True, slots=True)
class TrialEvidence:
    arm_id: str
    root: int
    learning_rate: float
    model_init_seed: int
    training_steps: int
    development_family_balanced_score: float
    development_family_scores: tuple[tuple[str, float], ...]
    checkpoint_digest: str
    trial_receipt_digest: str


def _selection(root: int):
    receipts = []
    for arm_index, arm in enumerate(EXP301_ARMS):
        trials = []
        for trial_index, lr in enumerate((1e-4, 3e-4)):
            marker = format(1 + arm_index * 2 + trial_index, "x")
            trials.append(
                TrialEvidence(
                    arm_id=arm,
                    root=root,
                    learning_rate=lr,
                    model_init_seed=100 + arm_index * 2 + trial_index,
                    training_steps=512,
                    development_family_balanced_score=0.8 - trial_index * 0.1,
                    development_family_scores=(("family", 0.8 - trial_index * 0.1),),
                    checkpoint_digest=marker * 64,
                    trial_receipt_digest=(format(8 + arm_index * 2 + trial_index, "x")[-1]) * 64,
                )
            )
        receipts.append(build_arm_selection_receipt(trials))
    return build_root_selection_manifest(receipts)


def _complete_root_material(root: int = 0):
    selection = _selection(root)
    challenge = materialize_root_challenge(
        root=root,
        beacon=BEACON,
        frozen_implementation_digest=FROZEN,
    )
    runtime = build_runtime_identity_for_root(
        frozen_implementation_digest=FROZEN,
        selection_manifest=selection,
        challenge=challenge,
        beacon=BEACON,
    )
    commitments = []
    rows = []
    for world in challenge.worlds:
        for arm in EXP301_ARMS:
            for effort in EXP301_EFFORTS:
                commitment = commit_prediction(
                    world,
                    arm_id=arm,
                    root=root,
                    effort_multiplier=effort,
                    candidate_answer=world.canonical_answer,
                    generation_token_count=96,
                )
                commitments.append(commitment)
                rows.append(score_committed_prediction(commitment, world))
    return selection, challenge, runtime, tuple(commitments), tuple(rows)


def test_root_evidence_roundtrip_recomputes_verifier_rows_and_binds_runtime_identity(tmp_path: Path) -> None:
    selection, challenge, runtime, commitments, rows = _complete_root_material()
    artifact = build_root_evidence_artifact(
        frozen_implementation_digest=FROZEN,
        selection_manifest=selection,
        challenge=challenge,
        runtime_identity=runtime,
        challenge_beacon=BEACON,
        commitments=commitments,
        evaluation_rows=rows,
    )
    assert artifact.root == 0
    assert artifact.run_identity == runtime.run_identity
    assert artifact.scientific_evidence_eligible is True
    assert len(artifact.commitments) == 512 * 3 * 6
    assert len(artifact.evaluation_rows) == 512 * 3 * 6
    assert len(artifact.artifact_digest) == 64

    path = tmp_path / "root-evidence.json"
    write_root_evidence_artifact(path, artifact)
    loaded = load_and_audit_root_evidence(path)
    assert loaded == artifact
    with pytest.raises(FileExistsError):
        write_root_evidence_artifact(path, artifact)


def test_root_evidence_audit_rejects_row_or_commitment_tampering(tmp_path: Path) -> None:
    selection, challenge, runtime, commitments, rows = _complete_root_material()
    artifact = build_root_evidence_artifact(
        frozen_implementation_digest=FROZEN,
        selection_manifest=selection,
        challenge=challenge,
        runtime_identity=runtime,
        challenge_beacon=BEACON,
        commitments=commitments,
        evaluation_rows=rows,
    )

    bad_row = replace(rows[0], verified_success=not rows[0].verified_success)
    with pytest.raises(ValueError, match="evaluation rows"):
        build_root_evidence_artifact(
            frozen_implementation_digest=FROZEN,
            selection_manifest=selection,
            challenge=challenge,
            runtime_identity=runtime,
            challenge_beacon=BEACON,
            commitments=commitments,
            evaluation_rows=(bad_row, *rows[1:]),
        )

    tampered = asdict(artifact)
    tampered["commitments"][0]["candidate_answer"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|commitment|artifact"):
        load_and_audit_root_evidence(path)
