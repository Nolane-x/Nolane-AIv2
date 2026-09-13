from __future__ import annotations

from nolane_ai.experiments.exp301_ceremony import (
    build_arm_selection_receipt,
    build_root_selection_manifest,
    build_runtime_identity_for_root,
    derive_challenge_nonce,
    materialize_root_challenge,
)
from nolane_ai.experiments.exp301_scientific_contract import ScientificTrialResult
from nolane_ai.experiments.exp301_worlds import materialize_world_set


def _trial(arm: str, root: int, lr: float, score: float, marker: str) -> ScientificTrialResult:
    return ScientificTrialResult(
        arm_id=arm,
        root=root,
        learning_rate=lr,
        model_init_seed=123,
        training_steps=512,
        development_family_balanced_score=score,
        development_family_scores=(("f", score),),
        checkpoint_digest=marker * 64,
        trial_receipt_digest=("a" if marker != "a" else "b") * 64,
    )


def test_arm_selection_receipt_binds_both_trials_and_selected_checkpoint() -> None:
    low = _trial("A_FIXED", 1, 1e-4, 0.50, "c")
    high = _trial("A_FIXED", 1, 3e-4, 0.60, "d")
    receipt = build_arm_selection_receipt((low, high))
    assert receipt.arm_id == "A_FIXED"
    assert receipt.root == 1
    assert receipt.selected_learning_rate == 3e-4
    assert receipt.selected_checkpoint_digest == "d" * 64
    assert set(receipt.trial_receipt_digests) == {low.trial_receipt_digest, high.trial_receipt_digest}
    assert len(receipt.selection_digest) == 64


def test_root_selection_manifest_requires_exactly_three_arms() -> None:
    markers = (("c", "d"), ("e", "f"), ("a", "b"))
    receipts = []
    for arm, (low_marker, high_marker) in zip(("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"), markers):
        receipts.append(
            build_arm_selection_receipt(
                (
                    _trial(arm, 0, 1e-4, 0.5, low_marker),
                    _trial(arm, 0, 3e-4, 0.4, high_marker),
                )
            )
        )
    manifest = build_root_selection_manifest(receipts)
    assert manifest.root == 0
    assert tuple(item.arm_id for item in manifest.arm_selections) == (
        "A_FIXED",
        "B_LOOP_SIMPLE",
        "C_NRS_CORE",
    )
    assert len(manifest.selection_manifest_digest) == 64


def test_challenge_nonce_is_post_freeze_beacon_bound() -> None:
    frozen = "1" * 64
    a = derive_challenge_nonce(beacon="run-100", frozen_implementation_digest=frozen)
    b = derive_challenge_nonce(beacon="run-101", frozen_implementation_digest=frozen)
    assert len(a) == 64
    assert a != b


def test_root_challenge_is_512_and_disjoint_from_public_splits() -> None:
    frozen = "2" * 64
    challenge = materialize_root_challenge(
        root=2,
        beacon="workflow-run-123",
        frozen_implementation_digest=frozen,
    )
    assert len(challenge.worlds) == 512
    assert {world.root for world in challenge.worlds} == {2}
    assert {world.split for world in challenge.worlds} == {"challenge"}
    assert len(challenge.materialization_digest) == 64

    train = materialize_world_set(split="train", roots=(2,), per_family=128)
    dev = materialize_world_set(split="development", roots=(2,), per_family=64)
    public_ids = {world.content_id for world in (*train, *dev)}
    assert not public_ids.intersection(world.content_id for world in challenge.worlds)


def test_runtime_identity_binds_selection_beacon_and_materialization() -> None:
    markers = (("c", "d"), ("e", "f"), ("a", "b"))
    arm_receipts = []
    for arm, (low_marker, high_marker) in zip(("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"), markers):
        arm_receipts.append(
            build_arm_selection_receipt(
                (
                    _trial(arm, 3, 1e-4, 0.6, low_marker),
                    _trial(arm, 3, 3e-4, 0.5, high_marker),
                )
            )
        )
    selection = build_root_selection_manifest(arm_receipts)
    frozen = "3" * 64
    challenge = materialize_root_challenge(
        root=3,
        beacon="workflow-run-456",
        frozen_implementation_digest=frozen,
    )
    identity = build_runtime_identity_for_root(
        frozen_implementation_digest=frozen,
        selection_manifest=selection,
        challenge=challenge,
        beacon="workflow-run-456",
    )
    assert identity.root == 3
    assert identity.selected_hyperparameter_receipt_digest == selection.selection_manifest_digest
    assert identity.challenge_materialization_digest == challenge.materialization_digest
    assert identity.scientific_evidence_eligible is True
    assert len(identity.run_identity) == 64
