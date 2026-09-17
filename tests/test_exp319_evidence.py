from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments import exp319_challenge as challenge
from nolane_ai.experiments.exp301_training import Exp301ByteTokenizer
from nolane_ai.experiments.exp319_contract import AUTHORIZATION_FLAGS, DISPOSITIONS, PRIMARY_ARMS
from nolane_ai.experiments.exp319_worlds import materialize_stage_c


def _evidence_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_evidence")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 evidence/reducer is not implemented yet: {exc}")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stage_a_seals(evidence, *, control: bool = True, nrs: bool = True):
    return (
        evidence.StageASelectionSeal(
            arm_id="A_FIXED",
            passes_floor=control,
            selected_learning_rate=1e-4,
            selection_digest=_digest("stage-a:A_FIXED"),
        ),
        evidence.StageASelectionSeal(
            arm_id="C_NRS_CORE",
            passes_floor=nrs,
            selected_learning_rate=1e-4,
            selection_digest=_digest("stage-a:C_NRS_CORE"),
        ),
    )


def _root_fixture(
    evidence,
    *,
    root: int,
    stage_a_control: bool = True,
    stage_a_nrs: bool = True,
    stage_b_control: bool = True,
    stage_b_nrs: bool = True,
    stage_c_control: bool = True,
    stage_c_nrs: bool = True,
):
    return evidence._seal_root_evidence(
        root=root,
        provenance_valid=True,
        source_digest=_digest("source"),
        training_contract_digest=_digest("training-contract"),
        scoring_contract_digest=_digest("scoring-contract"),
        stage_a_selections=_stage_a_seals(
            evidence,
            control=stage_a_control,
            nrs=stage_a_nrs,
        ),
        stage_b_selection_authority_digest=_digest("stage-b-authority"),
        stage_b_control_pass=stage_b_control,
        stage_b_nrs_pass=stage_b_nrs,
        stage_c_manifest_digest=_digest(f"manifest:{root}"),
        stage_c_commitment_grid_digest=_digest(f"commitments:{root}"),
        stage_c_scoring_digest=_digest(f"scoring:{root}"),
        stage_c_control_pass=stage_c_control,
        stage_c_nrs_pass=stage_c_nrs,
    )


def _four_roots(
    evidence,
    *,
    stage_a_control: bool = True,
    stage_a_nrs: bool = True,
    stage_b_control_pass_roots=(1, 2, 3, 4),
    stage_b_nrs_pass_roots=(1, 2, 3, 4),
    stage_c_control_pass_roots=(1, 2, 3, 4),
    stage_c_nrs_pass_roots=(1, 2, 3, 4),
):
    return tuple(
        _root_fixture(
            evidence,
            root=root,
            stage_a_control=stage_a_control,
            stage_a_nrs=stage_a_nrs,
            stage_b_control=root in stage_b_control_pass_roots,
            stage_b_nrs=root in stage_b_nrs_pass_roots,
            stage_c_control=root in stage_c_control_pass_roots,
            stage_c_nrs=root in stage_c_nrs_pass_roots,
        )
        for root in (1, 2, 3, 4)
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        ({"stage_a_control": False}, "TRAINING_STACK_NOT_LEARNABLE"),
        ({"stage_a_nrs": False}, "NRS_LOCAL_TRAINABILITY_FAIL"),
        ({"stage_b_control_pass_roots": (1, 2)}, "SUPERVISED_FOUNDATION_UNDERTRAINED"),
        ({"stage_b_nrs_pass_roots": (1, 2)}, "NRS_IID_GENERALIZATION_FLOOR_FAIL"),
        ({"stage_c_control_pass_roots": (1, 2)}, "SUPERVISED_HELDOUT_FOUNDATION_FAIL"),
        ({"stage_c_nrs_pass_roots": (1, 2)}, "NRS_HELDOUT_GENERALIZATION_FLOOR_FAIL"),
        ({}, "FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY"),
    ),
)
def test_closed_reducer_emits_each_capability_disposition_in_causal_order(kwargs, expected) -> None:
    evidence = _evidence_module()
    result = evidence.reduce_cross_root(_four_roots(evidence, **kwargs))

    assert result.decision == expected
    assert result.decision in DISPOSITIONS
    assert result.authorizations == AUTHORIZATION_FLAGS
    assert result.exp302_implementation_authorized is False
    assert result.exp320_implementation_authorized is False
    assert result.scale_authorized is False
    assert result.authorized_30m is False
    assert result.authorized_100m is False


def test_provenance_invalidity_precedes_every_capability_interpretation() -> None:
    evidence = _evidence_module()
    roots = list(
        _four_roots(
            evidence,
            stage_a_control=False,
            stage_a_nrs=False,
            stage_b_control_pass_roots=(),
            stage_b_nrs_pass_roots=(),
            stage_c_control_pass_roots=(),
            stage_c_nrs_pass_roots=(),
        )
    )
    roots[0] = replace(roots[0], artifact_digest="0" * 64)

    result = evidence.reduce_cross_root(tuple(roots))
    assert result.decision == "INVALID_DIAGNOSTIC"
    assert result.authorizations == AUTHORIZATION_FLAGS


def test_stage_c_control_failure_has_priority_over_nrs_failure() -> None:
    evidence = _evidence_module()
    result = evidence.reduce_cross_root(
        _four_roots(
            evidence,
            stage_c_control_pass_roots=(1, 2),
            stage_c_nrs_pass_roots=(1, 2),
        )
    )
    assert result.decision == "SUPERVISED_HELDOUT_FOUNDATION_FAIL"


def _selection_authority():
    selections = tuple(
        challenge.StageBSelectionSeal(
            arm_id=arm_id,
            root=root,
            selected_step=1024,
            passes_floor=True,
            checkpoint_digest=_digest(f"checkpoint:{arm_id}:{root}"),
            selection_digest=_digest(f"selection:{arm_id}:{root}"),
        )
        for arm_id in PRIMARY_ARMS
        for root in (1, 2, 3, 4)
    )
    return challenge.seal_stage_b_selection_authority(selections)


def _perfect_stage_c_grid(manifest):
    worlds = materialize_stage_c(
        root=manifest.root,
        challenge_beacon=manifest.challenge_beacon,
        stage_b_selection_digest=manifest.stage_b_selection_digest,
    )
    answers = {world.content_id: world.canonical_answer for world in worlds}
    tokenizer = Exp301ByteTokenizer()
    shards = []
    for shard_index in range(16):
        inputs = challenge.prediction_inputs_for_shard(manifest, shard_index=shard_index)
        outputs = []
        for item in inputs:
            answer = answers[item.content_id]
            token_ids = (*tokenizer.encode_text(answer), tokenizer.eos_id)
            for arm_id in PRIMARY_ARMS:
                outputs.append(
                    challenge.StageCPredictionOutput(
                        content_id=item.content_id,
                        arm_id=arm_id,
                        candidate_answer=answer,
                        generated_token_ids=token_ids,
                        stopped_on_eos=True,
                    )
                )
        shards.append(
            challenge.build_prediction_shard(
                manifest,
                shard_index=shard_index,
                outputs=tuple(outputs),
            )
        )
    commitments = challenge.merge_prediction_shards(manifest, tuple(shards))
    scored = challenge.score_stage_c_commitments(manifest, commitments)
    return commitments, scored


def test_root_evidence_binds_selection_commitment_scoring_and_frozen_thresholds() -> None:
    evidence = _evidence_module()
    authority = _selection_authority()
    manifest = challenge.materialize_stage_c_manifest(
        root=1,
        challenge_beacon="github-run-123-exp319-v1",
        source_digest=_digest("source"),
        selection_authority=authority,
    )
    commitments, scored = _perfect_stage_c_grid(manifest)

    root = evidence.build_root_evidence(
        root=1,
        source_digest=_digest("source"),
        training_contract_digest=_digest("training-contract"),
        scoring_contract_digest=_digest("scoring-contract"),
        stage_a_selections=_stage_a_seals(evidence),
        stage_b_selection_authority=authority,
        stage_c_manifest=manifest,
        stage_c_commitments=commitments,
        stage_c_scored_rows=scored,
    )

    assert root.provenance_valid is True
    assert root.stage_b_control_pass is True
    assert root.stage_b_nrs_pass is True
    assert root.stage_c_control_pass is True
    assert root.stage_c_nrs_pass is True
    assert root.stage_c_manifest_digest == manifest.manifest_digest
    assert len(root.stage_c_commitment_grid_digest) == 64
    assert len(root.stage_c_scoring_digest) == 64
    assert len(root.artifact_digest) == 64
    assert {summary.arm_id for summary in root.stage_c_summaries} == set(PRIMARY_ARMS)
    for summary in root.stage_c_summaries:
        assert summary.answer_token_accuracy == pytest.approx(1.0)
        assert summary.family_balanced_exact_match == pytest.approx(1.0)
        assert summary.nonzero_exact_families == 4
        assert summary.eos_correctness == pytest.approx(1.0)
        assert summary.invalid_output_rate == pytest.approx(0.0)
        assert summary.passes_floor is True


def test_valid_stage_b_stop_can_be_sealed_without_materializing_stage_c() -> None:
    evidence = _evidence_module()
    selections = tuple(
        challenge.StageBSelectionSeal(
            arm_id=arm_id,
            root=root,
            selected_step=1024,
            passes_floor=(arm_id != "A_FIXED" or root in (1, 2)),
            checkpoint_digest=_digest(f"stage-b-checkpoint:{arm_id}:{root}"),
            selection_digest=_digest(f"stage-b-selection:{arm_id}:{root}"),
        )
        for arm_id in PRIMARY_ARMS
        for root in (1, 2, 3, 4)
    )

    diagnostic_authority = challenge.seal_stage_b_diagnostic_authority(selections)
    with pytest.raises(ValueError, match="cross-root"):
        challenge.seal_stage_b_selection_authority(selections)

    roots = tuple(
        evidence.build_stage_b_root_evidence(
            root=root,
            source_digest=_digest("source"),
            training_contract_digest=_digest("training-contract"),
            scoring_contract_digest=_digest("scoring-contract"),
            stage_a_selections=_stage_a_seals(evidence),
            stage_b_selection_authority=diagnostic_authority,
        )
        for root in (1, 2, 3, 4)
    )
    assert all(item.stage_c_manifest_digest is None for item in roots)
    assert all(item.stage_c_commitment_grid_digest is None for item in roots)
    assert all(item.stage_c_scoring_digest is None for item in roots)
    assert all(item.stage_c_control_pass is None for item in roots)
    assert all(item.stage_c_nrs_pass is None for item in roots)

    result = evidence.reduce_cross_root(roots)
    assert result.decision == "SUPERVISED_FOUNDATION_UNDERTRAINED"
    assert result.stage_b_control_pass_roots == 2
    assert result.stage_b_nrs_pass_roots == 4
    assert result.stage_c_control_pass_roots is None
    assert result.stage_c_nrs_pass_roots is None
    assert result.authorizations == AUTHORIZATION_FLAGS
