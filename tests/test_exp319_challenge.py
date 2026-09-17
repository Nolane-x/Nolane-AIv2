from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib

import pytest

from nolane_ai.experiments.exp301_training import Exp301ByteTokenizer
from nolane_ai.experiments.exp319_contract import COMMON_GATING_EFFORT, PRIMARY_ARMS
from nolane_ai.experiments.exp319_worlds import materialize_stage_c


def _challenge_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_challenge")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 Stage C challenge contract is not implemented yet: {exc}")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _selection_seals(challenge, *, nrs_fail_roots: tuple[int, ...] = ()):
    seals = []
    for arm_id in PRIMARY_ARMS:
        for root in (1, 2, 3, 4):
            passes = not (arm_id == "C_NRS_CORE" and root in nrs_fail_roots)
            seals.append(
                challenge.StageBSelectionSeal(
                    arm_id=arm_id,
                    root=root,
                    selected_step=1024,
                    passes_floor=passes,
                    checkpoint_digest=_digest(f"checkpoint:{arm_id}:{root}"),
                    selection_digest=_digest(f"selection:{arm_id}:{root}"),
                )
            )
    return tuple(seals)


def _authority(challenge):
    return challenge.seal_stage_b_selection_authority(
        _selection_seals(challenge, nrs_fail_roots=(4,))
    )


def _manifest(challenge, *, root: int = 1, beacon: str = "github-run-123-exp319-v1"):
    return challenge.materialize_stage_c_manifest(
        root=root,
        challenge_beacon=beacon,
        source_digest=_digest("source"),
        selection_authority=_authority(challenge),
    )


def _encoded_answer(answer: str) -> tuple[int, ...]:
    tokenizer = Exp301ByteTokenizer()
    return (*tokenizer.encode_text(answer), tokenizer.eos_id)


def _outputs_for_inputs(challenge, inputs, *, answer_by_content=None):
    outputs = []
    for item in inputs:
        for arm_id in PRIMARY_ARMS:
            answer = "x" if answer_by_content is None else answer_by_content[item.content_id]
            outputs.append(
                challenge.StageCPredictionOutput(
                    content_id=item.content_id,
                    arm_id=arm_id,
                    candidate_answer=answer,
                    generated_token_ids=_encoded_answer(answer),
                    stopped_on_eos=True,
                )
            )
    return tuple(outputs)


def test_stage_b_selection_authority_requires_complete_sealed_cross_root_gate() -> None:
    challenge = _challenge_module()
    authority = _authority(challenge)

    assert authority.roots == (1, 2, 3, 4)
    assert authority.primary_arms == PRIMARY_ARMS
    assert len(authority.selections) == 8
    assert len(authority.authority_digest) == 64

    with pytest.raises(ValueError, match="complete|eight|8"):
        challenge.seal_stage_b_selection_authority(_selection_seals(challenge)[:-1])

    with pytest.raises(ValueError, match="cross-root"):
        challenge.seal_stage_b_selection_authority(
            _selection_seals(challenge, nrs_fail_roots=(3, 4))
        )


def test_stage_c_manifest_is_post_selection_run_bound_and_answer_blind() -> None:
    challenge = _challenge_module()
    authority = _authority(challenge)
    manifest = _manifest(challenge)

    assert manifest.root == 1
    assert manifest.selection_roots == (1, 2, 3, 4)
    assert manifest.primary_arms == PRIMARY_ARMS
    assert manifest.effort == COMMON_GATING_EFFORT == 4
    assert manifest.gradient_updates == 0
    assert manifest.world_count == 512
    assert len(manifest.content_ids) == 512
    assert len(set(manifest.content_ids)) == 512
    assert manifest.stage_b_selection_digest == authority.authority_digest
    assert len(manifest.generator_manifest_digest) == 64
    assert len(manifest.manifest_digest) == 64
    assert "canonical_answer" not in asdict(manifest)

    assert _manifest(challenge) == manifest
    changed = _manifest(challenge, beacon="github-run-124-exp319-v1")
    assert changed.manifest_digest != manifest.manifest_digest
    assert changed.content_ids != manifest.content_ids

    with pytest.raises(ValueError, match="source_digest"):
        challenge.materialize_stage_c_manifest(
            root=1,
            challenge_beacon="github-run-123-exp319-v1",
            source_digest="not-a-digest",
            selection_authority=authority,
        )


def test_stage_c_shards_cover_each_root_exactly_once_without_truth_labels() -> None:
    challenge = _challenge_module()
    manifest = _manifest(challenge)

    assert challenge.STAGE_C_SHARD_COUNT == 16
    assert challenge.STAGE_C_WORLDS_PER_SHARD == 32

    all_ids = []
    for shard_index in range(challenge.STAGE_C_SHARD_COUNT):
        inputs = challenge.prediction_inputs_for_shard(manifest, shard_index=shard_index)
        assert len(inputs) == 32
        assert all(item.root == manifest.root for item in inputs)
        assert all(not hasattr(item, "canonical_answer") for item in inputs)
        all_ids.extend(item.content_id for item in inputs)

    assert len(all_ids) == 512
    assert len(set(all_ids)) == 512
    assert set(all_ids) == set(manifest.content_ids)

    with pytest.raises(ValueError, match="shard"):
        challenge.prediction_inputs_for_shard(manifest, shard_index=-1)
    with pytest.raises(ValueError, match="shard"):
        challenge.prediction_inputs_for_shard(manifest, shard_index=16)


def test_prediction_shards_are_commitments_only_at_effort_four() -> None:
    challenge = _challenge_module()
    manifest = _manifest(challenge)
    inputs = challenge.prediction_inputs_for_shard(manifest, shard_index=0)
    outputs = _outputs_for_inputs(challenge, inputs)
    shard = challenge.build_prediction_shard(
        manifest,
        shard_index=0,
        outputs=outputs,
    )

    assert shard.shard_index == 0
    assert shard.world_count == 32
    assert len(shard.commitments) == 64
    assert all(item.effort == 4 for item in shard.commitments)
    assert all(item.challenge_manifest_digest == manifest.manifest_digest for item in shard.commitments)
    for item in shard.commitments:
        payload = asdict(item)
        assert "exact" not in payload
        assert "score" not in payload
        assert "canonical_answer" not in payload
        assert len(item.commitment_digest) == 64

    with pytest.raises(ValueError, match="complete|grid"):
        challenge.build_prediction_shard(
            manifest,
            shard_index=0,
            outputs=outputs[:-1],
        )
    with pytest.raises(ValueError, match="duplicate|grid"):
        challenge.build_prediction_shard(
            manifest,
            shard_index=0,
            outputs=(*outputs, outputs[0]),
        )
    with pytest.raises(ValueError, match="arm"):
        challenge.build_prediction_shard(
            manifest,
            shard_index=0,
            outputs=(replace(outputs[0], arm_id="B_LOOP_SIMPLE"), *outputs[1:]),
        )


def test_stage_c_merge_requires_all_sixteen_unique_shards_before_scoring() -> None:
    challenge = _challenge_module()
    manifest = _manifest(challenge)
    shards = []
    for shard_index in range(16):
        inputs = challenge.prediction_inputs_for_shard(manifest, shard_index=shard_index)
        shards.append(
            challenge.build_prediction_shard(
                manifest,
                shard_index=shard_index,
                outputs=_outputs_for_inputs(challenge, inputs),
            )
        )

    commitments = challenge.merge_prediction_shards(manifest, tuple(shards))
    assert len(commitments) == 1024

    with pytest.raises(ValueError, match="16|complete"):
        challenge.merge_prediction_shards(manifest, tuple(shards[:-1]))
    with pytest.raises(ValueError, match="duplicate|16|complete"):
        challenge.merge_prediction_shards(manifest, tuple((*shards[:-1], shards[0])))

    with pytest.raises(ValueError, match="complete commitment grid"):
        challenge.score_stage_c_commitments(manifest, commitments[:-1])


def test_stage_c_scoring_happens_only_after_commitment_and_reproduces_truth() -> None:
    challenge = _challenge_module()
    manifest = _manifest(challenge)
    worlds = materialize_stage_c(
        root=manifest.root,
        challenge_beacon=manifest.challenge_beacon,
        stage_b_selection_digest=manifest.stage_b_selection_digest,
    )
    answers = {world.content_id: world.canonical_answer for world in worlds}

    shards = []
    for shard_index in range(16):
        inputs = challenge.prediction_inputs_for_shard(manifest, shard_index=shard_index)
        shards.append(
            challenge.build_prediction_shard(
                manifest,
                shard_index=shard_index,
                outputs=_outputs_for_inputs(
                    challenge,
                    inputs,
                    answer_by_content=answers,
                ),
            )
        )

    commitments = challenge.merge_prediction_shards(manifest, tuple(shards))
    scored = challenge.score_stage_c_commitments(manifest, commitments)
    assert len(scored) == 1024
    assert all(row.exact is True for row in scored)
    assert all(row.answer_token_accuracy == pytest.approx(1.0) for row in scored)
    assert all(row.eos_correct is True for row in scored)
    assert all(row.invalid_output is False for row in scored)
    assert {row.arm_id for row in scored} == set(PRIMARY_ARMS)
    assert {row.family for row in scored} == {
        "iterative-grid-and-maze",
        "algorithmic-sequence-transform",
        "generator-heldout-abstract-transformation",
        "language-sequence-control",
    }
