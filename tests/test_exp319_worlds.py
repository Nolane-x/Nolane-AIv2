from __future__ import annotations

import importlib

import pytest


def _worlds_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_worlds")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 world generator is not implemented yet: {exc}")


def test_stage_a_is_fresh_deterministic_and_exactly_8_per_family() -> None:
    worlds = _worlds_module()

    first = worlds.materialize_stage_a()
    second = worlds.materialize_stage_a()

    assert worlds.WORLD_GENERATOR_VERSION == "exp319-worlds-v1"
    assert len(first) == 32
    assert first == second
    assert {world.stage for world in first} == {"A_SANITY"}
    assert {world.root for world in first} == {0}
    assert all(world.content_id.startswith("exp319:") for world in first)
    for family in worlds.EXP319_TASK_FAMILIES:
        family_worlds = [world for world in first if world.family == family]
        assert len(family_worlds) == 8
        assert {world.index for world in family_worlds} == set(range(8))


def test_stage_b_train_and_iid_are_root_bound_disjoint_and_not_stage_a() -> None:
    worlds = _worlds_module()

    stage_a = worlds.materialize_stage_a()
    train = worlds.materialize_stage_b(root=2, split="train")
    iid = worlds.materialize_stage_b(root=2, split="iid")

    assert len(train) == 512
    assert len(iid) == 256
    assert {world.root for world in train + iid} == {2}
    assert {world.stage for world in train} == {"B_TRAIN"}
    assert {world.stage for world in iid} == {"B_IID"}

    a_ids = {world.content_id for world in stage_a}
    train_ids = {world.content_id for world in train}
    iid_ids = {world.content_id for world in iid}
    assert a_ids.isdisjoint(train_ids)
    assert a_ids.isdisjoint(iid_ids)
    assert train_ids.isdisjoint(iid_ids)

    a_generators = {world.generator_identity for world in stage_a}
    b_generators = {world.generator_identity for world in train + iid}
    assert a_generators.isdisjoint(b_generators)


def test_stage_c_requires_post_selection_authority_and_template_shift() -> None:
    worlds = _worlds_module()
    selection_digest = "a" * 64

    with pytest.raises(ValueError, match="challenge_beacon"):
        worlds.materialize_stage_c(
            root=1,
            challenge_beacon="",
            stage_b_selection_digest=selection_digest,
        )
    with pytest.raises(ValueError, match="stage_b_selection_digest"):
        worlds.materialize_stage_c(
            root=1,
            challenge_beacon="github-run-123-exp319-v1",
            stage_b_selection_digest="",
        )
    with pytest.raises(ValueError, match="EXP-301"):
        worlds.materialize_stage_c(
            root=1,
            challenge_beacon="exp301-legacy-challenge-nonce",
            stage_b_selection_digest=selection_digest,
        )

    b_train = worlds.materialize_stage_b(root=1, split="train")
    b_iid = worlds.materialize_stage_b(root=1, split="iid")
    heldout = worlds.materialize_stage_c(
        root=1,
        challenge_beacon="github-run-123-exp319-v1",
        stage_b_selection_digest=selection_digest,
    )

    assert len(heldout) == 512
    assert {world.stage for world in heldout} == {"C_HELDOUT"}
    assert {world.root for world in heldout} == {1}
    b_templates = {world.template_identity for world in b_train + b_iid}
    c_templates = {world.template_identity for world in heldout}
    assert b_templates.isdisjoint(c_templates)

    rematerialized = worlds.materialize_stage_c(
        root=1,
        challenge_beacon="github-run-123-exp319-v1",
        stage_b_selection_digest=selection_digest,
    )
    changed_beacon = worlds.materialize_stage_c(
        root=1,
        challenge_beacon="github-run-124-exp319-v1",
        stage_b_selection_digest=selection_digest,
    )
    assert heldout == rematerialized
    assert {world.content_id for world in heldout} != {
        world.content_id for world in changed_beacon
    }


def test_content_ids_bind_stage_root_family_template_index_and_stage_c_authority() -> None:
    worlds = _worlds_module()
    family = worlds.EXP319_TASK_FAMILIES[0]

    a = worlds.generate_world_instance(family=family, stage="A_SANITY", root=0, index=0)
    b_root_1 = worlds.generate_world_instance(family=family, stage="B_TRAIN", root=1, index=0)
    b_root_2 = worlds.generate_world_instance(family=family, stage="B_TRAIN", root=2, index=0)
    c = worlds.generate_world_instance(
        family=family,
        stage="C_HELDOUT",
        root=1,
        index=0,
        challenge_beacon="github-run-123-exp319-v1",
        stage_b_selection_digest="b" * 64,
    )

    assert len({a.content_id, b_root_1.content_id, b_root_2.content_id, c.content_id}) == 4
    assert a.template_identity != b_root_1.template_identity
    assert c.template_identity != b_root_1.template_identity
    assert c.challenge_beacon == "github-run-123-exp319-v1"
    assert c.stage_b_selection_digest == "b" * 64


def test_world_json_round_trip_and_manifest_digest_are_canonical() -> None:
    worlds = _worlds_module()
    materialized = worlds.materialize_stage_c(
        root=3,
        challenge_beacon="github-run-900-exp319-v1",
        stage_b_selection_digest="c" * 64,
    )

    sample = materialized[0]
    assert worlds.Exp319WorldInstance.from_json_dict(sample.to_json_dict()) == sample
    assert isinstance(sample.operation_trace, tuple)
    assert isinstance(sample.tags, tuple)

    first_manifest = worlds.build_generator_manifest(materialized)
    second_manifest = worlds.build_generator_manifest(tuple(reversed(materialized)))
    assert first_manifest["schema"] == "EXP319-GENERATOR-MANIFEST-V1"
    assert first_manifest["world_count"] == 512
    assert first_manifest["digest"] == second_manifest["digest"]
    assert first_manifest["content_ids"] == sorted(first_manifest["content_ids"])
