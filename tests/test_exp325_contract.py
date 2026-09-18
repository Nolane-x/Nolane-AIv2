from __future__ import annotations

from dataclasses import replace

from nolane_ai.experiments.exp325_contract import (
    AUTHORIZATION_FLAGS,FAMILIES,LOCAL_CHECKPOINTS,WORLD_KEYS,
    WorldSnapshot,preregistration_digest,preregistration_payload,reduce_world_isolation,
)


def row(family,index,update,token=0.5,full=0.0):
    return WorldSnapshot(
        family=family,world_index=index,local_update=update,
        teacher_forced_answer_token_accuracy=token,
        teacher_forced_full_answer_exact=full,
        greedy_exact=full,
        answer_only_loss=1.0,
        gradient_norm_preclip=1.0,
        parameter_update_norm_ratio=0.01,
        nonfinite_events=0,
    )


def grid(passing=()):
    passing=set(passing)
    return {
        key: tuple(
            row(key[0],key[1],u,token=(1.0 if key in passing and u==32 else 0.5),full=(1.0 if key in passing and u==32 else 0.0))
            for u in LOCAL_CHECKPOINTS
        )
        for key in WORLD_KEYS
    }


def test_preregistration_binds_parent_and_matched_exposure_budget() -> None:
    p=preregistration_payload()
    assert p["authority"]["parent_exp324_disposition"]=="NO_FAMILIES_ISOLATED_FIT"
    assert p["training"]["optimizer_updates_per_world"]==32
    assert p["training"]["independent_clone_per_world"] is True
    assert p["population"]["total_worlds"]==32
    assert preregistration_digest()=="61e82d4fe6b556078243e35e57d4e4e247b8d804f34c73e085a50c5658ac5c61"


def test_every_authorization_remains_false() -> None:
    assert AUTHORIZATION_FLAGS and not any(AUTHORIZATION_FLAGS.values())


def test_all_some_and_none_dispositions() -> None:
    d,passed=reduce_world_isolation(grid(WORLD_KEYS))
    assert d=="ALL_WORLDS_SINGLE_FIT" and len(passed)==32

    target=(WORLD_KEYS[0],WORLD_KEYS[-1])
    d,passed=reduce_world_isolation(grid(target))
    assert d=="SOME_WORLDS_SINGLE_FIT" and set(passed)==set(target)

    d,passed=reduce_world_isolation(grid())
    assert d=="NO_WORLDS_SINGLE_FIT" and passed==()


def test_split_floor_across_checkpoints_does_not_pass() -> None:
    records=grid()
    key=WORLD_KEYS[0]
    rows=list(records[key])
    rows[0]=replace(rows[0],teacher_forced_answer_token_accuracy=1.0,teacher_forced_full_answer_exact=0.0)
    rows[1]=replace(rows[1],teacher_forced_answer_token_accuracy=0.5,teacher_forced_full_answer_exact=1.0)
    records[key]=tuple(rows)
    d,passed=reduce_world_isolation(records)
    assert key not in passed
    assert d=="NO_WORLDS_SINGLE_FIT"


def test_missing_duplicate_misordered_or_nonfinite_geometry_fails_closed() -> None:
    records=grid(); records.pop(WORLD_KEYS[-1])
    assert reduce_world_isolation(records)[0]=="INVALID_WORLD_ISOLATION"

    records=grid(); key=WORLD_KEYS[0]; records[key]=tuple(reversed(records[key]))
    assert reduce_world_isolation(records)[0]=="INVALID_WORLD_ISOLATION"

    records=grid(); key=WORLD_KEYS[0]; rows=list(records[key]); rows[-1]=replace(rows[-1],nonfinite_events=1); records[key]=tuple(rows)
    assert reduce_world_isolation(records)[0]=="INVALID_WORLD_ISOLATION"


def test_world_geometry_is_exactly_four_families_times_eight_indices() -> None:
    assert len(FAMILIES)==4
    assert len(WORLD_KEYS)==32
    assert set(index for _,index in WORLD_KEYS)==set(range(8))
