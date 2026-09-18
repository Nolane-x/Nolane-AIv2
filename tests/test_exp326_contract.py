from __future__ import annotations

from dataclasses import replace

from nolane_ai.experiments.exp326_contract import (
    AUTHORIZATION_FLAGS,
    EXPOSURE_CHECKPOINTS,
    FAMILIES,
    GROUP_IDS,
    GROUP_MEMBERS,
    GroupSnapshot,
    expected_effort_trace,
    preregistration_digest,
    preregistration_payload,
    reduce_breakpoint,
)


def row(family: str, group_id: str, exposures: int, *, passed: bool = True) -> GroupSnapshot:
    n=len(GROUP_MEMBERS[group_id])
    tokens=(1.0,)*n if passed else (0.98,)+(1.0,)*(n-1)
    full=(1.0,)*n
    return GroupSnapshot(
        family=family,
        group_id=group_id,
        exposures_per_world=exposures,
        total_optimizer_updates=n*exposures,
        world_token_accuracies=tokens,
        world_full_answer_exact=full,
    )


def grid(overrides=None):
    overrides=overrides or {}
    return {
        (family,gid): tuple(
            row(family,gid,e,passed=overrides.get((family,gid),True))
            for e in EXPOSURE_CHECKPOINTS
        )
        for family in FAMILIES
        for gid in GROUP_IDS
    }


def test_preregistration_binds_exp325_and_balanced_effort_geometry() -> None:
    p=preregistration_payload()
    assert p["authority"]["parent_exp325_disposition"]=="ALL_WORLDS_SINGLE_FIT"
    assert p["authority"]["parent_exp325_final_evidence_digest"]=="0ce6a8bd4f96864bf754b22fc7c9b95552f6081225a4c305394d92ed39e7cbc2"
    assert p["training"]["exposures_per_world"]==32
    assert p["training"]["effort_cycle"]==[1,2,4,8]
    assert preregistration_digest()=="fc301d29766860a9b48d41ff91f3702c0731217b4f6a774f5e6e2db9fd4efd6e"


def test_every_world_gets_exactly_eight_of_each_effort() -> None:
    trace=expected_effort_trace()
    assert len(trace)==32
    assert tuple(trace[:8])==(1,2,4,8,1,2,4,8)
    for effort in (1,2,4,8):
        assert trace.count(effort)==8


def test_all_registered_groups_fit_has_no_break() -> None:
    decision,passed,violations=reduce_breakpoint(grid())
    assert decision=="NO_BREAK_THROUGH_EIGHT"
    assert all(passed.values())
    assert violations==()


def test_pair_break_has_earliest_precedence() -> None:
    decision,_,_=reduce_breakpoint(grid({("iterative-grid-and-maze","P01"):False}))
    assert decision=="PAIR_LEVEL_BREAK_PRESENT"


def test_quartet_break_requires_all_pairs_to_fit() -> None:
    decision,_,_=reduce_breakpoint(grid({("algorithmic-sequence-transform","Q0123"):False}))
    assert decision=="QUARTET_LEVEL_BREAK_PRESENT"


def test_octet_break_requires_pairs_and_quartets_to_fit() -> None:
    decision,_,_=reduce_breakpoint(grid({("language-sequence-control","O01234567"):False}))
    assert decision=="OCTET_LEVEL_BREAK_PRESENT"


def test_nonmonotonic_parent_pass_child_fail_is_separate_disposition() -> None:
    records=grid({("generator-heldout-abstract-transformation","P01"):False})
    # Force the parent quartet to remain passing: this is intentionally non-monotonic.
    decision,_,violations=reduce_breakpoint(records)
    assert decision=="NONMONOTONIC_GROUP_FIT"
    assert ("generator-heldout-abstract-transformation","Q0123","P01") in violations


def test_missing_or_bad_geometry_fails_closed() -> None:
    records=grid()
    records.pop((FAMILIES[0],"P01"))
    assert reduce_breakpoint(records)[0]=="INVALID_BREAKPOINT_COURT"

    records=grid()
    key=(FAMILIES[0],"P01")
    broken=list(records[key])
    broken[-1]=replace(broken[-1],total_optimizer_updates=999)
    records[key]=tuple(broken)
    assert reduce_breakpoint(records)[0]=="INVALID_BREAKPOINT_COURT"


def test_authorization_remains_false() -> None:
    assert AUTHORIZATION_FLAGS
    assert not any(AUTHORIZATION_FLAGS.values())
