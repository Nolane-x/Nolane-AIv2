from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp325_contract import FAMILIES, LOCAL_CHECKPOINTS, WORLD_INDICES
from nolane_ai.experiments.exp325_identity import build_execution_identity
from nolane_ai.experiments.exp325_runtime import (
    build_family_evidence,
    build_final_evidence,
    family_worlds,
)


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def record(family,index,passed=False):
    return {
        "family":family,
        "world_index":index,
        "content_id":"exp319:" + f"{index:064x}",
        "snapshots":[
            {
                "family":family,
                "world_index":index,
                "local_update":u,
                "teacher_forced_answer_token_accuracy":1.0 if passed and u==32 else 0.5,
                "teacher_forced_full_answer_exact":1.0 if passed and u==32 else 0.0,
                "greedy_exact":1.0 if passed and u==32 else 0.0,
                "answer_only_loss":0.1,
                "gradient_norm_preclip":1.0,
                "parameter_update_norm_ratio":0.01,
                "nonfinite_events":0,
            } for u in LOCAL_CHECKPOINTS
        ],
        "passed":passed,
        "final_state":{"model_state_digest":"1"*64,"optimizer_state_digest":"2"*64,"rng_state_digest":"3"*64},
        "invalid_reason":None,
    }


def family_arm(family,passed_indices=()):
    return build_family_evidence(
        family=family,
        identity=identity(),
        reconstruction={"verified":True,"completed_step":2048,"model_state_digest":"4"*64,"optimizer_state_digest":"5"*64,"rng_state_digest":"6"*64,"nonfinite_events":0},
        worlds=[record(family,i,i in set(passed_indices)) for i in WORLD_INDICES],
    )


def test_materialized_family_worlds_are_exactly_indices_zero_through_seven() -> None:
    for family in FAMILIES:
        worlds=family_worlds(family)
        assert tuple(int(w.index) for w in worlds)==WORLD_INDICES
        assert len({w.content_id for w in worlds})==8


def test_final_evidence_reports_world_and_family_pass_counts() -> None:
    arms=[
        family_arm(FAMILIES[0],(0,1)),
        family_arm(FAMILIES[1],()),
        family_arm(FAMILIES[2],(7,)),
        family_arm(FAMILIES[3],()),
    ]
    final=build_final_evidence(arms)
    assert final["decision"]=="SOME_WORLDS_SINGLE_FIT"
    assert final["passed_world_count"]==3
    assert final["family_pass_counts"][FAMILIES[0]]==2
    assert final["family_pass_counts"][FAMILIES[2]]==1
    assert len(final["failed_worlds"])==29


def test_duplicate_world_record_is_rejected() -> None:
    arm=family_arm(FAMILIES[0])
    forged=deepcopy(arm)
    forged["worlds"][1]=deepcopy(forged["worlds"][0])
    from nolane_ai.experiments.exp325_runtime import _digest
    materialized=dict(forged); materialized.pop("family_evidence_digest")
    forged["family_evidence_digest"]=_digest(materialized)
    from nolane_ai.experiments.exp325_runtime import validate_family_evidence
    with pytest.raises(ValueError):
        validate_family_evidence(forged)
