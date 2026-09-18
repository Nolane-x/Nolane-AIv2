from __future__ import annotations

from copy import deepcopy
import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp324_contract import AUTHORIZATION_FLAGS,FAMILIES,FamilySnapshot,LOCAL_CHECKPOINTS
from nolane_ai.experiments.exp324_identity import build_execution_identity
from nolane_ai.experiments.exp324_runtime import _build_arm_evidence,build_final_evidence


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def arm(family,passing=False):
    rows=tuple(FamilySnapshot(
        family=family,local_update=u,cumulative_step=2048+u,
        teacher_forced_answer_token_accuracy=(0.995 if passing and u==256 else 0.95),
        teacher_forced_full_answer_exact=(1.0 if passing and u==256 else 0.75),
        greedy_exact=0.75,answer_only_loss=0.4,gradient_norm_preclip=0.001,
        parameter_update_norm_ratio=1e-6,nonfinite_events=0,
    ) for u in LOCAL_CHECKPOINTS)
    return _build_arm_evidence(
        family=family,identity=identity(),
        reconstruction={"verified":True,"completed_step":2048,"model_state_digest":"1"*64,"optimizer_state_digest":"2"*64,"rng_state_digest":"3"*64,"nonfinite_events":0},
        snapshots=rows,all32_summaries={str(u):{} for u in LOCAL_CHECKPOINTS},
        final_model_digest="4"*64,final_optimizer_digest="5"*64,final_rng_digest="6"*64,invalid_reason=None,
    )


def test_final_evidence_uses_only_family_primary_for_decision() -> None:
    arms=[arm(f,passing=True) for f in FAMILIES]
    final=build_final_evidence(arms)
    assert final["decision"]=="ALL_FAMILIES_ISOLATED_FIT"
    assert final["passed_families"]==list(FAMILIES)
    assert not any(final[k] for k in AUTHORIZATION_FLAGS)


def test_duplicate_family_fails_closed() -> None:
    arms=[arm(FAMILIES[0]) for _ in FAMILIES]
    with pytest.raises(ValueError,match="duplicate"): build_final_evidence(arms)
