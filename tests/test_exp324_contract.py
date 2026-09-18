from __future__ import annotations

from dataclasses import replace

from nolane_ai.experiments.exp324_contract import (
    AUTHORIZATION_FLAGS,
    CUMULATIVE_CHECKPOINTS,
    FAMILIES,
    LOCAL_CHECKPOINTS,
    FamilySnapshot,
    preregistration_digest,
    preregistration_payload,
    reduce_family_isolation,
)


def row(family: str, local: int, *, token: float=0.95, full: float=0.75):
    return FamilySnapshot(
        family=family,
        local_update=local,
        cumulative_step=2048+local,
        teacher_forced_answer_token_accuracy=token,
        teacher_forced_full_answer_exact=full,
        greedy_exact=full,
        answer_only_loss=0.4,
        gradient_norm_preclip=0.001,
        parameter_update_norm_ratio=1e-6,
        nonfinite_events=0,
    )


def grid(passing=()):
    passing=set(passing)
    return {
        family: tuple(
            row(
                family, local,
                token=(0.995 if family in passing and local==256 else 0.95),
                full=(1.0 if family in passing and local==256 else 0.75),
            )
            for local in LOCAL_CHECKPOINTS
        )
        for family in FAMILIES
    }


def test_preregistration_binds_recovered_exp323r_and_exact_reconstruction() -> None:
    p=preregistration_payload()
    a=p["authority"]
    assert a["parent_exp323r_recovery_run_id"]==35356470634
    assert a["parent_exp323r_final_artifact_id"]==10552276255
    assert a["parent_exp323r_final_evidence_digest"]=="a92d56a45151411a8fa4e0799949d72aed701f52ad9adae5fdb90983be09f051"
    assert a["reconstruction_artifact_id"]==10547681681
    assert a["reconstruction_checkpoint_sha256"]=="4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
    assert preregistration_digest()=="ae2e572b02284ab4c9b49f2a98ffeef3de70e0927c45960e1571201427fc7599"


def test_family_update_budget_matches_historical_per_family_exposure() -> None:
    p=preregistration_payload()
    assert p["training"]["isolated_optimizer_updates_per_family"]==256
    assert p["historical_joint_control"]["implied_updates_per_family"]==256
    assert tuple(p["training"]["local_checkpoints"])==LOCAL_CHECKPOINTS
    assert tuple(p["training"]["cumulative_checkpoints"])==CUMULATIVE_CHECKPOINTS


def test_authorization_remains_false() -> None:
    assert AUTHORIZATION_FLAGS and not any(AUTHORIZATION_FLAGS.values())


def test_reducer_all_some_none_and_invalid() -> None:
    assert reduce_family_isolation(grid(FAMILIES))=="ALL_FAMILIES_ISOLATED_FIT"
    assert reduce_family_isolation(grid((FAMILIES[0],)))=="SOME_FAMILIES_ISOLATED_FIT"
    assert reduce_family_isolation(grid())=="NO_FAMILIES_ISOLATED_FIT"
    broken=grid(); broken.pop(FAMILIES[0])
    assert reduce_family_isolation(broken)=="INVALID_FAMILY_ISOLATION"


def test_split_floor_across_checkpoints_does_not_pass() -> None:
    family=FAMILIES[0]
    rows=(
        row(family,64,token=0.995,full=0.875),
        row(family,128,token=0.98,full=1.0),
        row(family,256,token=0.98,full=0.875),
    )
    g=grid(); g[family]=rows
    assert reduce_family_isolation(g)=="NO_FAMILIES_ISOLATED_FIT"


def test_nonfinite_or_wrong_geometry_fails_closed() -> None:
    g=grid()
    rows=list(g[FAMILIES[1]])
    rows[-1]=replace(rows[-1],nonfinite_events=1)
    g[FAMILIES[1]]=tuple(rows)
    assert reduce_family_isolation(g)=="INVALID_FAMILY_ISOLATION"
