from nolane_ai.experiments.exp334_contract import (
    GROUP_IDS, GROUP_MEMBERS, MODES, PARENT_CONTROL_PASS,
    training_schedule, reduce_higher_order, GroupArmResult,
)

def _arm(group, mode, passed=True, projections=0):
    n=len(GROUP_MEMBERS[group])
    vals=(1.0,)*n if passed else (0.5,)+((1.0,)*(n-1))
    exact=(1.0,)*n if passed else (0.0,)+((1.0,)*(n-1))
    return GroupArmResult(group,mode,n*32,vals,exact,"a"*64,"b"*64,"c"*64,0,projections,projections,projections)

def test_geometry_is_frozen():
    assert len(GROUP_IDS)==7
    assert MODES==("CONTROL","SHAM","SUBSPACE_PROJECT")
    assert PARENT_CONTROL_PASS=={
        "T012":False,"T013":True,"T023":False,"T123":True,
        "Q0123":False,"Q4567":True,"O01234567":False,
    }
    for group in GROUP_IDS:
        schedule=training_schedule(group)
        assert len(schedule)==len(GROUP_MEMBERS[group])*32

def test_reducer_strongest_positive():
    results={}
    for group in GROUP_IDS:
        control_pass=PARENT_CONTROL_PASS[group]
        control=_arm(group,"CONTROL",control_pass)
        sham=GroupArmResult(
            group,"SHAM",control.total_optimizer_updates,control.world_token_accuracies,
            control.world_full_answer_exact,control.model_state_digest,control.optimizer_state_digest,
            control.rng_state_digest,0,4,0,0
        )
        project=_arm(group,"SUBSPACE_PROJECT",True,3 if not control_pass else 1)
        results[f"{group}:CONTROL"]=control
        results[f"{group}:SHAM"]=sham
        results[f"{group}:SUBSPACE_PROJECT"]=project
    decision, passed, baseline, rescued, unresolved, regressions, no_trigger = reduce_higher_order(
        results,parent_reproduced=True
    )
    assert decision=="HIGHER_ORDER_SUBSPACE_RESCUE_NO_REGRESSION"
    assert baseline==("T012","T023","Q0123","O01234567")
    assert rescued==baseline
    assert unresolved==regressions==no_trigger==()

def test_fail_closed_on_sham_state_drift():
    results={}
    for group in GROUP_IDS:
        cp=PARENT_CONTROL_PASS[group]
        control=_arm(group,"CONTROL",cp)
        sham=GroupArmResult(group,"SHAM",control.total_optimizer_updates,control.world_token_accuracies,
            control.world_full_answer_exact,"d"*64,control.optimizer_state_digest,control.rng_state_digest,0,2,0,0)
        project=_arm(group,"SUBSPACE_PROJECT",True,1)
        results[f"{group}:CONTROL"]=control;results[f"{group}:SHAM"]=sham;results[f"{group}:SUBSPACE_PROJECT"]=project
    assert reduce_higher_order(results,parent_reproduced=True)[0]=="SHAM_HIGHER_ORDER_MISMATCH"
