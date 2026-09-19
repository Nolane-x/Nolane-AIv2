from nolane_ai.experiments.exp328_contract import (
    ARM_IDS,EXPOSURES_PER_WORLD,ArmResult,effort_for_exposure,
    training_schedule,reduce_order_geometry,
)

def _r(arm,passed):
    x=(1.0,1.0) if passed else (0.98,1.0)
    return ArmResult(arm,64,x,(1.0,1.0),0)

def test_all_schedules_hold_exact_per_world_exposure_and_effort_budget():
    for arm in ARM_IDS:
        s=training_schedule(arm)
        assert len(s)==64
        for world in (0,2):
            rows=[x for x in s if x[0]==world]
            assert len(rows)==EXPOSURES_PER_WORLD
            assert [x[1] for x in rows]==list(range(EXPOSURES_PER_WORLD))
            assert {e:[x[2] for x in rows].count(e) for e in (1,2,4,8)}=={1:8,2:8,4:8,8:8}

def test_schedule_shapes_are_exact():
    a=training_schedule("ALT_0_2")
    assert a[:4]==((0,0,1),(2,0,1),(0,1,2),(2,1,2))
    b=training_schedule("ALT_2_0")
    assert b[:4]==((2,0,1),(0,0,1),(2,1,2),(0,1,2))
    c=training_schedule("BLOCK_0_2")
    assert all(w==0 for w,_,_ in c[:32]) and all(w==2 for w,_,_ in c[32:])
    d=training_schedule("BLOCK_2_0")
    assert all(w==2 for w,_,_ in d[:32]) and all(w==0 for w,_,_ in d[32:])

def test_reducer_fail_closes_and_requires_parent_reproduction_failure():
    missing={a:_r(a,False) for a in ARM_IDS[:-1]}
    assert reduce_order_geometry(missing)[0]=="INVALID_ORDER_GEOMETRY_COURT"
    all_pass={a:_r(a,True) for a in ARM_IDS}
    assert reduce_order_geometry(all_pass)[0]=="PARENT_P02_REPRODUCTION_MISMATCH"

def test_reducer_localizes_order_sensitive_rescue():
    rs={a:_r(a,False) for a in ARM_IDS}
    rs["ALT_2_0"]=_r("ALT_2_0",True)
    decision,passed,rescued=reduce_order_geometry(rs)
    assert decision=="ORDER_GEOMETRY_SENSITIVE_INTERFERENCE"
    assert passed["ALT_0_2"] is False
    assert rescued==("ALT_2_0",)

def test_reducer_can_rule_out_registered_simple_reorderings():
    rs={a:_r(a,False) for a in ARM_IDS}
    decision,passed,rescued=reduce_order_geometry(rs)
    assert decision=="ORDER_GEOMETRY_INVARIANT_PAIR_FAILURE"
    assert not any(passed.values())
    assert rescued==()
