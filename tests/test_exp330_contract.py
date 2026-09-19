import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp330_contract import (
    ARM_IDS,CONTROL_ARM,PROJECT_ARM,SHAM_ARM,ArmResult,
    effort_for_round,reduce_projection,training_schedule,
)

def _r(arm,flags=(False,True),*,model="a",optimizer="b",rng="c",negative=0,projected=0):
    token=tuple(1.0 if x else 0.5 for x in flags)
    exact=tuple(1.0 if x else 0.0 for x in flags)
    return ArmResult(
        arm_id=arm,total_optimizer_updates=64,
        world_token_accuracies=token,world_full_answer_exact=exact,
        model_state_digest=model*64,optimizer_state_digest=optimizer*64,rng_state_digest=rng*64,
        nonfinite_events=0,negative_dot_count=negative,projection_event_count=projected,
    )

def test_training_schedule_preserves_exact_alt_geometry():
    rows=training_schedule()
    assert len(rows)==64
    assert rows[:4]==((0,0,1),(2,0,1),(0,1,2),(2,1,2))
    for world in (0,2):
        wr=[x for x in rows if x[0]==world]
        assert [x[1] for x in wr]==list(range(32))
        assert [x[2] for x in wr[:8]]==[1,2,4,8,1,2,4,8]

def test_parent_reproduction_gate_is_first_scientific_gate():
    rs={
        CONTROL_ARM:_r(CONTROL_ARM,(True,True)),
        SHAM_ARM:_r(SHAM_ARM,(True,True),negative=5),
        PROJECT_ARM:_r(PROJECT_ARM,(True,True),negative=5,projected=5),
    }
    assert reduce_projection(rs)[0]=="PARENT_ALT_REPRODUCTION_MISMATCH"

def test_sham_must_be_bitwise_state_equivalent_to_control():
    rs={
        CONTROL_ARM:_r(CONTROL_ARM),
        SHAM_ARM:_r(SHAM_ARM,model="d",negative=5),
        PROJECT_ARM:_r(PROJECT_ARM,negative=5,projected=5),
    }
    assert reduce_projection(rs)[0]=="SHAM_TRAJECTORY_MISMATCH"

def test_projection_must_act_before_rescue_claim():
    control=_r(CONTROL_ARM)
    rs={
        CONTROL_ARM:control,
        SHAM_ARM:_r(SHAM_ARM,negative=5),
        PROJECT_ARM:_r(PROJECT_ARM,negative=0,projected=0),
    }
    assert reduce_projection(rs)[0]=="PROJECTION_NOT_TRIGGERED"

def test_projection_rescue_and_no_rescue_are_distinct():
    control=_r(CONTROL_ARM)
    common={CONTROL_ARM:control,SHAM_ARM:_r(SHAM_ARM,negative=5)}
    rescue={**common,PROJECT_ARM:_r(PROJECT_ARM,(True,True),negative=7,projected=7)}
    assert reduce_projection(rescue)[0]=="CONFLICT_PROJECTION_RESCUE"
    no_rescue={**common,PROJECT_ARM:_r(PROJECT_ARM,(False,True),negative=7,projected=7)}
    assert reduce_projection(no_rescue)[0]=="CONFLICT_PROJECTION_NO_RESCUE"

def test_preregistration_canonical_digest_is_locked():
    from nolane_ai.experiments.exp330_contract import APPROVED_PREREGISTRATION_DIGEST
    root=Path(__file__).resolve().parents[1]
    raw=(root/"protocols/v017/exp330_preregistration_v1.json").read_bytes()
    payload=json.loads(raw)
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==APPROVED_PREREGISTRATION_DIGEST
    checksum=(root/"protocols/v017/exp330_preregistration_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest()==checksum
