import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp329_contract import (
    PROBE_ROUNDS,ProbeRecord,directed_damage,effort_for_round,reduce_cross_update,
)

def _p(round_index,w0=False,w2=False):
    return ProbeRecord(
        round_index=round_index,
        effort=effort_for_round(round_index),
        world0_loss_pre=1.0,world2_loss_pre=1.0,
        gradient_dot=-1.0,gradient_cosine=-0.5,
        world0_gradient_norm=2.0,world2_gradient_norm=1.0,
        world0_update_self_delta=-0.1,
        world0_update_cross_delta_on_world2=0.1 if w0 else 0.0,
        world2_update_self_delta=-0.1,
        world2_update_cross_delta_on_world0=0.1 if w2 else 0.0,
        world0_direct_damage=w0,world2_direct_damage=w2,nonfinite_events=0,
    )

def test_effort_geometry_is_world_local_cycle():
    assert [effort_for_round(i) for i in range(8)]==[1,2,4,8,1,2,4,8]

def test_directed_damage_requires_self_improvement_and_cross_harm():
    assert directed_damage(self_delta=-1e-3,cross_delta=1e-3)
    assert not directed_damage(self_delta=0.0,cross_delta=1e-3)
    assert not directed_damage(self_delta=-1e-3,cross_delta=0.0)

def test_parent_reproduction_gate_precedes_local_claim():
    rows=[_p(i,w2=True) for i in PROBE_ROUNDS]
    assert reduce_cross_update(rows,final_token_accuracies=(1.0,1.0),final_full_answer_exact=(1.0,1.0))[0]=="PARENT_ALT_REPRODUCTION_MISMATCH"

def test_reducer_localizes_directed_damage():
    rows=[_p(i,w0=(i==8),w2=(i==16)) for i in PROBE_ROUNDS]
    decision,w0,w2=reduce_cross_update(rows,final_token_accuracies=(0.5,1.0),final_full_answer_exact=(0.0,1.0))
    assert decision=="BIDIRECTIONAL_LOCAL_CROSS_DAMAGE"
    assert w0==(8,) and w2==(16,)

def test_reducer_can_find_world2_only_damage():
    rows=[_p(i,w2=(i in (8,16))) for i in PROBE_ROUNDS]
    decision,w0,w2=reduce_cross_update(rows,final_token_accuracies=(0.5,1.0),final_full_answer_exact=(0.0,1.0))
    assert decision=="WORLD2_ONLY_LOCAL_CROSS_DAMAGE"
    assert w0==() and w2==(8,16)

def test_reducer_fail_closes_incomplete_probe_lattice():
    rows=[_p(i) for i in PROBE_ROUNDS[:-1]]
    assert reduce_cross_update(rows,final_token_accuracies=(0.5,1.0),final_full_answer_exact=(0.0,1.0))[0]=="INVALID_CROSS_UPDATE_COURT"


def test_preregistration_canonical_and_exact_file_digests_are_locked():
    from nolane_ai.experiments.exp329_contract import APPROVED_PREREGISTRATION_DIGEST
    root=Path(__file__).resolve().parents[1]
    path=root/"protocols/v017/exp329_preregistration_v1.json"
    raw=path.read_bytes()
    payload=json.loads(raw)
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==APPROVED_PREREGISTRATION_DIGEST
    checksum=(root/"protocols/v017/exp329_preregistration_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest()==checksum
