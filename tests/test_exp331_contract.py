import hashlib,json
from pathlib import Path
from nolane_ai.experiments.exp331_contract import PAIR_IDS,PAIR_MEMBERS,PairArmResult,effort_for_round,reduce_pair_lattice,training_schedule

def _arm(pair,mode,passed=True,model="a",optimizer="b",rng="c",neg=0,proj=0):
    vals=(1.0,1.0) if passed else (0.5,1.0); exact=(1.0,1.0) if passed else (0.0,1.0)
    return PairArmResult(pair,mode,64,vals,exact,model*64,optimizer*64,rng*64,0,neg,proj)

def _lattice(p02_project=True,regress=()):
    out={}
    for p in PAIR_IDS:
        control_pass=p!="P02"
        out[f"{p}:CONTROL"]=_arm(p,"CONTROL",control_pass)
        out[f"{p}:SHAM"]=_arm(p,"SHAM",control_pass,neg=4)
        project_pass=(p02_project if p=="P02" else p not in regress)
        out[f"{p}:PROJECT"]=_arm(p,"PROJECT",project_pass,model="d",optimizer="e",neg=5,proj=5 if p=="P02" else 0)
    return out

def test_pair_schedule_matches_exp327_geometry():
    for pair in PAIR_IDS:
        rows=training_schedule(pair);assert len(rows)==64;assert tuple(x[0] for x in rows[:2])==PAIR_MEMBERS[pair]
        assert [effort_for_round(i) for i in range(8)]==[1,2,4,8,1,2,4,8]

def test_strong_positive_requires_p02_rescue_and_no_regression():
    decision,_,reg=reduce_pair_lattice(_lattice(),parent_reproduced=True)
    assert decision=="PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION" and reg==()

def test_regression_is_separate_disposition():
    decision,_,reg=reduce_pair_lattice(_lattice(regress=("P13",)),parent_reproduced=True)
    assert decision=="P02_RESCUE_WITH_PAIR_REGRESSION" and reg==("P13",)

def test_p02_no_rescue_is_preserved():
    assert reduce_pair_lattice(_lattice(p02_project=False),parent_reproduced=True)[0]=="P02_PROJECTION_NO_RESCUE"

def test_parent_and_sham_fail_closed():
    assert reduce_pair_lattice(_lattice(),parent_reproduced=False)[0]=="PARENT_PAIR_LATTICE_REPRODUCTION_MISMATCH"
    rows=_lattice();rows["P01:SHAM"]=_arm("P01","SHAM",True,model="z",neg=2)
    assert reduce_pair_lattice(rows,parent_reproduced=True)[0]=="SHAM_PAIR_LATTICE_MISMATCH"

def test_p02_projection_must_trigger():
    rows=_lattice();rows["P02:PROJECT"]=_arm("P02","PROJECT",True,model="d",optimizer="e",neg=0,proj=0)
    assert reduce_pair_lattice(rows,parent_reproduced=True)[0]=="P02_PROJECTION_NOT_TRIGGERED"

def test_incomplete_lattice_is_invalid():
    rows=_lattice();rows.pop("P23:PROJECT")
    assert reduce_pair_lattice(rows,parent_reproduced=True)[0]=="INVALID_PAIR_LATTICE_PROJECTION_COURT"

def test_preregistration_digest_is_locked():
    from nolane_ai.experiments.exp331_contract import APPROVED_PREREGISTRATION_DIGEST
    root=Path(__file__).resolve().parents[1];path=root/"protocols/v017/exp331_preregistration_v1.json";raw=path.read_bytes();p=json.loads(raw)
    canonical=json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==APPROVED_PREREGISTRATION_DIGEST
    assert hashlib.sha256(raw).hexdigest()==(root/"protocols/v017/exp331_preregistration_v1.sha256").read_text().split()[0]
