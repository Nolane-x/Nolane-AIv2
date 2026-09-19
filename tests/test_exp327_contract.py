from __future__ import annotations
import pytest
from nolane_ai.experiments.exp327_contract import *

pytest.importorskip("torch")

from nolane_ai.experiments.exp327_runtime import run_court
from nolane_ai.experiments.exp327_identity import build_execution_identity

def snap(g,p,e):
 n=len(GROUP_MEMBERS[g]);v=1.0 if p else 0.0
 return SubsetSnapshot(g,e,n*e,(v,)*n,(v,)*n)
def records(fails=()):
 return {g:tuple(snap(g,g not in fails,e) for e in EXPOSURE_CHECKPOINTS) for g in GROUP_IDS}
def test_geometry_and_effort():
 assert len(PAIR_IDS)==6 and len(TRIPLE_IDS)==4 and QUARTET_ID=="Q0123"
 for g,m in GROUP_MEMBERS.items():
  s=training_schedule(g)
  for w in m:assert [e for wi,_,e in s if wi==w]==[1,2,4,8]*8
def test_prereg_digest():assert preregistration_digest()=="f76f738d70ea24b1dd7044070de005513dbda4f867a3803826b190903bf8df9d"
def test_reducer_cases():
 assert reduce_subset(records({"Q0123"}))[0]=="QUARTET_ONLY_FAILURE_CONFIRMED"
 assert reduce_subset(records({"T012","Q0123"}))[0]=="TRIPLE_MINIMAL_FAILURE_PRESENT"
 assert reduce_subset(records({"P02","T012","T023","Q0123"}))[0]=="PAIR_MINIMAL_FAILURE_PRESENT"
 assert reduce_subset(records())[0]=="PARENT_QUARTET_REPRODUCTION_MISMATCH"
 assert reduce_subset(records({"P02","Q0123"}))[0]=="NONMONOTONIC_SUBSET_FIT"
def test_implementation_surface_exists():
 assert callable(run_court) and callable(build_execution_identity)
