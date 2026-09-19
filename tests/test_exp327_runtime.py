from __future__ import annotations
from nolane_ai.experiments.exp327_contract import *
def snap(g,p,e):
 n=len(GROUP_MEMBERS[g]);v=1.0 if p else 0.0;return SubsetSnapshot(g,e,n*e,(v,)*n,(v,)*n)
def records(fails):
 return {g:tuple(snap(g,g not in fails,e) for e in EXPOSURE_CHECKPOINTS) for g in GROUP_IDS}
def test_nonmonotonic_is_detected_across_cross_pair():
 d,_,v=reduce_subset(records({"P03","Q0123"}));assert d=="NONMONOTONIC_SUBSET_FIT";assert ("T013","P03") in v or ("T023","P03") in v
def test_invalid_missing_group_fails_closed():
 r=records({"Q0123"});r.pop("P02");assert reduce_subset(r)[0]=="INVALID_SUBSET_COURT"
def test_pair_failure_wins_after_monotonicity():
 fails={"P02","T012","T023","Q0123"};assert reduce_subset(records(fails))[0]=="PAIR_MINIMAL_FAILURE_PRESENT"
