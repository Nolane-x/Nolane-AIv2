from __future__ import annotations
from dataclasses import replace
import pytest
from nolane_ai.experiments.exp327_identity import *
def ident():return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)
def test_identity_binds_parent_and_geometry():
 i=ident();assert i.parent_run_id==35409935457;assert i.parent_final_artifact_id==10573847041;assert i.group_ids==("P01","P02","P03","P12","P13","P23","T012","T013","T023","T123","Q0123")
def test_parent_forgery_rejected_even_if_digest_recomputed():
 i=ident();x=replace(i,parent_run_id=1,exp327_execution_digest="");x=replace(x,exp327_execution_digest=execution_digest(x))
 with pytest.raises(ValueError):validate_execution_identity(x)
def test_geometry_forgery_rejected():
 i=ident();x=replace(i,group_ids=("P01",),exp327_execution_digest="");x=replace(x,exp327_execution_digest=execution_digest(x))
 with pytest.raises(ValueError):validate_execution_identity(x)
