from __future__ import annotations
import argparse,hashlib,json,subprocess
from dataclasses import fields
from pathlib import Path
from nolane_ai.experiments.exp327_freeze import BASE_SHA,WORKFLOW_PATH,marker_json_bytes,marker_sidecar_bytes,validate_marker_paths,validate_source_paths
from nolane_ai.experiments.exp327_identity import Exp327ExecutionIdentity,source_tree_digest_from_git_tree_sha,validate_execution_identity
def git(root,*a):return subprocess.check_output(["git","-C",str(root),*a],text=True).strip()
def gb(root,*a):return subprocess.check_output(["git","-C",str(root),*a])
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--repo-root",required=True);p.add_argument("--source-commit-sha",required=True);p.add_argument("--marker-json",required=True);p.add_argument("--marker-sha256",required=True);a=p.parse_args(argv);root=Path(a.repo_root).resolve()
 raw=json.loads(Path(a.marker_json).read_text())
 if set(raw)!={f.name for f in fields(Exp327ExecutionIdentity)}:raise SystemExit("EXP-327 marker fields mismatch")
 i=Exp327ExecutionIdentity(**raw);validate_execution_identity(i)
 if i.source_commit_sha!=a.source_commit_sha:raise SystemExit("EXP-327 source mismatch")
 if Path(a.marker_json).read_bytes()!=marker_json_bytes(i):raise SystemExit("EXP-327 marker noncanonical")
 if Path(a.marker_sha256).read_bytes()!=marker_sidecar_bytes(i):raise SystemExit("EXP-327 sidecar mismatch")
 tree=git(root,"rev-parse",f"{a.source_commit_sha}^{{tree}}")
 if i.source_tree_digest!=source_tree_digest_from_git_tree_sha(tree):raise SystemExit("EXP-327 tree mismatch")
 if hashlib.sha256(gb(root,"show",f"{a.source_commit_sha}:{WORKFLOW_PATH}")).hexdigest()!=i.workflow_sha256:raise SystemExit("EXP-327 workflow mismatch")
 if git(root,"merge-base",BASE_SHA,a.source_commit_sha)!=BASE_SHA:raise SystemExit("EXP-327 lineage mismatch")
 validate_source_paths(x for x in git(root,"diff","--name-only",BASE_SHA,a.source_commit_sha).splitlines() if x)
 if git(root,"rev-parse","HEAD^")!=a.source_commit_sha:raise SystemExit("EXP-327 marker parent mismatch")
 validate_marker_paths(x for x in git(root,"diff-tree","--no-commit-id","--name-only","-r","HEAD").splitlines() if x)
 return 0
if __name__=="__main__":raise SystemExit(main())
