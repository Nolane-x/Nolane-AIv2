from __future__ import annotations

import argparse,json
from dataclasses import fields
from pathlib import Path

from nolane_ai.experiments.exp324_contract import FAMILIES,canonical_json_bytes
from nolane_ai.experiments.exp324_identity import Exp324ExecutionIdentity,validate_execution_identity
from nolane_ai.experiments.exp324_runtime import run_family_arm


def main(argv=None)->int:
    p=argparse.ArgumentParser()
    p.add_argument("--reconstruction-checkpoint",required=True)
    p.add_argument("--reconstruction-receipt",required=True)
    p.add_argument("--selection-lock",required=True)
    p.add_argument("--parent-final",required=True)
    p.add_argument("--family",choices=FAMILIES,required=True)
    p.add_argument("--execution-identity",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    raw=json.loads(Path(a.execution_identity).read_text(encoding="utf-8"))
    if set(raw)!={f.name for f in fields(Exp324ExecutionIdentity)}: raise SystemExit("EXP-324 identity fields mismatch")
    identity=Exp324ExecutionIdentity(**raw); validate_execution_identity(identity)
    payload=run_family_arm(
        checkpoint_path=a.reconstruction_checkpoint,
        receipt_path=a.reconstruction_receipt,
        selection_lock_path=a.selection_lock,
        parent_final_path=a.parent_final,
        family=a.family,
        execution_identity=identity,
    )
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as h: h.write(canonical_json_bytes(payload)+b"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
