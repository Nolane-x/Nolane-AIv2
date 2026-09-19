from __future__ import annotations
import argparse,json
from pathlib import Path

from nolane_ai.experiments.exp328_identity import Exp328ExecutionIdentity
from nolane_ai.experiments.exp328_runtime import run_court,validate_final_evidence

def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--reconstruction-checkpoint",required=True)
    p.add_argument("--reconstruction-receipt",required=True)
    p.add_argument("--selection-lock",required=True)
    p.add_argument("--parent-final",required=True)
    p.add_argument("--execution-identity",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    identity=Exp328ExecutionIdentity(**json.loads(Path(a.execution_identity).read_text()))
    payload=run_court(
        checkpoint_path=a.reconstruction_checkpoint,
        receipt_path=a.reconstruction_receipt,
        selection_lock_path=a.selection_lock,
        parent_final_path=a.parent_final,
        execution_identity=identity,
    )
    validate_final_evidence(payload)
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:
        json.dump(payload,f,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        f.write("\n")

if __name__=="__main__":
    main()
