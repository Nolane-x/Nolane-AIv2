from __future__ import annotations
import argparse,json
from pathlib import Path
from nolane_ai.experiments.exp331_identity import Exp331ExecutionIdentity
from nolane_ai.experiments.exp331_runtime import run_court,validate_final_evidence

def main()->None:
    p=argparse.ArgumentParser()
    for name in ("reconstruction-checkpoint","reconstruction-receipt","selection-lock","exp330-parent","exp327-parent","execution-identity","output"):p.add_argument(f"--{name}",required=True)
    a=p.parse_args();identity=Exp331ExecutionIdentity(**json.loads(Path(a.execution_identity).read_text()))
    payload=run_court(checkpoint_path=a.reconstruction_checkpoint,receipt_path=a.reconstruction_receipt,selection_lock_path=a.selection_lock,exp330_parent_path=a.exp330_parent,exp327_parent_path=a.exp327_parent,execution_identity=identity)
    validate_final_evidence(payload);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:json.dump(payload,f,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False);f.write("\n")
if __name__=="__main__":main()
