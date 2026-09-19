from __future__ import annotations
import argparse,json
from dataclasses import fields
from pathlib import Path
from nolane_ai.experiments.exp327_contract import canonical_json_bytes
from nolane_ai.experiments.exp327_identity import Exp327ExecutionIdentity,validate_execution_identity
from nolane_ai.experiments.exp327_runtime import run_court
def main(argv=None):
 p=argparse.ArgumentParser()
 for x in ("reconstruction-checkpoint","reconstruction-receipt","selection-lock","parent-final","execution-identity","output"):p.add_argument("--"+x,required=True)
 a=p.parse_args(argv);raw=json.loads(Path(a.execution_identity).read_text())
 if set(raw)!={f.name for f in fields(Exp327ExecutionIdentity)}:raise SystemExit("EXP-327 identity field mismatch")
 i=Exp327ExecutionIdentity(**raw);validate_execution_identity(i)
 payload=run_court(checkpoint_path=a.reconstruction_checkpoint,receipt_path=a.reconstruction_receipt,selection_lock_path=a.selection_lock,parent_final_path=a.parent_final,execution_identity=i)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("xb") as h:h.write(canonical_json_bytes(payload)+b"\n")
 return 0
if __name__=="__main__":raise SystemExit(main())
