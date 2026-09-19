from __future__ import annotations
import argparse, json
from pathlib import Path
from nolane_ai.experiments.exp334_identity import Exp334ExecutionIdentity
from nolane_ai.experiments.exp334_runtime import run_court

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reconstruction-checkpoint", required=True)
    p.add_argument("--reconstruction-receipt", required=True)
    p.add_argument("--selection-lock", required=True)
    p.add_argument("--parent-exp333", required=True)
    p.add_argument("--parent-exp327", required=True)
    p.add_argument("--parent-exp326-family", required=True)
    p.add_argument("--execution-identity", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    identity = Exp334ExecutionIdentity(**json.loads(Path(a.execution_identity).read_text()))
    payload = run_court(
        checkpoint_path=a.reconstruction_checkpoint,
        receipt_path=a.reconstruction_receipt,
        selection_lock_path=a.selection_lock,
        parent_exp333_path=a.parent_exp333,
        parent_exp327_path=a.parent_exp327,
        parent_exp326_family_path=a.parent_exp326_family,
        execution_identity=identity,
    )
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
