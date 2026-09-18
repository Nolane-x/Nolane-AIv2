from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from nolane_ai.experiments.exp323_contract import canonical_json_bytes
from nolane_ai.experiments.exp323r_identity import Exp323RExecutionIdentity, validate_execution_identity
from nolane_ai.experiments.exp323r_repair import run_locked_intervention_arm


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser()
    p.add_argument("--reconstruction-checkpoint", required=True)
    p.add_argument("--reconstruction-receipt", required=True)
    p.add_argument("--selection-lock", required=True)
    p.add_argument("--parent-exp322", required=True)
    p.add_argument("--arm", choices=("HOLD_5E5","DECAY_2P5E5"), required=True)
    p.add_argument("--execution-identity", required=True)
    p.add_argument("--output", required=True)
    return p


def main(argv=None) -> int:
    a=parser().parse_args(argv)
    raw=json.loads(Path(a.execution_identity).read_text(encoding="utf-8"))
    expected={f.name for f in fields(Exp323RExecutionIdentity)}
    if set(raw)!=expected:
        raise SystemExit("EXP-323R execution identity field mismatch")
    identity=Exp323RExecutionIdentity(**raw)
    validate_execution_identity(identity)
    payload=run_locked_intervention_arm(
        checkpoint_path=a.reconstruction_checkpoint,
        receipt_path=a.reconstruction_receipt,
        selection_lock_path=a.selection_lock,
        parent_exp322_path=a.parent_exp322,
        arm=a.arm,
        execution_identity=identity,
    )
    target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as h:
        h.write(canonical_json_bytes(payload)+b"\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
