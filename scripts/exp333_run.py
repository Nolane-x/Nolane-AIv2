from __future__ import annotations
import argparse, json
from pathlib import Path

from nolane_ai.experiments.exp333_identity import Exp333ExecutionIdentity
from nolane_ai.experiments.exp333_runtime import run_court, validate_final_evidence

def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "reconstruction-checkpoint",
        "reconstruction-receipt",
        "selection-lock",
        "parent-exp332",
        "execution-identity",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    identity = Exp333ExecutionIdentity(
        **json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    )
    payload = run_court(
        checkpoint_path=args.reconstruction_checkpoint,
        receipt_path=args.reconstruction_receipt,
        selection_lock_path=args.selection_lock,
        parent_exp332_path=args.parent_exp332,
        execution_identity=identity,
    )
    validate_final_evidence(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")

if __name__ == "__main__":
    main()
