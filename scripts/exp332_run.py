from __future__ import annotations
import argparse, json
from pathlib import Path

from nolane_ai.experiments.exp332_identity import Exp332ExecutionIdentity
from nolane_ai.experiments.exp332_runtime import run_court, validate_final_evidence

def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "reconstruction-checkpoint",
        "reconstruction-receipt",
        "selection-lock",
        "exp330-parent",
        "exp327-original",
        "exp327-replay",
        "execution-identity",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    identity = Exp332ExecutionIdentity(**json.loads(Path(args.execution_identity).read_text(encoding="utf-8")))
    payload = run_court(
        checkpoint_path=args.reconstruction_checkpoint,
        receipt_path=args.reconstruction_receipt,
        selection_lock_path=args.selection_lock,
        exp330_parent_path=args.exp330_parent,
        exp327_original_path=args.exp327_original,
        exp327_replay_path=args.exp327_replay,
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
