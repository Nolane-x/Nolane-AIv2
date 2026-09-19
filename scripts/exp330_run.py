from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp330_identity import Exp330ExecutionIdentity
from nolane_ai.experiments.exp330_runtime import run_court, validate_final_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-checkpoint", required=True)
    parser.add_argument("--reconstruction-receipt", required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--parent-final", required=True)
    parser.add_argument("--execution-identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    identity = Exp330ExecutionIdentity(
        **json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    )
    payload = run_court(
        checkpoint_path=args.reconstruction_checkpoint,
        receipt_path=args.reconstruction_receipt,
        selection_lock_path=args.selection_lock,
        parent_final_path=args.parent_final,
        execution_identity=identity,
    )
    validate_final_evidence(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
