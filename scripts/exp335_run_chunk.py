from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp335_identity import Exp335ExecutionIdentity
from nolane_ai.experiments.exp335_runtime import run_chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--execution-identity", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--parent-artifact-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reconstruction-checkpoint")
    parser.add_argument("--reconstruction-receipt")
    parser.add_argument("--selection-lock")
    parser.add_argument("--parent-exp334")
    parser.add_argument("--previous-dir")
    args = parser.parse_args()

    identity = Exp335ExecutionIdentity(
        **json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    )
    run_chunk(
        chunk_index=args.chunk_index,
        execution_identity=identity,
        code_root=args.code_root,
        parent_artifact_digest=args.parent_artifact_digest,
        output_dir=args.output_dir,
        reconstruction_checkpoint=args.reconstruction_checkpoint,
        reconstruction_receipt=args.reconstruction_receipt,
        selection_lock=args.selection_lock,
        parent_exp334=args.parent_exp334,
        previous_dir=args.previous_dir,
    )


if __name__ == "__main__":
    main()
