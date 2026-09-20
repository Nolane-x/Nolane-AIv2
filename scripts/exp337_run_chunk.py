from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp337_identity import Exp337ExecutionIdentity
from nolane_ai.experiments.exp337_runtime import run_chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--execution-identity", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--parent-artifact-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-receipt")
    parser.add_argument("--previous-dir")
    args = parser.parse_args()

    identity = Exp337ExecutionIdentity(
        **json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    )
    run_chunk(
        chunk_index=args.chunk_index,
        execution_identity=identity,
        code_root=args.code_root,
        parent_artifact_digest=args.parent_artifact_digest,
        output_dir=args.output_dir,
        parent_checkpoint=args.parent_checkpoint,
        parent_receipt=args.parent_receipt,
        previous_dir=args.previous_dir,
    )


if __name__ == "__main__":
    main()
