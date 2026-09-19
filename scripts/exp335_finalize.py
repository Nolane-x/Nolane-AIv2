from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp335_identity import Exp335ExecutionIdentity
from nolane_ai.experiments.exp335_runtime import build_final_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-identity", required=True)
    parser.add_argument("--parent-exp334", required=True)
    parser.add_argument("--chunk-dir", action="append", required=True)
    parser.add_argument("--chunk-artifact-digest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    identity = Exp335ExecutionIdentity(
        **json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    )
    payload = build_final_evidence(
        chunk_dirs=args.chunk_dir,
        chunk_artifact_digests=args.chunk_artifact_digest,
        execution_identity=identity,
        parent_exp334_path=args.parent_exp334,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
