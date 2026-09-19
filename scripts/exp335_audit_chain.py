from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp335_artifact_audit import audit_chunk_chain, audit_final_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently verify EXP-335 immutable chunk artifact ZIPs."
    )
    parser.add_argument(
        "--execution-identity",
        required=True,
        help="Path to exp335_execution_identity_v1.json",
    )
    parser.add_argument(
        "--chunk-zip",
        action="append",
        required=True,
        help="Chunk artifact ZIP in chain order. Repeat once per chunk.",
    )
    parser.add_argument(
        "--final-json",
        help="Optional EXP-335 final evidence JSON; requires a complete 8-chunk chain.",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    identity = json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise SystemExit("EXP-335 execution identity must be a JSON object")

    chain_report = audit_chunk_chain(args.chunk_zip, identity=identity)
    report: dict[str, object] = {"chain": chain_report}
    if args.final_json:
        report["final"] = audit_final_evidence(
            args.final_json,
            chain_report=chain_report,
            identity=identity,
        )
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
