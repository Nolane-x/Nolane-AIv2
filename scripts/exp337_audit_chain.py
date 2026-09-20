from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_ai.experiments.exp337_artifact_audit import audit_chunk_chain, audit_final_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently verify EXP-337 immutable chunk artifact ZIPs."
    )
    parser.add_argument("--execution-identity", required=True)
    parser.add_argument("--chunk-zip", action="append", required=True)
    parser.add_argument(
        "--final-json",
        help="Optional EXP-337 final evidence JSON; requires a complete 8-chunk chain.",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    identity = json.loads(Path(args.execution_identity).read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise SystemExit("EXP-337 execution identity must be a JSON object")

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
