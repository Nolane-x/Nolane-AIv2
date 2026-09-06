from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.experiments.runner import execute_stage_a


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EV-E2 Stage-A executable smoke harness")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--full-open", action="store_true", help="use the frozen 32 open replicate indices")
    parser.add_argument("--root-seed", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    replicates = 32 if args.full_open else args.replicates
    packet = execute_stage_a(
        protocol_path=ROOT / "protocols" / "stage_a_v1.json",
        digest_path=ROOT / "protocols" / "stage_a_v1.sha256",
        code_root=ROOT,
        replicates=replicates,
        bootstrap_samples=args.bootstrap_samples,
        root_seed=args.root_seed,
    )
    rendered = json.dumps(packet, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
