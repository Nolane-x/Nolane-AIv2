from __future__ import annotations

import argparse, json
from pathlib import Path

from nolane_ai.experiments.exp323_contract import canonical_json_bytes
from nolane_ai.experiments.exp323r_repair import build_final_evidence


def load(path: str):
    p=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(p,dict): raise SystemExit("EXP-323R evidence must be object")
    return p


def main(argv=None)->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--hold",required=True); ap.add_argument("--decay",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(argv)
    payload=build_final_evidence(load(a.hold),load(a.decay))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as h: h.write(canonical_json_bytes(payload)+b"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
