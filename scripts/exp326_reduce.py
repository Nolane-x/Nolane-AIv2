from __future__ import annotations

import argparse,json
from pathlib import Path

from nolane_ai.experiments.exp326_contract import canonical_json_bytes
from nolane_ai.experiments.exp326_runtime import build_final_evidence


def load(path:str):
    p=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(p,dict): raise SystemExit("EXP-326 evidence must be object")
    return p


def main(argv=None)->int:
    p=argparse.ArgumentParser()
    for i in range(4): p.add_argument(f"--arm-{i}",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    payload=build_final_evidence([load(getattr(a,f"arm_{i}")) for i in range(4)])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as h: h.write(canonical_json_bytes(payload)+b"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
