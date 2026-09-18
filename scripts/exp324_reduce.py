from __future__ import annotations

import argparse,json
from pathlib import Path

from nolane_ai.experiments.exp324_contract import FAMILIES,canonical_json_bytes
from nolane_ai.experiments.exp324_runtime import build_final_evidence


def main(argv=None)->int:
    p=argparse.ArgumentParser()
    for i,f in enumerate(FAMILIES): p.add_argument(f"--arm-{i}",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    paths=[getattr(a,f"arm_{i}") for i in range(len(FAMILIES))]
    arms=[]
    for path in paths:
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(obj,dict): raise SystemExit("EXP-324 arm evidence must be object")
        arms.append(obj)
    final=build_final_evidence(arms)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as h: h.write(canonical_json_bytes(final)+b"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
