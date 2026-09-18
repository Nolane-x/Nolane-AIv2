from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

from nolane_ai.experiments.exp325_contract import canonical_json_bytes, preregistration_digest, preregistration_payload


def main(argv=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--json",required=True); p.add_argument("--sha256",required=True); a=p.parse_args(argv)
    payload=json.loads(Path(a.json).read_text(encoding="utf-8"))
    if payload!=preregistration_payload(): raise SystemExit("EXP-325 preregistration payload mismatch")
    digest=hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest!=preregistration_digest(): raise SystemExit("EXP-325 implementation digest mismatch")
    if Path(a.sha256).read_text(encoding="ascii").strip().split()[0]!=digest: raise SystemExit("EXP-325 sidecar digest mismatch")
    return 0


if __name__=="__main__": raise SystemExit(main())
