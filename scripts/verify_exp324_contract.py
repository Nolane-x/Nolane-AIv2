from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp324_contract import (
    canonical_json_bytes, preregistration_digest, preregistration_payload,
)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--json",required=True)
    p.add_argument("--sha256",required=True)
    return p


def main(argv=None)->int:
    a=parser().parse_args(argv)
    payload=json.loads(Path(a.json).read_text(encoding="utf-8"))
    if payload!=preregistration_payload():
        raise SystemExit("EXP-324 preregistration payload mismatch")
    digest=hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest!=preregistration_digest():
        raise SystemExit("EXP-324 implementation digest mismatch")
    sidecar=Path(a.sha256).read_text(encoding="ascii").strip().split()[0]
    if sidecar!=digest:
        raise SystemExit("EXP-324 sidecar digest mismatch")
    return 0


if __name__=="__main__": raise SystemExit(main())
