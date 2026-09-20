import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp336_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PREREGISTRATION_JSON_SHA256,
    canonical_json_bytes,
)


def test_preregistration_bytes_and_canonical_digest_are_frozen():
    path = Path("protocols/v017/exp336_preregistration_v1.json")
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == PREREGISTRATION_JSON_SHA256
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == APPROVED_PREREGISTRATION_DIGEST
    sidecar = Path("protocols/v017/exp336_preregistration_v1.sha256").read_text().split()[0]
    assert sidecar == APPROVED_PREREGISTRATION_DIGEST


def test_preregistration_keeps_downstream_authority_closed():
    p = json.loads(Path("protocols/v017/exp336_preregistration_v1.json").read_text())
    assert p["authorizations"]["exp336_implementation_authorized"] is False
    assert p["authorizations"]["stage_b_implementation_authorized"] is False
    assert p["authorizations"]["stage_c_implementation_authorized"] is False
    assert p["authorizations"]["scale_authorized"] is False
