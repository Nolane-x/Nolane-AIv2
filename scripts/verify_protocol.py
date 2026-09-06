from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.protocol.schema import load_and_validate_protocol
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def main() -> int:
    spec = load_and_validate_protocol(PROTOCOL)
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    expected = DIGEST.read_text(encoding="utf-8").strip() if DIGEST.exists() else ""
    if digest != expected:
        raise SystemExit(f"protocol digest mismatch: expected {expected!r}, got {digest}")
    print(f"{spec.protocol_id}: {spec.status}; sha256={digest}; experiments={len(spec.experiments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
