from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel


def main() -> int:
    config = NLMConfig.stage_a_pilot_16m()
    model = NolaneLivingModel(config, device="meta")
    audit = audit_model(model)
    payload = {
        "budget_version": config.budget.version,
        "device": "meta",
        "total_parameters": audit.total_parameters,
        "functional_parameters": audit.functional_parameters,
        "reserved_parameters": audit.reserved_parameters,
        "trainable_allocation": audit.trainable_parameters,
    }
    if payload["total_parameters"] != 16_000_000:
        raise SystemExit(f"pilot parameter drift: {payload['total_parameters']}")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
