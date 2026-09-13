from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class Exp290Geometry:
    canonical_indices: tuple[int, ...] = (0, 1, 2, 3)
    train_replicates: int = 64
    eval_replicates: int = 32
    eval_start_replicate: int = 10000
    batch_size: int = 8
    variables: int = 8
    restarts: int = 4
    max_search_steps: int = 24
    timesteps: int = 4
    d_model: int = 64
    hidden_size: int = 48
    target_parameters: int = 500000
    noise_std: float = 0.05
    lr: float = 0.002
    weight_decay: float = 0.0
    root_prefix: str = "20260913-exp290-learned-clause-transfer-v1-dev"
    authority_scope: str = "DEVELOPMENT_EV_E2_ONLY"


EXP290_GEOMETRY = Exp290Geometry()


def exp290_geometry_manifest() -> dict[str, object]:
    payload = asdict(EXP290_GEOMETRY)
    payload["canonical_indices"] = list(EXP290_GEOMETRY.canonical_indices)
    return payload


def exp290_geometry_digest() -> str:
    raw = (
        json.dumps(
            exp290_geometry_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(raw).hexdigest()
