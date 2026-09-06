from __future__ import annotations

import hashlib

STREAM_NAMES = (
    "model_init",
    "data_order",
    "environment",
    "intervention",
    "augmentation",
    "replay_sampling",
    "controller_noise",
    "evaluation",
)


def _uint63(payload: bytes) -> int:
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_stream_seed(root_seed: str, experiment_id: str, replicate: int, stream: str) -> int:
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown RNG stream: {stream}")
    if not experiment_id.startswith("EXP-"):
        raise ValueError("experiment_id must use EXP-### namespace")
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    payload = f"NLM-V0161|open|{root_seed}|{experiment_id}|{replicate}|{stream}".encode()
    return _uint63(payload)


def derive_challenge_seed(
    protocol_digest: str,
    beacon_pulse: str,
    experiment_id: str,
    stream: str,
    replicate: int,
) -> int:
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown RNG stream: {stream}")
    if not protocol_digest or not beacon_pulse:
        raise ValueError("protocol_digest and beacon_pulse are required")
    payload = (
        f"NLM-V0161|challenge|{protocol_digest}|{beacon_pulse}|"
        f"{experiment_id}|{stream}|{replicate}"
    ).encode()
    return _uint63(payload)
