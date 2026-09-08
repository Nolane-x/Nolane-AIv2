from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


SCHEMA = "NLM-EXP-277-POST-FREEZE-CHALLENGE-BATCH-V1"
SCOPE = "POST_FREEZE_CHALLENGE"


@dataclass(frozen=True, slots=True)
class Exp277ChallengeBatch:
    surface_events: torch.Tensor
    variable_states: torch.Tensor
    oracle_incidence: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, Any]
    arm_visible_metadata: dict[str, Any]
    digest: str
    replicate: int
    schema: str = SCHEMA
    scope: str = SCOPE
    experiment_id: str = "EXP-277"

    def arm_visible(self) -> dict[str, Any]:
        return {
            "surface_events": self.surface_events,
            "variable_states": self.variable_states,
            "metadata": dict(self.arm_visible_metadata),
        }


def _tensor_digest_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "byteorder": tensor_byteorder()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + tensor_raw_bytes(cpu)


def _seed_digest(challenge_seed: int) -> str:
    payload = f"NLM-EXP-277-CHALLENGE-SEED-DIGEST-V1\0{challenge_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _materialization_seed(challenge_seed: int, replicate: int) -> int:
    payload = f"NLM-EXP-277-POST-FREEZE-CHALLENGE-SEED-V1\0{challenge_seed}\0{replicate}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _batch_digest(metadata: dict[str, Any], tensors: tuple[tuple[str, torch.Tensor], ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-277-POST-FREEZE-CHALLENGE-BATCH-V1\0")
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(tensor))
    return hasher.hexdigest()


def build_exp277_challenge_batch(
    *,
    challenge_seed: int,
    replicate: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    d_model: int,
    noise_std: float,
    device: str | torch.device = "cpu",
) -> Exp277ChallengeBatch:
    """Materialize one deterministic EXP-277 post-freeze challenge batch.

    This API is deliberately separate from the DEVELOPMENT generator: callers
    provide only a post-freeze challenge seed plus frozen geometry. No
    DEVELOPMENT root seed, rng stream, or mutable scope is accepted.
    """

    if isinstance(challenge_seed, bool) or not isinstance(challenge_seed, int) or challenge_seed < 0:
        raise ValueError("challenge_seed must be a non-negative integer")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    if min(batch_size, timesteps, variables, constraints, d_model) <= 0:
        raise ValueError("EXP-277 challenge geometry must be positive")
    if constraints > variables:
        raise ValueError("constraints cannot exceed variables")
    if timesteps < constraints:
        raise ValueError("timesteps must cover every constraint component")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(_materialization_seed(challenge_seed, replicate))

    membership_rows: list[torch.Tensor] = []
    base = torch.arange(variables, dtype=torch.long) % constraints
    for _ in range(batch_size):
        membership_rows.append(base[torch.randperm(variables, generator=generator)])
    membership = torch.stack(membership_rows, dim=0)

    incidence = torch.zeros(batch_size, constraints, variables, dtype=torch.float32)
    incidence.scatter_(1, membership.unsqueeze(1), 1.0)

    labels = torch.randint(0, 2, (batch_size, constraints), generator=generator, dtype=torch.long)
    targets = labels.gather(1, membership)

    component_codes = torch.randn(batch_size, constraints, d_model, generator=generator, dtype=torch.float32)
    component_codes = component_codes / component_codes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    signal_axis = torch.randn(d_model, generator=generator, dtype=torch.float32)
    signal_axis = signal_axis / signal_axis.norm().clamp_min(1e-8)

    gather_index = membership.unsqueeze(-1).expand(-1, -1, d_model)
    variable_states = component_codes.gather(1, gather_index)
    if noise_std:
        variable_states = variable_states + noise_std * torch.randn(
            batch_size, variables, d_model, generator=generator, dtype=torch.float32
        )

    order = torch.randperm(constraints, generator=generator)
    event_rows: list[torch.Tensor] = []
    for step in range(timesteps):
        component = int(order[step % constraints].item())
        sign = labels[:, component].to(torch.float32).mul(2.0).sub(1.0).unsqueeze(-1)
        event = component_codes[:, component, :] + sign * signal_axis.unsqueeze(0)
        if noise_std:
            event = event + noise_std * torch.randn(
                batch_size, d_model, generator=generator, dtype=torch.float32
            )
        event_rows.append(event)
    surface_events = torch.stack(event_rows, dim=1)

    challenge_seed_digest = _seed_digest(challenge_seed)
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "experiment_id": "EXP-277",
        "replicate": replicate,
        "challenge_seed_digest": challenge_seed_digest,
        "batch_size": batch_size,
        "timesteps": timesteps,
        "variables": variables,
        "constraints": constraints,
        "d_model": d_model,
        "noise_std": float(noise_std),
        "surface_semantics": "uncompiled component/anchor events",
        "oracle_semantics": "ground-truth component-variable incidence",
    }
    arm_visible_metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "experiment_id": "EXP-277",
        "replicate": replicate,
        "challenge_seed_digest": challenge_seed_digest,
        "batch_size": batch_size,
        "timesteps": timesteps,
        "variables": variables,
        "constraints": constraints,
        "d_model": d_model,
        "noise_std": float(noise_std),
        "surface_semantics": "uncompiled component events",
    }

    surface_events = surface_events.to(device)
    variable_states = variable_states.to(device)
    incidence = incidence.to(device)
    targets = targets.to(device)
    tensors = (
        ("surface_events", surface_events),
        ("variable_states", variable_states),
        ("oracle_incidence", incidence),
        ("targets", targets),
    )
    return Exp277ChallengeBatch(
        surface_events=surface_events,
        variable_states=variable_states,
        oracle_incidence=incidence,
        targets=targets,
        metadata=metadata,
        arm_visible_metadata=arm_visible_metadata,
        digest=_batch_digest(metadata, tensors),
        replicate=replicate,
    )
