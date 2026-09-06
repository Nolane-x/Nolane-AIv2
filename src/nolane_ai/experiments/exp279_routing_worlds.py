from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes

STRATA = ("PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL")


@dataclass(frozen=True, slots=True)
class Exp279RoutingBatch:
    surface_events: torch.Tensor
    variable_states: torch.Tensor
    incidence: torch.Tensor
    targets: torch.Tensor
    stratum: str
    metadata: dict[str, Any]
    digest: str
    replicate: int
    rng_stream: str


def _tensor_digest_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "byteorder": tensor_byteorder()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + tensor_raw_bytes(cpu)


def _batch_digest(metadata: dict[str, Any], tensors: tuple[tuple[str, torch.Tensor], ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(tensor))
    return hasher.hexdigest()


class Exp279RoutingGenerator:
    """Deterministic paired DEVELOPMENT worlds with predeclared EXP-279 strata."""

    def __init__(self, *, root_seed: str) -> None:
        if not root_seed:
            raise ValueError("root_seed must be non-empty")
        self.root_seed = root_seed

    def make_batch(
        self,
        *,
        replicate: int,
        batch_size: int,
        timesteps: int,
        variables: int,
        constraints: int,
        d_model: int,
        noise_std: float,
        rng_stream: str,
        stratum: str,
        scope: str = "synthetic-exp279-routing-development",
        device: str | torch.device = "cpu",
    ) -> Exp279RoutingBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if min(batch_size, timesteps, variables, constraints, d_model) <= 0:
            raise ValueError("EXP-279 batch geometry must be positive")
        if constraints > variables:
            raise ValueError("constraints cannot exceed variables")
        if timesteps < constraints:
            raise ValueError("timesteps must cover each constraint component")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if rng_stream not in {"augmentation", "evaluation"}:
            raise ValueError("rng_stream must be augmentation or evaluation")
        if stratum not in STRATA:
            raise ValueError(f"stratum must be one of {STRATA}")
        if not scope:
            raise ValueError("scope must be non-empty")

        seed = derive_stream_seed(self.root_seed, "EXP-279", replicate, rng_stream)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        base = torch.arange(variables, dtype=torch.long) % constraints
        membership = torch.stack(
            [base[torch.randperm(variables, generator=generator)] for _ in range(batch_size)],
            dim=0,
        )
        incidence = torch.zeros(batch_size, constraints, variables, dtype=torch.float32)
        incidence.scatter_(1, membership.unsqueeze(1), 1.0)

        anchors = torch.randint(0, 2, (batch_size, constraints), generator=generator, dtype=torch.long)
        targets = anchors.gather(1, membership)

        component_codes = torch.randn(batch_size, constraints, d_model, generator=generator)
        component_codes = component_codes / component_codes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        anchor_axis = torch.randn(d_model, generator=generator)
        anchor_axis = anchor_axis / anchor_axis.norm().clamp_min(1e-8)
        gather_index = membership.unsqueeze(-1).expand(-1, -1, d_model)
        variable_states = component_codes.gather(1, gather_index)
        variable_sign = targets.to(torch.float32).mul(2.0).sub(1.0).unsqueeze(-1)

        if stratum == "PROPAGATION_FIT":
            variable_states = variable_states + 0.75 * variable_sign * anchor_axis
            variable_noise = noise_std * 0.5
            event_anchor_scale = 0.45
            semantics = "dense aligned factor support; low residual ambiguity after propagation"
        elif stratum == "BRANCH_FIT":
            variable_noise = noise_std * 2.0 + 0.05
            event_anchor_scale = 1.25
            semantics = "ambiguous local factor evidence; stronger sequential anchor evidence for recurrent branch deliberation"
        else:
            component_mask = (membership % 2 == 0).to(torch.float32).unsqueeze(-1)
            variable_states = variable_states + 0.55 * component_mask * variable_sign * anchor_axis
            variable_noise = noise_std
            event_anchor_scale = 0.85
            semantics = "mixed factor support with residual ambiguity requiring routing discrimination"

        if variable_noise:
            variable_states = variable_states + variable_noise * torch.randn(
                batch_size, variables, d_model, generator=generator
            )

        order = torch.randperm(constraints, generator=generator)
        event_rows: list[torch.Tensor] = []
        for step in range(timesteps):
            component = int(order[step % constraints].item())
            sign = anchors[:, component].to(torch.float32).mul(2.0).sub(1.0).unsqueeze(-1)
            event = component_codes[:, component, :] + event_anchor_scale * sign * anchor_axis.unsqueeze(0)
            if noise_std:
                event = event + noise_std * torch.randn(batch_size, d_model, generator=generator)
            event_rows.append(event)
        surface_events = torch.stack(event_rows, dim=1)

        metadata: dict[str, Any] = {
            "scope": scope,
            "root_seed": self.root_seed,
            "experiment_id": "EXP-279",
            "replicate": replicate,
            "rng_stream": rng_stream,
            "seed": seed,
            "stratum": stratum,
            "stratum_semantics": semantics,
            "predeclared_structure_fit_strata": list(STRATA),
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "noise_std": float(noise_std),
        }

        surface_events = surface_events.to(device)
        variable_states = variable_states.to(device)
        incidence = incidence.to(device)
        targets = targets.to(device)
        tensors = (
            ("surface_events", surface_events),
            ("variable_states", variable_states),
            ("incidence", incidence),
            ("targets", targets),
        )
        return Exp279RoutingBatch(
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
            targets=targets,
            stratum=stratum,
            metadata=metadata,
            digest=_batch_digest(metadata, tensors),
            replicate=replicate,
            rng_stream=rng_stream,
        )
