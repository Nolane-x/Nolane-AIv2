from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


STRATA = ("PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL")
_STRATUM_SEMANTICS = {
    "PROPAGATION_FIT": "truthful incidence strongly aligns variable component codes with anchored structure",
    "BRANCH_FIT": "truthful incidence retained while variable component cues are weakened and surface sequence carries more usable evidence",
    "MIXED_RESIDUAL": "truthful incidence with a deterministic mix of strong and weak structural cues",
}


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
    """Deterministic paired DEVELOPMENT worlds for frozen EXP-279 strata."""

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
            raise ValueError("timesteps must cover every constraint component")
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

        base_membership = torch.arange(variables, dtype=torch.long) % constraints
        memberships = [base_membership[torch.randperm(variables, generator=generator)] for _ in range(batch_size)]
        membership = torch.stack(memberships, dim=0)
        incidence = torch.zeros(batch_size, constraints, variables, dtype=torch.float32)
        incidence.scatter_(1, membership.unsqueeze(1), 1.0)

        anchors = torch.randint(0, 2, (batch_size, constraints), generator=generator, dtype=torch.long)
        targets = anchors.gather(1, membership)
        component_codes = torch.randn(batch_size, constraints, d_model, generator=generator, dtype=torch.float32)
        component_codes = component_codes / component_codes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        surface_axis = torch.randn(d_model, generator=generator, dtype=torch.float32)
        surface_axis = surface_axis / surface_axis.norm().clamp_min(1e-8)
        nuisance = torch.randn(batch_size, variables, d_model, generator=generator, dtype=torch.float32)
        nuisance = nuisance / nuisance.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        gather_index = membership.unsqueeze(-1).expand(-1, -1, d_model)
        structural = component_codes.gather(1, gather_index)
        if stratum == "PROPAGATION_FIT":
            variable_states = structural
            surface_scale = 0.65
        elif stratum == "BRANCH_FIT":
            variable_states = 0.20 * structural + 0.80 * nuisance
            surface_scale = 1.25
        else:
            mask = (torch.arange(variables) % 2 == 0).view(1, variables, 1)
            weak = 0.25 * structural + 0.75 * nuisance
            variable_states = torch.where(mask, structural, weak)
            surface_scale = 0.95

        if noise_std:
            variable_states = variable_states + noise_std * torch.randn(
                batch_size,
                variables,
                d_model,
                generator=generator,
                dtype=torch.float32,
            )

        component_order = torch.randperm(constraints, generator=generator)
        event_rows: list[torch.Tensor] = []
        for step in range(timesteps):
            component = int(component_order[step % constraints].item())
            sign = anchors[:, component].to(torch.float32).mul(2.0).sub(1.0).unsqueeze(-1)
            event = component_codes[:, component, :] + surface_scale * sign * surface_axis.unsqueeze(0)
            if noise_std:
                event = event + noise_std * torch.randn(
                    batch_size,
                    d_model,
                    generator=generator,
                    dtype=torch.float32,
                )
            event_rows.append(event)
        surface_events = torch.stack(event_rows, dim=1)

        metadata: dict[str, Any] = {
            "scope": scope,
            "root_seed": self.root_seed,
            "experiment_id": "EXP-279",
            "replicate": replicate,
            "rng_stream": rng_stream,
            "seed": seed,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "noise_std": float(noise_std),
            "stratum": stratum,
            "stratum_semantics": _STRATUM_SEMANTICS[stratum],
            "incidence_semantics": "truthful ground-truth constraint-variable membership; never a solution-label oracle",
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
