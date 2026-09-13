from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class Exp287ConflictBatch:
    surface_events: torch.Tensor
    variable_states: torch.Tensor
    solution_targets: torch.Tensor
    bad_branch_values: torch.Tensor
    core_masks: torch.Tensor
    decoy_order: torch.Tensor
    metadata: dict[str, Any]
    digest: str
    replicate: int
    rng_stream: str


def _tensor_digest_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "byteorder": tensor_byteorder(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + tensor_raw_bytes(cpu)


def _batch_digest(
    metadata: dict[str, Any],
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(tensor))
    return hasher.hexdigest()


class Exp287ConflictGenerator:
    """Deterministic synthetic conflict worlds for learned-localization DEVELOPMENT."""

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
        decoys: int,
        d_model: int,
        noise_std: float,
        rng_stream: str,
        scope: str = "synthetic-exp287-conflict-development",
        device: str | torch.device = "cpu",
    ) -> Exp287ConflictBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if min(batch_size, timesteps, variables, d_model) <= 0:
            raise ValueError("EXP-287 batch geometry must be positive")
        if variables < 2:
            raise ValueError("EXP-287 requires room for a two-variable core")
        if decoys < 0 or decoys > variables - 2:
            raise ValueError("decoys must leave two variables for the conflict core")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if rng_stream not in {"augmentation", "evaluation"}:
            raise ValueError("rng_stream must be augmentation or evaluation")
        if not scope:
            raise ValueError("scope must be non-empty")

        seed = derive_stream_seed(self.root_seed, "EXP-287", replicate, rng_stream)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        solution_targets = torch.randint(
            0, 2, (batch_size, variables), generator=generator, dtype=torch.long
        )
        bad_branch_values = 1 - solution_targets

        core_masks = torch.zeros(batch_size, variables, dtype=torch.float32)
        order_rows: list[torch.Tensor] = []
        episodes: list[dict[str, Any]] = []
        for batch_index in range(batch_size):
            permutation = torch.randperm(variables, generator=generator)
            decoy_variables = permutation[:decoys]
            core_variables = permutation[decoys : decoys + 2]
            remainder = permutation[decoys + 2 :]
            order_rows.append(torch.cat((decoy_variables, core_variables, remainder), dim=0))
            core_masks[batch_index, core_variables] = 1.0

            conflict_variable = int(core_variables[0].item())
            support_variable = int(core_variables[1].item())
            episodes.append(
                {
                    "episode_index": int(batch_index),
                    "globally_solvable": True,
                    "productive_solution_exists": True,
                    "local_contradiction_present": True,
                    "conflict_variable": conflict_variable,
                    "support_variable": support_variable,
                    "core_variables": [conflict_variable, support_variable],
                    "core_minimality": "generator_exact_by_construction",
                    "decoy_variables": [int(value.item()) for value in decoy_variables],
                    "bad_branch_value": int(
                        bad_branch_values[batch_index, conflict_variable].item()
                    ),
                    "solution_value_at_conflict_variable": int(
                        solution_targets[batch_index, conflict_variable].item()
                    ),
                }
            )

        decoy_order = torch.stack(order_rows, dim=0)
        target_sign = solution_targets.to(torch.float32).mul(2.0).sub(1.0)
        solution_axis = torch.randn(d_model, generator=generator)
        solution_axis = solution_axis / solution_axis.norm().clamp_min(1e-8)
        conflict_axis = torch.randn(d_model, generator=generator)
        conflict_axis = conflict_axis / conflict_axis.norm().clamp_min(1e-8)

        variable_states = torch.randn(
            batch_size, variables, d_model, generator=generator
        )
        variable_states = variable_states / variable_states.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        variable_states = (
            variable_states
            + 0.55 * target_sign.unsqueeze(-1) * solution_axis
            + 0.35 * core_masks.unsqueeze(-1) * conflict_axis
        )
        if noise_std:
            variable_states = variable_states + noise_std * torch.randn(
                batch_size, variables, d_model, generator=generator
            )

        event_rows: list[torch.Tensor] = []
        for step in range(timesteps):
            variable_index = decoy_order[:, step % variables]
            gather_index = variable_index.view(batch_size, 1, 1).expand(-1, 1, d_model)
            event = variable_states.gather(1, gather_index).squeeze(1)
            is_core = core_masks.gather(1, variable_index.view(batch_size, 1)).squeeze(1)
            event = event + 0.25 * is_core.unsqueeze(-1) * conflict_axis
            if noise_std:
                event = event + noise_std * torch.randn(
                    batch_size, d_model, generator=generator
                )
            event_rows.append(event)
        surface_events = torch.stack(event_rows, dim=1)

        metadata: dict[str, Any] = {
            "scope": scope,
            "root_seed": self.root_seed,
            "experiment_id": "EXP-287",
            "replicate": int(replicate),
            "rng_stream": rng_stream,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "timesteps": int(timesteps),
            "variables": int(variables),
            "decoys": int(decoys),
            "d_model": int(d_model),
            "noise_std": float(noise_std),
            "world_semantics": "globally_solvable_with_generator_exact_two_variable_conflict",
            "oracle_delivery_boundary": "core_materialized_only_after_current_contradiction",
            "episodes": episodes,
        }

        surface_events = surface_events.to(device)
        variable_states = variable_states.to(device)
        solution_targets = solution_targets.to(device)
        bad_branch_values = bad_branch_values.to(device)
        core_masks = core_masks.to(device)
        decoy_order = decoy_order.to(device)
        tensors = (
            ("surface_events", surface_events),
            ("variable_states", variable_states),
            ("solution_targets", solution_targets),
            ("bad_branch_values", bad_branch_values),
            ("core_masks", core_masks),
            ("decoy_order", decoy_order),
        )
        return Exp287ConflictBatch(
            surface_events=surface_events,
            variable_states=variable_states,
            solution_targets=solution_targets,
            bad_branch_values=bad_branch_values,
            core_masks=core_masks,
            decoy_order=decoy_order,
            metadata=metadata,
            digest=_batch_digest(metadata, tensors),
            replicate=int(replicate),
            rng_stream=rng_stream,
        )
