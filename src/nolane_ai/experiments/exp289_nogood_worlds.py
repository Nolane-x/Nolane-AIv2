from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.reasoning.cps import CanonicalProblemState, TableConstraint, Variable
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class Exp289NogoodBatch:
    surface_events: torch.Tensor
    variable_states: torch.Tensor
    solution_targets: torch.Tensor
    restart_orders: torch.Tensor
    restart_value_orders: torch.Tensor
    metadata: dict[str, Any]
    digest: str
    replicate: int
    rng_stream: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_digest_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = _canonical_json(
        {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "byteorder": tensor_byteorder(),
        }
    )
    return header + b"\0" + tensor_raw_bytes(cpu)


def _batch_digest(
    metadata: dict[str, Any],
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json(metadata))
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(tensor))
    return hasher.hexdigest()


def _problem_payload(targets: list[int], *, world_id: str) -> dict[str, Any]:
    variables = [
        {"name": f"v{index}", "domain": [0, 1]}
        for index in range(len(targets))
    ]
    constraints = [
        {
            "name": f"target-v{index}",
            "scope": [f"v{index}"],
            "allowed": [[int(value)]],
        }
        for index, value in enumerate(targets)
    ]
    return {
        "world_id": world_id,
        "variables": variables,
        "constraints": constraints,
    }


def problem_from_exp289_episode(episode: dict[str, Any]) -> CanonicalProblemState:
    """Reconstruct the bounded CSP used by one EXP-289 evaluator episode."""

    payload = episode.get("problem")
    if not isinstance(payload, dict):
        raise ValueError("EXP-289 episode is missing problem payload")
    variables_payload = payload.get("variables")
    constraints_payload = payload.get("constraints")
    if not isinstance(variables_payload, list) or not isinstance(constraints_payload, list):
        raise ValueError("EXP-289 problem payload is malformed")

    variables = tuple(
        Variable(
            name=str(item["name"]),
            domain=tuple(int(value) for value in item["domain"]),
        )
        for item in variables_payload
    )
    constraints = tuple(
        TableConstraint(
            name=str(item["name"]),
            scope=tuple(str(name) for name in item["scope"]),
            allowed=tuple(
                tuple(int(value) for value in row)
                for row in item["allowed"]
            ),
        )
        for item in constraints_payload
    )
    return CanonicalProblemState(
        variables=variables,
        constraints=constraints,
        world_id=str(payload.get("world_id", "")),
    )


class Exp289NogoodGenerator:
    """Deterministic DEVELOPMENT worlds with frozen cross-restart dead-end opportunities."""

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
        restarts: int,
        variables: int,
        decoys: int,
        d_model: int,
        noise_std: float,
        rng_stream: str,
        scope: str = "synthetic-exp289-nogood-development",
        device: str | torch.device = "cpu",
    ) -> Exp289NogoodBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if min(batch_size, timesteps, variables, d_model) <= 0:
            raise ValueError("EXP-289 batch geometry must be positive")
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        if decoys < 0 or decoys > variables - 1:
            raise ValueError("decoys must leave at least one productive variable")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if rng_stream not in {"augmentation", "evaluation"}:
            raise ValueError("rng_stream must be augmentation or evaluation")
        if not scope:
            raise ValueError("scope must be non-empty")

        seed = derive_stream_seed(self.root_seed, "EXP-289", replicate, rng_stream)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        solution_targets = torch.randint(
            0,
            2,
            (batch_size, variables),
            generator=generator,
            dtype=torch.long,
        )

        restart_orders = torch.empty(
            batch_size,
            restarts,
            variables,
            dtype=torch.long,
        )
        restart_value_orders = torch.empty(
            batch_size,
            restarts,
            variables,
            2,
            dtype=torch.long,
        )

        episode_decoys: list[list[int]] = []
        for batch_index in range(batch_size):
            decoy_permutation = torch.randperm(variables, generator=generator)
            decoy_variables = [
                int(value.item()) for value in decoy_permutation[:decoys]
            ]
            episode_decoys.append(decoy_variables)
            decoy_set = set(decoy_variables)

            for restart_index in range(restarts):
                restart_orders[batch_index, restart_index] = torch.randperm(
                    variables,
                    generator=generator,
                )
                for variable_index in range(variables):
                    target = int(solution_targets[batch_index, variable_index].item())
                    wrong = 1 - target
                    repeated_probe = (
                        variable_index in decoy_set and restart_index in {0, 1}
                    )
                    if repeated_probe:
                        restart_value_orders[
                            batch_index,
                            restart_index,
                            variable_index,
                        ] = torch.tensor([wrong, target], dtype=torch.long)
                    else:
                        restart_value_orders[
                            batch_index,
                            restart_index,
                            variable_index,
                        ] = torch.tensor([target, wrong], dtype=torch.long)

        target_sign = solution_targets.to(torch.float32).mul(2.0).sub(1.0)
        solution_axis = torch.randn(d_model, generator=generator)
        solution_axis = solution_axis / solution_axis.norm().clamp_min(1e-8)
        decoy_axis = torch.randn(d_model, generator=generator)
        decoy_axis = decoy_axis / decoy_axis.norm().clamp_min(1e-8)

        decoy_mask = torch.zeros(batch_size, variables, dtype=torch.float32)
        for batch_index, decoy_variables in enumerate(episode_decoys):
            if decoy_variables:
                decoy_mask[batch_index, decoy_variables] = 1.0

        variable_states = torch.randn(
            batch_size,
            variables,
            d_model,
            generator=generator,
        )
        variable_states = variable_states / variable_states.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        variable_states = (
            variable_states
            + 0.55 * target_sign.unsqueeze(-1) * solution_axis
            + 0.25 * decoy_mask.unsqueeze(-1) * decoy_axis
        )
        if noise_std:
            variable_states = variable_states + noise_std * torch.randn(
                batch_size,
                variables,
                d_model,
                generator=generator,
            )

        event_rows: list[torch.Tensor] = []
        for timestep in range(timesteps):
            variable_index = restart_orders[:, 0, timestep % variables]
            gather_index = variable_index.view(batch_size, 1, 1).expand(-1, 1, d_model)
            event = variable_states.gather(1, gather_index).squeeze(1)
            is_decoy = decoy_mask.gather(
                1,
                variable_index.view(batch_size, 1),
            ).squeeze(1)
            event = event + 0.20 * is_decoy.unsqueeze(-1) * decoy_axis
            if noise_std:
                event = event + noise_std * torch.randn(
                    batch_size,
                    d_model,
                    generator=generator,
                )
            event_rows.append(event)
        surface_events = torch.stack(event_rows, dim=1)

        episodes: list[dict[str, Any]] = []
        for batch_index in range(batch_size):
            targets = [
                int(solution_targets[batch_index, variable].item())
                for variable in range(variables)
            ]
            world_id = f"EXP-289-r{replicate}-e{batch_index}-{rng_stream}"
            problem = _problem_payload(targets, world_id=world_id)
            problem_digest = _sha256_json(problem)

            restart_payload = {
                "restart_orders": restart_orders[batch_index].tolist(),
                "restart_value_orders": restart_value_orders[batch_index].tolist(),
            }
            episode_digest = _sha256_json(
                {
                    "problem_digest": problem_digest,
                    "replicate": replicate,
                    "episode_index": batch_index,
                    "rng_stream": rng_stream,
                    "restart_payload": restart_payload,
                }
            )

            repeat_opportunities: list[dict[str, Any]] = []
            if restarts >= 2:
                for variable_index in episode_decoys[batch_index]:
                    target = targets[variable_index]
                    wrong = 1 - target
                    canonical_key = json.dumps(
                        [[f"v{variable_index}", wrong]],
                        separators=(",", ":"),
                    )
                    lineage_digest = _sha256_json(
                        {
                            "episode_digest": episode_digest,
                            "canonical_dead_end_key": canonical_key,
                            "first_restart_index": 0,
                            "later_restart_indices": [1],
                        }
                    )
                    repeat_opportunities.append(
                        {
                            "canonical_dead_end_key": canonical_key,
                            "first_restart_index": 0,
                            "later_restart_indices": [1],
                            "world_restart_lineage_digest": lineage_digest,
                        }
                    )

            valid_solution = [
                [f"v{variable}", int(value)]
                for variable, value in enumerate(targets)
            ]
            opportunity_count = len(repeat_opportunities)
            episodes.append(
                {
                    "episode_index": batch_index,
                    "world_id": world_id,
                    "episode_digest": episode_digest,
                    "problem_digest": problem_digest,
                    "problem": problem,
                    "globally_solvable": True,
                    "productive_solution_exists": True,
                    "valid_solutions": [valid_solution],
                    "decoy_variables": episode_decoys[batch_index],
                    "repeat_opportunity_semantics": "generator_frozen_before_arm_execution",
                    "repeat_opportunities": repeat_opportunities,
                    "predeclared_repeat_opportunities": opportunity_count,
                    "rder_denominator_eligible": opportunity_count > 0,
                    "rder_defined": opportunity_count > 0,
                    "zero_opportunity_policy": "retain_raw_episode_exclude_from_rder_denominator",
                    "evaluator_metadata_delivered_to_arm": False,
                    "arm_specific_world_mutation": False,
                    "valid_solution_metadata_delivered_to_arm": False,
                    "opportunity_manifest_delivered_to_arm": False,
                    "restart_schedule_frozen_before_arm_execution": True,
                }
            )

        metadata: dict[str, Any] = {
            "scope": scope,
            "root_seed": self.root_seed,
            "experiment_id": "EXP-289",
            "replicate": int(replicate),
            "rng_stream": rng_stream,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "timesteps": int(timesteps),
            "restarts": int(restarts),
            "variables": int(variables),
            "decoys": int(decoys),
            "d_model": int(d_model),
            "noise_std": float(noise_std),
            "world_semantics": "globally_solvable_with_predeclared_repeat_dead_end_opportunities",
            "repeat_opportunity_semantics": "generator_frozen_before_arm_execution",
            "evaluator_metadata_delivered_to_arm": False,
            "arm_specific_world_mutation": False,
            "episodes": episodes,
        }

        surface_events = surface_events.to(device)
        variable_states = variable_states.to(device)
        solution_targets = solution_targets.to(device)
        restart_orders = restart_orders.to(device)
        restart_value_orders = restart_value_orders.to(device)
        tensors = (
            ("surface_events", surface_events),
            ("variable_states", variable_states),
            ("solution_targets", solution_targets),
            ("restart_orders", restart_orders),
            ("restart_value_orders", restart_value_orders),
        )
        return Exp289NogoodBatch(
            surface_events=surface_events,
            variable_states=variable_states,
            solution_targets=solution_targets,
            restart_orders=restart_orders,
            restart_value_orders=restart_value_orders,
            metadata=metadata,
            digest=_batch_digest(metadata, tensors),
            replicate=replicate,
            rng_stream=rng_stream,
        )
