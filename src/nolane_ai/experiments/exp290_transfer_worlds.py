from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.experiments.exp290_transfer_geometry import (
    EXP290_GEOMETRY,
    exp290_geometry_digest,
)
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class Exp290TransferBatch:
    source_surface_events: torch.Tensor
    target_surface_events: torch.Tensor
    source_variable_states: torch.Tensor
    target_variable_states: torch.Tensor
    source_restart_orders: torch.Tensor
    target_restart_orders: torch.Tensor
    source_restart_value_orders: torch.Tensor
    target_restart_value_orders: torch.Tensor
    metadata: dict[str, Any]
    digest: str
    replicate: int
    rng_stream: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = _canonical_json(
        {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "byteorder": tensor_byteorder(),
        }
    )
    return header + b"\0" + tensor_raw_bytes(cpu)


def _digest(
    metadata: dict[str, Any],
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json(metadata))
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_part(tensor))
    return hasher.hexdigest()


def _opaque(prefix: str, seed: int, pair_index: int, variable_index: int) -> str:
    token = hashlib.sha256(
        f"{prefix}|{seed}|{pair_index}|{variable_index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}_{token}"


def _problem_payload(
    names: list[str],
    solution: list[int],
    clauses: list[tuple[int, int, int, int]],
    world_id: str,
) -> dict[str, Any]:
    constraints: list[dict[str, Any]] = []
    all_rows = ((0, 0), (0, 1), (1, 0), (1, 1))
    for clause_index, (left, left_value, right, right_value) in enumerate(clauses):
        allowed = [
            list(row)
            for row in all_rows
            if row != (left_value, right_value)
        ]
        constraints.append(
            {
                "name": f"pair-{clause_index}",
                "scope": [names[left], names[right]],
                "allowed": allowed,
            }
        )
    return {
        "world_id": world_id,
        "variables": [{"name": name, "domain": [0, 1]} for name in names],
        "constraints": constraints,
        "evaluator_solution": [
            [names[index], int(value)] for index, value in enumerate(solution)
        ],
    }


class Exp290TransferGenerator:
    """Deterministic source/target structural isomorphs for EXP-290 DEVELOPMENT."""

    def __init__(self, *, root_seed: str) -> None:
        if not root_seed:
            raise ValueError("root_seed must be non-empty")
        self.root_seed = root_seed

    def make_batch(
        self,
        *,
        replicate: int,
        rng_stream: str,
        device: str | torch.device = "cpu",
    ) -> Exp290TransferBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if rng_stream not in {"augmentation", "evaluation"}:
            raise ValueError("rng_stream must be augmentation or evaluation")

        geometry = EXP290_GEOMETRY
        seed = derive_stream_seed(
            self.root_seed,
            "EXP-290",
            replicate,
            rng_stream,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)

        batch_size = geometry.batch_size
        variables = geometry.variables
        d_model = geometry.d_model
        timesteps = geometry.timesteps
        restarts = geometry.restarts

        source_variable_states = torch.empty(batch_size, variables, d_model)
        target_variable_states = torch.empty(batch_size, variables, d_model)
        source_surface_events = torch.empty(batch_size, timesteps, d_model)
        target_surface_events = torch.empty(batch_size, timesteps, d_model)
        source_restart_orders = torch.empty(
            batch_size,
            restarts,
            variables,
            dtype=torch.long,
        )
        target_restart_orders = torch.empty(
            batch_size,
            restarts,
            variables,
            dtype=torch.long,
        )
        source_restart_value_orders = torch.empty(
            batch_size,
            restarts,
            variables,
            2,
            dtype=torch.long,
        )
        target_restart_value_orders = torch.empty(
            batch_size,
            restarts,
            variables,
            2,
            dtype=torch.long,
        )

        pairs: list[dict[str, Any]] = []
        for pair_index in range(batch_size):
            canonical_solution = torch.randint(
                0,
                2,
                (variables,),
                generator=generator,
                dtype=torch.long,
            )
            source_role_order = torch.randperm(variables, generator=generator)
            target_role_order = torch.randperm(variables, generator=generator)
            source_roles = [int(value) for value in source_role_order.tolist()]
            target_roles = [int(value) for value in target_role_order.tolist()]
            target_index_for_role = {
                role: index for index, role in enumerate(target_roles)
            }
            hidden_source_to_target_index = [
                target_index_for_role[role] for role in source_roles
            ]

            source_names = [
                _opaque("src", seed, pair_index, index)
                for index in range(variables)
            ]
            target_names = [
                _opaque("tgt", seed, pair_index, index)
                for index in range(variables)
            ]

            role_pairs = ((0, 1), (2, 3), (4, 5))
            canonical_clauses: list[tuple[int, int, int, int]] = []
            for left_role, right_role in role_pairs:
                canonical_clauses.append(
                    (
                        left_role,
                        1 - int(canonical_solution[left_role].item()),
                        right_role,
                        1 - int(canonical_solution[right_role].item()),
                    )
                )

            structural_prototypes = torch.randn(
                variables,
                d_model,
                generator=generator,
            )
            structural_prototypes = structural_prototypes / structural_prototypes.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            source_states = torch.stack(
                [structural_prototypes[role] for role in source_roles]
            )
            target_states = torch.stack(
                [structural_prototypes[role] for role in target_roles]
            )
            if geometry.noise_std:
                source_states = source_states + geometry.noise_std * torch.randn(
                    variables,
                    d_model,
                    generator=generator,
                )
                target_states = target_states + geometry.noise_std * torch.randn(
                    variables,
                    d_model,
                    generator=generator,
                )

            source_variable_states[pair_index] = source_states
            target_variable_states[pair_index] = target_states

            for restart_index in range(restarts):
                source_restart_orders[pair_index, restart_index] = torch.randperm(
                    variables,
                    generator=generator,
                )
                target_restart_orders[pair_index, restart_index] = torch.randperm(
                    variables,
                    generator=generator,
                )
                for source_index, role in enumerate(source_roles):
                    solution_value = int(canonical_solution[role].item())
                    source_restart_value_orders[
                        pair_index,
                        restart_index,
                        source_index,
                    ] = torch.tensor(
                        [solution_value, 1 - solution_value],
                        dtype=torch.long,
                    )
                for target_index, role in enumerate(target_roles):
                    solution_value = int(canonical_solution[role].item())
                    target_restart_value_orders[
                        pair_index,
                        restart_index,
                        target_index,
                    ] = torch.tensor(
                        [solution_value, 1 - solution_value],
                        dtype=torch.long,
                    )

            source_dead_end_opportunities: list[dict[str, Any]] = []
            target_transfer_opportunities: list[dict[str, Any]] = []
            for clause_index, (
                left_role,
                left_value,
                right_role,
                right_value,
            ) in enumerate(canonical_clauses):
                source_left = source_roles.index(left_role)
                source_right = source_roles.index(right_role)
                target_left = target_roles.index(left_role)
                target_right = target_roles.index(right_role)
                restart_index = clause_index

                source_remaining = [
                    index
                    for index in range(variables)
                    if index not in {source_left, source_right}
                ]
                target_remaining = [
                    index
                    for index in range(variables)
                    if index not in {target_left, target_right}
                ]
                source_restart_orders[pair_index, restart_index] = torch.tensor(
                    [source_left, source_right, *source_remaining],
                    dtype=torch.long,
                )
                target_restart_orders[pair_index, restart_index] = torch.tensor(
                    [target_left, target_right, *target_remaining],
                    dtype=torch.long,
                )
                source_restart_value_orders[
                    pair_index,
                    restart_index,
                    source_left,
                ] = torch.tensor([left_value, 1 - left_value], dtype=torch.long)
                source_restart_value_orders[
                    pair_index,
                    restart_index,
                    source_right,
                ] = torch.tensor([right_value, 1 - right_value], dtype=torch.long)
                target_restart_value_orders[
                    pair_index,
                    restart_index,
                    target_left,
                ] = torch.tensor([left_value, 1 - left_value], dtype=torch.long)
                target_restart_value_orders[
                    pair_index,
                    restart_index,
                    target_right,
                ] = torch.tensor([right_value, 1 - right_value], dtype=torch.long)

                source_clause = [
                    [source_names[source_left], left_value],
                    [source_names[source_right], right_value],
                ]
                target_clause = [
                    [target_names[target_left], left_value],
                    [target_names[target_right], right_value],
                ]
                source_clause_digest = _sha(
                    {
                        "pair_index": pair_index,
                        "source_clause": source_clause,
                        "restart_index": restart_index,
                    }
                )
                source_dead_end_opportunities.append(
                    {
                        "source_clause": source_clause,
                        "restart_index": restart_index,
                        "source_clause_digest": source_clause_digest,
                    }
                )
                target_transfer_opportunities.append(
                    {
                        "source_clause_digest": source_clause_digest,
                        "mapped_target_clause": target_clause,
                        "target_restart_index": restart_index,
                        "pair_lineage_digest": _sha(
                            {
                                "pair_index": pair_index,
                                "target_clause": target_clause,
                                "restart_index": restart_index,
                            }
                        ),
                    }
                )

            source_solution = [
                int(canonical_solution[role].item()) for role in source_roles
            ]
            target_solution = [
                int(canonical_solution[role].item()) for role in target_roles
            ]
            source_clauses: list[tuple[int, int, int, int]] = []
            target_clauses: list[tuple[int, int, int, int]] = []
            for left_role, left_value, right_role, right_value in canonical_clauses:
                source_clauses.append(
                    (
                        source_roles.index(left_role),
                        left_value,
                        source_roles.index(right_role),
                        right_value,
                    )
                )
                target_clauses.append(
                    (
                        target_roles.index(left_role),
                        left_value,
                        target_roles.index(right_role),
                        right_value,
                    )
                )

            source_problem = _problem_payload(
                source_names,
                source_solution,
                source_clauses,
                f"EXP-290-src-r{replicate}-p{pair_index}-{rng_stream}",
            )
            target_problem = _problem_payload(
                target_names,
                target_solution,
                target_clauses,
                f"EXP-290-tgt-r{replicate}-p{pair_index}-{rng_stream}",
            )

            source_surface_events[pair_index] = source_states[
                source_restart_orders[pair_index, 0, :timesteps]
            ]
            target_surface_events[pair_index] = target_states[
                target_restart_orders[pair_index, 0, :timesteps]
            ]

            pairs.append(
                {
                    "pair_index": pair_index,
                    "source_surface_names": source_names,
                    "target_surface_names": target_names,
                    "hidden_source_to_target_index": hidden_source_to_target_index,
                    "surface_identity_overlap": False,
                    "structurally_isomorphic": True,
                    "target_local_clause_learning_enabled": False,
                    "hidden_correspondence_delivered_to_learned_mode": False,
                    "target_validity_truth_delivered_to_learned_mode": False,
                    "transfer_opportunity_manifest_delivered_to_learned_mode": False,
                    "surface_names_delivered_as_neural_features": False,
                    "source_problem": source_problem,
                    "target_problem": target_problem,
                    "source_dead_end_opportunities": source_dead_end_opportunities,
                    "target_transfer_opportunities": target_transfer_opportunities,
                    "transfer_opportunity_manifest_frozen_before_mode_execution": True,
                }
            )

        metadata: dict[str, Any] = {
            "scope": "synthetic-exp290-clause-transfer-development",
            "root_seed": self.root_seed,
            "experiment_id": "EXP-290",
            "replicate": int(replicate),
            "rng_stream": rng_stream,
            "seed": int(seed),
            "geometry_digest": exp290_geometry_digest(),
            "pairs": pairs,
        }
        tensors = (
            ("source_surface_events", source_surface_events),
            ("target_surface_events", target_surface_events),
            ("source_variable_states", source_variable_states),
            ("target_variable_states", target_variable_states),
            ("source_restart_orders", source_restart_orders),
            ("target_restart_orders", target_restart_orders),
            ("source_restart_value_orders", source_restart_value_orders),
            ("target_restart_value_orders", target_restart_value_orders),
        )
        tensors_device = tuple(
            (name, tensor.to(device)) for name, tensor in tensors
        )
        by_name = {name: tensor for name, tensor in tensors_device}
        return Exp290TransferBatch(
            source_surface_events=by_name["source_surface_events"],
            target_surface_events=by_name["target_surface_events"],
            source_variable_states=by_name["source_variable_states"],
            target_variable_states=by_name["target_variable_states"],
            source_restart_orders=by_name["source_restart_orders"],
            target_restart_orders=by_name["target_restart_orders"],
            source_restart_value_orders=by_name["source_restart_value_orders"],
            target_restart_value_orders=by_name["target_restart_value_orders"],
            metadata=metadata,
            digest=_digest(metadata, tensors_device),
            replicate=replicate,
            rng_stream=rng_stream,
        )
