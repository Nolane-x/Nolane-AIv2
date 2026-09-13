from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class Exp290SurfaceBatch:
    surface_events: torch.Tensor
    variable_states: torch.Tensor
    solution_targets: torch.Tensor
    restart_orders: torch.Tensor
    restart_value_orders: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Exp290TransferPairBatch:
    source: Exp290SurfaceBatch
    target: Exp290SurfaceBatch
    metadata: dict[str, Any]
    digest: str
    replicate: int
    rng_stream: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _child_seed(parent_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"EXP-290|{parent_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _tensor_digest_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = _canonical_json(
        {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "byteorder": tensor_byteorder()}
    )
    return header + b"\0" + tensor_raw_bytes(cpu)


def _pair_digest(metadata: dict[str, Any], tensors: tuple[tuple[str, torch.Tensor], ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json(metadata))
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(tensor))
    return hasher.hexdigest()


def _nonidentity_permutation(variables: int, generator: torch.Generator) -> torch.Tensor:
    permutation = torch.randperm(variables, generator=generator)
    identity = torch.arange(variables, dtype=torch.long)
    if torch.equal(permutation, identity):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def _source_to_target_mapping(source_visible_to_latent: torch.Tensor, target_visible_to_latent: torch.Tensor) -> list[int]:
    mapping: list[int] = []
    for source_visible in range(int(source_visible_to_latent.numel())):
        latent = int(source_visible_to_latent[source_visible].item())
        target_hits = (target_visible_to_latent == latent).nonzero(as_tuple=False).flatten()
        if int(target_hits.numel()) != 1:
            raise ValueError("EXP-290 surface permutation is not bijective")
        mapping.append(int(target_hits[0].item()))
    return mapping


def _view_batch(
    *,
    latent_states: torch.Tensor,
    latent_targets: torch.Tensor,
    latent_decoys: list[list[int]],
    visible_to_latent: torch.Tensor,
    timesteps: int,
    restarts: int,
    noise_std: float,
    generator: torch.Generator,
    world_prefix: str,
    replicate: int,
    rng_stream: str,
    device: str | torch.device,
) -> tuple[Exp290SurfaceBatch, list[dict[str, Any]]]:
    batch_size, variables, d_model = latent_states.shape
    visible_states = torch.empty_like(latent_states)
    visible_targets = torch.empty_like(latent_targets)
    restart_orders = torch.empty(batch_size, restarts, variables, dtype=torch.long)
    restart_value_orders = torch.empty(batch_size, restarts, variables, 2, dtype=torch.long)
    visible_decoys: list[list[int]] = []
    episode_meta: list[dict[str, Any]] = []

    for batch_index in range(batch_size):
        permutation = visible_to_latent[batch_index]
        visible_states[batch_index] = latent_states[batch_index, permutation]
        visible_targets[batch_index] = latent_targets[batch_index, permutation]
        inverse = {int(latent.item()): visible for visible, latent in enumerate(permutation)}
        decoys = sorted(inverse[index] for index in latent_decoys[batch_index])
        visible_decoys.append(decoys)
        decoy_set = set(decoys)
        for restart_index in range(restarts):
            restart_orders[batch_index, restart_index] = torch.randperm(variables, generator=generator)
            for visible_index in range(variables):
                target = int(visible_targets[batch_index, visible_index].item())
                wrong = 1 - target
                repeated_probe = visible_index in decoy_set and restart_index in {0, 1}
                values = [wrong, target] if repeated_probe else [target, wrong]
                restart_value_orders[batch_index, restart_index, visible_index] = torch.tensor(values, dtype=torch.long)

        world_id = f"EXP-290-{world_prefix}-r{replicate}-e{batch_index}-{rng_stream}"
        problem_payload = {
            "world_id": world_id,
            "visible_solution_targets": visible_targets[batch_index].tolist(),
            "visible_decoy_variables": decoys,
            "visible_to_latent_variable_permutation": permutation.tolist(),
        }
        episode_meta.append(
            {
                "episode_index": batch_index,
                "world_id": world_id,
                "problem_digest": _sha256_json(problem_payload),
                "visible_decoy_variables": decoys,
                "visible_solution_targets": visible_targets[batch_index].tolist(),
                "evaluator_metadata_delivered_to_model": False,
            }
        )

    if noise_std:
        visible_states = visible_states + noise_std * torch.randn(
            visible_states.shape, generator=generator, dtype=visible_states.dtype
        )

    event_rows: list[torch.Tensor] = []
    event_variable_orders = torch.empty(batch_size, timesteps, dtype=torch.long)
    for batch_index in range(batch_size):
        base = torch.randperm(variables, generator=generator)
        if timesteps <= variables:
            event_variable_orders[batch_index] = base[:timesteps]
        else:
            repeats = (timesteps + variables - 1) // variables
            event_variable_orders[batch_index] = base.repeat(repeats)[:timesteps]
    for timestep in range(timesteps):
        variable_index = event_variable_orders[:, timestep]
        gather = variable_index.view(batch_size, 1, 1).expand(-1, 1, d_model)
        event = visible_states.gather(1, gather).squeeze(1)
        if noise_std:
            event = event + noise_std * torch.randn(event.shape, generator=generator, dtype=event.dtype)
        event_rows.append(event)
    surface_events = torch.stack(event_rows, dim=1)

    metadata = {
        "world_prefix": world_prefix,
        "replicate": int(replicate),
        "rng_stream": rng_stream,
        "event_order_randomized": True,
        "variable_identity_randomized": True,
        "variable_order_randomized": True,
        "value_labels_remapped": False,
        "episodes": episode_meta,
    }
    return (
        Exp290SurfaceBatch(
            surface_events=surface_events.to(device),
            variable_states=visible_states.to(device),
            solution_targets=visible_targets.to(device),
            restart_orders=restart_orders.to(device),
            restart_value_orders=restart_value_orders.to(device),
            metadata=metadata,
        ),
        episode_meta,
    )


class Exp290TransferGenerator:
    """Deterministic paired source/target surfaces for EXP-290 DEVELOPMENT.

    Hidden source→target mappings are serialized only in pair-level evaluator
    metadata. Model-visible tensors contain structural state but never the
    explicit mapping or target-equivalent clause labels.
    """

    def __init__(self, *, root_seed: str) -> None:
        if not isinstance(root_seed, str) or not root_seed:
            raise ValueError("root_seed must be non-empty")
        self.root_seed = root_seed

    def make_pair(
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
        device: str | torch.device = "cpu",
    ) -> Exp290TransferPairBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if min(batch_size, timesteps, restarts, variables, d_model) <= 0:
            raise ValueError("EXP-290 paired geometry must be positive")
        if decoys <= 0 or decoys > variables - 1:
            raise ValueError("decoys must leave at least one productive variable")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if rng_stream not in {"augmentation", "evaluation"}:
            raise ValueError("rng_stream must be augmentation or evaluation")

        stream_seed = derive_stream_seed(self.root_seed, "EXP-290", replicate, rng_stream)
        latent_generator = torch.Generator(device="cpu").manual_seed(_child_seed(stream_seed, "latent"))
        source_generator = torch.Generator(device="cpu").manual_seed(_child_seed(stream_seed, "source-surface"))
        target_generator = torch.Generator(device="cpu").manual_seed(_child_seed(stream_seed, "target-surface"))

        latent_targets = torch.randint(0, 2, (batch_size, variables), generator=latent_generator, dtype=torch.long)
        latent_decoys: list[list[int]] = []
        for _ in range(batch_size):
            permutation = torch.randperm(variables, generator=latent_generator)
            latent_decoys.append([int(value.item()) for value in permutation[:decoys]])

        solution_axis = torch.randn(d_model, generator=latent_generator)
        solution_axis = solution_axis / solution_axis.norm().clamp_min(1e-8)
        decoy_axis = torch.randn(d_model, generator=latent_generator)
        decoy_axis = decoy_axis / decoy_axis.norm().clamp_min(1e-8)
        latent_states = torch.randn(batch_size, variables, d_model, generator=latent_generator)
        latent_states = latent_states / latent_states.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        signs = latent_targets.to(torch.float32).mul(2.0).sub(1.0)
        latent_states = latent_states + 0.55 * signs.unsqueeze(-1) * solution_axis
        decoy_mask = torch.zeros(batch_size, variables, dtype=torch.float32)
        for batch_index, values in enumerate(latent_decoys):
            decoy_mask[batch_index, values] = 1.0
        latent_states = latent_states + 0.25 * decoy_mask.unsqueeze(-1) * decoy_axis

        source_visible_to_latent = torch.stack(
            [_nonidentity_permutation(variables, source_generator) for _ in range(batch_size)]
        )
        target_visible_to_latent_rows: list[torch.Tensor] = []
        mappings: list[list[int]] = []
        for batch_index in range(batch_size):
            source_perm = source_visible_to_latent[batch_index]
            target_perm = _nonidentity_permutation(variables, target_generator)
            mapping = _source_to_target_mapping(source_perm, target_perm)
            if mapping == list(range(variables)):
                target_perm = torch.roll(target_perm, shifts=1)
                mapping = _source_to_target_mapping(source_perm, target_perm)
            target_visible_to_latent_rows.append(target_perm)
            mappings.append(mapping)
        target_visible_to_latent = torch.stack(target_visible_to_latent_rows)

        source, source_meta = _view_batch(
            latent_states=latent_states,
            latent_targets=latent_targets,
            latent_decoys=latent_decoys,
            visible_to_latent=source_visible_to_latent,
            timesteps=timesteps,
            restarts=restarts,
            noise_std=noise_std,
            generator=source_generator,
            world_prefix="source",
            replicate=replicate,
            rng_stream=rng_stream,
            device=device,
        )
        target, target_meta = _view_batch(
            latent_states=latent_states,
            latent_targets=latent_targets,
            latent_decoys=latent_decoys,
            visible_to_latent=target_visible_to_latent,
            timesteps=timesteps,
            restarts=restarts,
            noise_std=noise_std,
            generator=target_generator,
            world_prefix="target",
            replicate=replicate,
            rng_stream=rng_stream,
            device=device,
        )

        latent_problem_digest = _sha256_json(
            {
                "replicate": replicate,
                "rng_stream": rng_stream,
                "latent_solution_targets": latent_targets.tolist(),
                "latent_decoys": latent_decoys,
            }
        )
        episodes: list[dict[str, Any]] = []
        for batch_index in range(batch_size):
            episodes.append(
                {
                    "episode_index": batch_index,
                    "latent_problem_digest": latent_problem_digest,
                    "source_world_id": source_meta[batch_index]["world_id"],
                    "target_world_id": target_meta[batch_index]["world_id"],
                    "source_problem_digest": source_meta[batch_index]["problem_digest"],
                    "target_problem_digest": target_meta[batch_index]["problem_digest"],
                    "evaluator_only_source_to_target_variable_permutation": mappings[batch_index],
                    "source_visible_to_latent_variable_permutation": source_visible_to_latent[batch_index].tolist(),
                    "target_visible_to_latent_variable_permutation": target_visible_to_latent[batch_index].tolist(),
                    "mapping_delivered_to_learned_mode": False,
                    "value_labels_remapped": False,
                    "globally_solvable": True,
                    "evaluator_truth_gates_learned_action": False,
                }
            )

        metadata: dict[str, Any] = {
            "scope": "synthetic-exp290-structural-clause-transfer-development",
            "experiment_id": "EXP-290",
            "root_seed": self.root_seed,
            "replicate": int(replicate),
            "rng_stream": rng_stream,
            "stream_seed": int(stream_seed),
            "latent_problem_digest": latent_problem_digest,
            "surface_randomization": {
                "variable_identity_randomized": True,
                "variable_order_randomized": True,
                "event_order_randomized": True,
                "value_labels_remapped": False,
            },
            "mapping_delivered_to_learned_mode": False,
            "episodes": episodes,
        }
        tensors = (
            ("source_surface_events", source.surface_events),
            ("source_variable_states", source.variable_states),
            ("source_solution_targets", source.solution_targets),
            ("source_restart_orders", source.restart_orders),
            ("source_restart_value_orders", source.restart_value_orders),
            ("target_surface_events", target.surface_events),
            ("target_variable_states", target.variable_states),
            ("target_solution_targets", target.solution_targets),
            ("target_restart_orders", target.restart_orders),
            ("target_restart_value_orders", target.restart_value_orders),
        )
        return Exp290TransferPairBatch(
            source=source,
            target=target,
            metadata=metadata,
            digest=_pair_digest(metadata, tensors),
            replicate=int(replicate),
            rng_stream=rng_stream,
        )
