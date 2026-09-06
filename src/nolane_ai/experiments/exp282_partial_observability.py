from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class Exp282PartialObservabilityBatch:
    observations: torch.Tensor
    targets: torch.Tensor
    visibility_mask: torch.Tensor
    metadata: dict[str, Any]
    digest: str


def _tensor_digest_part(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "byteorder": tensor_byteorder()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + tensor_raw_bytes(cpu)


def _batch_digest(metadata: dict[str, Any], *, observations: torch.Tensor, targets: torch.Tensor, visibility_mask: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in (
        ("observations", observations),
        ("targets", targets),
        ("visibility_mask", visibility_mask),
    ):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(tensor))
    return hasher.hexdigest()


class Exp282PartialObservabilityGenerator:
    """Deterministic synthetic partial-observability worlds for EXP-282 development.

    The latent world is derived from the `environment` stream so the same
    replicate has the same target under training and held-out observation
    streams. Observation masks/noise are independently derived from either
    `augmentation` or `evaluation`.
    """

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
        d_model: int,
        visibility_rate: float,
        noise_std: float,
        rng_stream: str,
        device: str | torch.device = "cpu",
    ) -> Exp282PartialObservabilityBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if min(batch_size, timesteps, variables, d_model) <= 0:
            raise ValueError("batch_size, timesteps, variables and d_model must be positive")
        if not (0.0 < visibility_rate < 1.0):
            raise ValueError("visibility_rate must be strictly between 0 and 1")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if rng_stream not in {"augmentation", "evaluation"}:
            raise ValueError("rng_stream must be augmentation or evaluation")

        latent_seed = derive_stream_seed(self.root_seed, "EXP-282", replicate, "environment")
        observation_seed = derive_stream_seed(self.root_seed, "EXP-282", replicate, rng_stream)
        code_seed = derive_stream_seed(self.root_seed, "EXP-282", 0, "intervention")
        latent_generator = torch.Generator(device="cpu").manual_seed(latent_seed)
        observation_generator = torch.Generator(device="cpu").manual_seed(observation_seed)
        code_generator = torch.Generator(device="cpu").manual_seed(code_seed)

        targets = torch.randint(0, 2, (batch_size, variables), generator=latent_generator, dtype=torch.long)
        visibility_mask = torch.rand(
            batch_size,
            timesteps,
            variables,
            generator=observation_generator,
        ) < visibility_rate
        if timesteps >= 2:
            flat = visibility_mask.transpose(1, 2).reshape(batch_size * variables, timesteps)
            all_hidden = ~flat.any(dim=1)
            all_visible = flat.all(dim=1)
            flat[all_hidden, 0] = True
            flat[all_visible, -1] = False
            visibility_mask = flat.reshape(batch_size, variables, timesteps).transpose(1, 2).contiguous()

        code = torch.randn(d_model, generator=code_generator, dtype=torch.float32)
        code = code / code.norm().clamp_min(1e-8)
        signs = targets.to(torch.float32).mul(2.0).sub(1.0)
        signal = signs[:, None, :, None] * code[None, None, None, :]
        noise = noise_std * torch.randn(
            batch_size,
            timesteps,
            variables,
            d_model,
            generator=observation_generator,
            dtype=torch.float32,
        )
        observations = noise + visibility_mask[..., None].to(torch.float32) * signal

        metadata: dict[str, Any] = {
            "scope": "synthetic-exp282-partial-observability-development",
            "root_seed": self.root_seed,
            "replicate": replicate,
            "rng_stream": rng_stream,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "d_model": d_model,
            "visibility_rate": float(visibility_rate),
            "noise_std": float(noise_std),
            "latent_seed": latent_seed,
            "observation_seed": observation_seed,
            "code_seed": code_seed,
        }
        observations = observations.to(device)
        targets = targets.to(device)
        visibility_mask = visibility_mask.to(device)
        return Exp282PartialObservabilityBatch(
            observations=observations,
            targets=targets,
            visibility_mask=visibility_mask,
            metadata=metadata,
            digest=_batch_digest(
                metadata,
                observations=observations,
                targets=targets,
                visibility_mask=visibility_mask,
            ),
        )
