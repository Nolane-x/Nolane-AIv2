from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from nolane_ai.protocol.seeds import derive_stream_seed
from .stage_a import StageAMultitaskBatch
from .tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class CurriculumBatch:
    batch: StageAMultitaskBatch
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


def _batch_digest(metadata: dict[str, Any], batch: StageAMultitaskBatch) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in (
        "variable_states",
        "incidence",
        "belief_targets",
        "conflict_targets",
        "source_semantics",
        "candidate_semantics",
        "fidelity_targets",
    ):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest_part(getattr(batch, name)))
    return hasher.hexdigest()


class StageACurriculum:
    """Deterministic synthetic development curriculum for Stage-A neural paths.

    This generator is deliberately scoped to optimizer/path verification. Its
    outputs are development data and cannot themselves establish EV-E3 evidence.
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
        variables: int,
        constraints: int,
        d_model: int,
        device: str | torch.device = "cpu",
    ) -> CurriculumBatch:
        if replicate < 0:
            raise ValueError("replicate must be non-negative")
        if min(batch_size, variables, constraints, d_model) <= 0:
            raise ValueError("batch_size, variables, constraints and d_model must be positive")

        reasoning_seed = derive_stream_seed(self.root_seed, "EXP-277", replicate, "augmentation")
        belief_seed = derive_stream_seed(self.root_seed, "EXP-282", replicate, "augmentation")
        fidelity_seed = derive_stream_seed(self.root_seed, "EXP-297", replicate, "augmentation")

        reasoning_generator = torch.Generator(device="cpu").manual_seed(reasoning_seed)
        belief_generator = torch.Generator(device="cpu").manual_seed(belief_seed)
        fidelity_generator = torch.Generator(device="cpu").manual_seed(fidelity_seed)

        belief_targets = torch.randint(
            0,
            2,
            (batch_size, variables),
            generator=belief_generator,
            dtype=torch.long,
        )
        variable_states = 0.25 * torch.randn(
            batch_size,
            variables,
            d_model,
            generator=reasoning_generator,
        )
        variable_states[..., 0] += belief_targets.to(torch.float32).mul(2.0).sub(1.0)

        incidence = (
            torch.rand(batch_size, constraints, variables, generator=reasoning_generator) < 0.45
        ).to(torch.float32)
        for batch_index in range(batch_size):
            for constraint_index in range(constraints):
                if not incidence[batch_index, constraint_index].any():
                    variable_index = int(
                        torch.randint(0, variables, (1,), generator=reasoning_generator).item()
                    )
                    incidence[batch_index, constraint_index, variable_index] = 1.0

        selected_true = torch.bmm(
            incidence,
            belief_targets.to(torch.float32).unsqueeze(-1),
        ).squeeze(-1)
        conflict_targets = (selected_true.remainder(2.0) > 0.5).to(torch.float32)

        source_semantics = torch.randn(batch_size, d_model, generator=fidelity_generator)
        fidelity_targets = torch.randint(
            0,
            2,
            (batch_size,),
            generator=fidelity_generator,
            dtype=torch.long,
        ).to(torch.float32)
        faithful_noise = 0.05 * torch.randn(batch_size, d_model, generator=fidelity_generator)
        wrong_noise = 0.15 * torch.randn(batch_size, d_model, generator=fidelity_generator)
        faithful_candidate = source_semantics + faithful_noise
        wrong_candidate = -source_semantics + wrong_noise
        candidate_semantics = torch.where(
            fidelity_targets.unsqueeze(-1).bool(),
            faithful_candidate,
            wrong_candidate,
        )

        batch = StageAMultitaskBatch(
            variable_states=variable_states.to(device),
            incidence=incidence.to(device),
            belief_targets=belief_targets.to(device),
            conflict_targets=conflict_targets.to(device),
            source_semantics=source_semantics.to(device),
            candidate_semantics=candidate_semantics.to(device),
            fidelity_targets=fidelity_targets.to(device),
        )
        metadata: dict[str, Any] = {
            "scope": "synthetic-stage-a-development-curriculum",
            "root_seed": self.root_seed,
            "replicate": replicate,
            "batch_size": batch_size,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "stream_seeds": {
                "reasoning": reasoning_seed,
                "belief": belief_seed,
                "fidelity": fidelity_seed,
            },
        }
        return CurriculumBatch(batch=batch, metadata=metadata, digest=_batch_digest(metadata, batch))
