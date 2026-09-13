from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class Exp290CorrespondenceOutput:
    source_embeddings: torch.Tensor
    target_embeddings: torch.Tensor
    source_to_target_similarity: torch.Tensor
    target_to_source_similarity: torch.Tensor


class Exp290CorrespondenceModel(nn.Module):
    """Shared source/target structural correspondence encoder for EXP-290."""

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        target_parameters: int,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-290 dimensions and target_parameters must be positive")
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)

        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.context_gru = nn.GRU(
            hidden_size,
            hidden_size,
            batch_first=True,
            device=device,
        )
        self.embedding_projection = nn.Linear(
            hidden_size,
            hidden_size,
            device=device,
        )
        self.context_gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(
            self,
            target_parameters,
            device=device,
            frozen=False,
        )

    def _encode(
        self,
        surface_events: torch.Tensor,
        variable_states: torch.Tensor,
    ) -> torch.Tensor:
        if surface_events.ndim != 3 or surface_events.shape[-1] != self.d_model:
            raise ValueError("surface_events must be [batch,time,d_model]")
        if variable_states.ndim != 3 or variable_states.shape[-1] != self.d_model:
            raise ValueError("variable_states must be [batch,variables,d_model]")
        if surface_events.shape[0] != variable_states.shape[0]:
            raise ValueError("surface_events and variable_states batch dimensions must match")
        if surface_events.shape[1] <= 0 or variable_states.shape[1] <= 0:
            raise ValueError("EXP-290 encoder inputs must be non-empty")

        events = F.silu(self.event_projection(surface_events))
        _, hidden = self.context_gru(events)
        context = hidden[-1].unsqueeze(1)
        variables = F.silu(self.variable_projection(variable_states))
        mixed = variables + torch.sigmoid(self.context_gate) * context
        embeddings = self.embedding_projection(F.silu(mixed))
        return F.normalize(embeddings, p=2.0, dim=-1, eps=1e-8)

    def forward(
        self,
        source_surface_events: torch.Tensor,
        target_surface_events: torch.Tensor,
        source_variable_states: torch.Tensor,
        target_variable_states: torch.Tensor,
    ) -> Exp290CorrespondenceOutput:
        source_embeddings = self._encode(
            source_surface_events,
            source_variable_states,
        )
        target_embeddings = self._encode(
            target_surface_events,
            target_variable_states,
        )
        source_to_target = torch.bmm(
            source_embeddings,
            target_embeddings.transpose(1, 2),
        )
        target_to_source = source_to_target.transpose(1, 2).contiguous()
        return Exp290CorrespondenceOutput(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
            source_to_target_similarity=source_to_target,
            target_to_source_similarity=target_to_source,
        )


def learned_row_top1_mapping(similarity: torch.Tensor) -> torch.Tensor:
    if similarity.ndim != 3:
        raise ValueError("similarity must be [batch,source,target]")
    if similarity.shape[1] <= 0 or similarity.shape[2] <= 0:
        raise ValueError("similarity matrix must be non-empty")
    if not torch.isfinite(similarity).all():
        raise ValueError("similarity must be finite")
    # torch.argmax returns the first maximal index, giving the frozen ascending
    # target-index tie break without a secondary oracle-assisted path.
    return torch.argmax(similarity, dim=-1)


def exp290_correspondence_loss(
    source_to_target_similarity: torch.Tensor,
    target_to_source_similarity: torch.Tensor,
    *,
    source_to_target_labels: torch.Tensor,
    target_to_source_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if source_to_target_similarity.ndim != 3 or target_to_source_similarity.ndim != 3:
        raise ValueError("EXP-290 similarity tensors must be rank 3")
    if source_to_target_labels.shape != source_to_target_similarity.shape[:2]:
        raise ValueError("source_to_target_labels shape mismatch")
    if target_to_source_labels.shape != target_to_source_similarity.shape[:2]:
        raise ValueError("target_to_source_labels shape mismatch")

    source_loss = F.cross_entropy(
        source_to_target_similarity.reshape(-1, source_to_target_similarity.shape[-1]),
        source_to_target_labels.reshape(-1),
    )
    target_loss = F.cross_entropy(
        target_to_source_similarity.reshape(-1, target_to_source_similarity.shape[-1]),
        target_to_source_labels.reshape(-1),
    )
    loss = 0.5 * source_loss + 0.5 * target_loss
    audit: dict[str, Any] = {
        "loss_weights": {
            "source_to_target_ce": 0.5,
            "target_to_source_ce": 0.5,
        },
        "calibration_used": False,
        "early_stopping": False,
        "hard_negative_mining": False,
    }
    return loss, audit


def training_contract_receipt() -> dict[str, Any]:
    return {
        "rng_stream": "augmentation",
        "evaluation_lineage_consumed": False,
        "evaluation_correspondence_used_for_training": False,
        "loss_weights": {
            "source_to_target_ce": 0.5,
            "target_to_source_ce": 0.5,
        },
        "calibration_used": False,
        "early_stopping": False,
        "hard_negative_mining": False,
    }


def exp290_model_state_digest(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "byteorder": tensor_byteorder(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(header)
        hasher.update(b"\0")
        hasher.update(tensor_raw_bytes(cpu))
    return hasher.hexdigest()
