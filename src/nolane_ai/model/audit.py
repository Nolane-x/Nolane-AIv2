from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class RegionAudit:
    total_parameters: int
    functional_parameters: int
    reserved_parameters: int
    trainable_parameters: int


@dataclass(frozen=True, slots=True)
class ModelAudit:
    total_parameters: int
    functional_parameters: int
    reserved_parameters: int
    trainable_parameters: int
    regions: dict[str, RegionAudit]


def _module_counts(module: nn.Module) -> RegionAudit:
    total = sum(p.numel() for p in module.parameters())
    reserve = 0
    if hasattr(module, "capacity_reserve") and module.capacity_reserve is not None:
        reserve = int(module.capacity_reserve.numel())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return RegionAudit(total, total - reserve, reserve, trainable)


def audit_model(model: nn.Module) -> ModelAudit:
    regions: dict[str, RegionAudit] = {}
    # Language binding is split between token embedding and its budgeted adapter.
    if hasattr(model, "token_embedding") and hasattr(model, "language_adapter"):
        embedding = sum(p.numel() for p in model.token_embedding.parameters())
        adapter = _module_counts(model.language_adapter)
        regions["language_perception_binding"] = RegionAudit(
            total_parameters=embedding + adapter.total_parameters,
            functional_parameters=embedding + adapter.functional_parameters,
            reserved_parameters=adapter.reserved_parameters,
            trainable_parameters=sum(p.numel() for p in model.token_embedding.parameters() if p.requires_grad)
            + adapter.trainable_parameters,
        )
    if hasattr(model, "regions"):
        for name, region in model.regions.items():
            regions[name] = _module_counts(region)
    total = sum(p.numel() for p in model.parameters())
    reserved = sum(r.reserved_parameters for r in regions.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelAudit(
        total_parameters=total,
        functional_parameters=total - reserved,
        reserved_parameters=reserved,
        trainable_parameters=trainable,
        regions=regions,
    )
