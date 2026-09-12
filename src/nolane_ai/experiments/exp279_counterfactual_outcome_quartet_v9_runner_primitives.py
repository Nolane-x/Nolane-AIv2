from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch.nn import functional as F

from .exp279_counterfactual_outcome_quartet_v9 import (
    CONTROL_FAMILY,
    STUDENT_GEOMETRY,
    STUDENT_OPTIMIZER,
    QuartetClass,
    RescueOnlyMLPControl,
    deterministic_student_seed,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_bytes(list(value.shape)))
        digest.update(b"\0")
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
        digest.update(b"\0")
    return digest.hexdigest()


def _cyclic_batch_indices(order: torch.Tensor, *, step: int, batch_size: int) -> torch.Tensor:
    total = int(order.numel())
    if total <= 0:
        raise ValueError("V9 control fit set must not be empty")
    start = (step * batch_size) % total
    end = start + batch_size
    if end <= total:
        return order[start:end]
    return torch.cat((order[start:], order[: end - total]))


def fit_rescue_control(
    features: torch.Tensor,
    quartet_labels: torch.Tensor,
    *,
    train_replicates: int,
    canonical_index: int,
    fit_roots: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[1] != STUDENT_GEOMETRY["input_size"]:
        raise ValueError("V9 control features must be [episodes,144]")
    if quartet_labels.ndim != 1 or quartet_labels.shape[0] != features.shape[0] or quartet_labels.numel() == 0:
        raise ValueError("V9 control quartet labels must align with a non-empty feature set")
    labels = quartet_labels.to(dtype=torch.long, device=features.device)
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= STUDENT_GEOMETRY["classes"]:
        raise ValueError("V9 control quartet labels must be in [0,3]")

    target = (labels == int(QuartetClass.RESCUE)).to(dtype=features.dtype)
    seed = deterministic_student_seed(CONTROL_FAMILY, train_replicates, canonical_index, fit_roots)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RescueOnlyMLPControl().to(device=features.device, dtype=features.dtype)
    initial_digest = _state_digest(model)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(int(features.shape[0]), generator=generator).to(device=features.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STUDENT_OPTIMIZER["lr"],
        weight_decay=STUDENT_OPTIMIZER["weight_decay"],
    )
    model.train()
    loss = torch.zeros((), dtype=features.dtype, device=features.device)
    for step in range(STUDENT_OPTIMIZER["steps"]):
        indices = _cyclic_batch_indices(order, step=step, batch_size=STUDENT_OPTIMIZER["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(features[indices]), target[indices])
        loss.backward()
        optimizer.step()

    positive_count = int(target.sum().item())
    total = int(target.numel())
    return {
        "model": model,
        "family": CONTROL_FAMILY,
        "seed": seed,
        "optimizer": dict(STUDENT_OPTIMIZER),
        "steps": STUDENT_OPTIMIZER["steps"],
        "batch_size": STUDENT_OPTIMIZER["batch_size"],
        "positive_count": positive_count,
        "negative_count": total - positive_count,
        "initial_digest": initial_digest,
        "final_digest": _state_digest(model),
        "loss": float(loss.detach().cpu().item()),
    }


def combine_decision_root_metrics(root_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(root_metrics) != 2:
        raise ValueError("V9 canonical-root aggregation requires exactly two decision roots")
    sum_keys = (
        "episodes",
        "routed_episodes",
        "raw_rescues",
        "raw_harms",
        "selected_rescues",
        "selected_harms",
        "stop_solutions",
        "branch_solutions",
        "policy_solutions",
        "stop_total_accounted_flops",
        "branch_total_accounted_flops",
        "policy_total_accounted_flops",
    )
    for root in root_metrics:
        missing = [key for key in sum_keys if key not in root]
        if missing:
            raise ValueError(f"V9 decision-root metrics missing keys: {missing}")

    pooled: dict[str, Any] = {key: sum(float(root[key]) for root in root_metrics) for key in sum_keys}
    count_keys = (
        "episodes",
        "routed_episodes",
        "raw_rescues",
        "raw_harms",
        "selected_rescues",
        "selected_harms",
        "stop_solutions",
        "branch_solutions",
        "policy_solutions",
    )
    for key in count_keys:
        value = pooled[key]
        if not float(value).is_integer():
            raise ValueError(f"V9 pooled count {key} is not integral")
        pooled[key] = int(value)

    episodes = int(pooled["episodes"])
    routed = int(pooled["routed_episodes"])
    stop_flops = float(pooled["stop_total_accounted_flops"])
    branch_flops = float(pooled["branch_total_accounted_flops"])
    policy_flops = float(pooled["policy_total_accounted_flops"])
    if episodes <= 0 or not (0 <= routed <= episodes):
        raise ValueError("V9 pooled decision counts are invalid")
    if min(stop_flops, branch_flops, policy_flops) <= 0.0:
        raise ValueError("V9 pooled decision FLOPs must be positive")

    pooled.update(
        {
            "route_fraction": routed / episodes,
            "raw_rescue_prevalence": int(pooled["raw_rescues"]) / episodes,
            "selected_rescue_prevalence": int(pooled["selected_rescues"]) / routed if routed else 0.0,
            "stop_baseline_utility": int(pooled["stop_solutions"]) / stop_flops,
            "branch_baseline_utility": int(pooled["branch_solutions"]) / branch_flops,
            "policy_utility": int(pooled["policy_solutions"]) / policy_flops,
            "evidence_boundary_closed": all(root.get("evidence_boundary_closed") is True for root in root_metrics),
        }
    )
    return pooled
