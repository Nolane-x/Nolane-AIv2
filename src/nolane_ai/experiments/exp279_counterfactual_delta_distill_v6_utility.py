from __future__ import annotations

from typing import Any

import torch

from .exp279_counterfactual_delta_distill_v6_primitives import (
    average_precision,
    class_counts,
    roc_auc,
    score_outcome_digest,
)


def fold_direct_utility_metrics(
    scores: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    *,
    stop_accounted_flops_per_episode: int | float,
    branch_accounted_flops_per_episode: int | float,
    inference_flops_per_episode: int | float,
) -> dict[str, Any]:
    if not (scores.ndim == stop_exact.ndim == branch_exact.ndim == 1):
        raise ValueError("V6 fold inputs must be rank-1")
    if not (scores.shape == stop_exact.shape == branch_exact.shape):
        raise ValueError("V6 fold inputs must align")
    stop_cost = float(stop_accounted_flops_per_episode)
    branch_cost = float(branch_accounted_flops_per_episode)
    inference_cost = float(inference_flops_per_episode)
    if stop_cost <= 0 or branch_cost <= stop_cost or inference_cost < 0:
        raise ValueError("invalid V6 cost ledger")

    scores64 = scores.detach().cpu().to(torch.float64)
    stop = stop_exact.detach().cpu().to(torch.bool)
    branch = branch_exact.detach().cpu().to(torch.bool)
    rescue = (~stop) & branch
    labels = rescue.to(torch.long)
    counts = class_counts(labels)
    support_closed = counts["positive"] > 0 and counts["negative"] > 0
    route_k = counts["positive"]
    selected = torch.zeros(scores64.numel(), dtype=torch.bool)
    if route_k:
        selected[torch.argsort(scores64, descending=True, stable=True)[:route_k]] = True

    harms = stop & (~branch)
    selected_rescues = int((selected & rescue).sum().item())
    selected_harms = int((selected & harms).sum().item())
    selected_neutral = int(selected.sum().item()) - selected_rescues - selected_harms
    routed_exact = torch.where(selected, branch, stop)
    episode_count = int(scores64.numel())
    stop_solutions = int(stop.sum().item())
    routed_solutions = int(routed_exact.sum().item())
    if routed_solutions - stop_solutions != selected_rescues - selected_harms:
        raise RuntimeError("V6 rescue/harm accounting mismatch")

    stop_flops = stop_cost * episode_count
    routed_flops = (
        (episode_count - route_k) * (stop_cost + inference_cost)
        + route_k * (branch_cost + inference_cost)
    )
    stop_utility = stop_solutions / stop_flops
    routed_utility = routed_solutions / routed_flops
    relative_gain = (routed_utility - stop_utility) / abs(stop_utility) if stop_utility > 0 else None
    return {
        "episode_count": episode_count,
        "rescue_count": counts["positive"],
        "non_rescue_count": counts["negative"],
        "support_closed": support_closed,
        "route_k": route_k,
        "route_fraction": route_k / episode_count,
        "selected_rescues": selected_rescues,
        "selected_harms": selected_harms,
        "selected_neutral": selected_neutral,
        "selected_rescue_precision": selected_rescues / route_k if route_k else None,
        "stop_solution_count": stop_solutions,
        "routed_solution_count": routed_solutions,
        "solution_count_delta": routed_solutions - stop_solutions,
        "stop_total_accounted_flops": stop_flops,
        "cdd_routed_total_accounted_flops": routed_flops,
        "stop_verified_utility": stop_utility,
        "cdd_routed_verified_utility": routed_utility,
        "relative_verified_utility_gain": relative_gain,
        "direct_utility_improved": support_closed and routed_utility > stop_utility,
        "roc_auc": roc_auc(scores64, labels),
        "average_precision": average_precision(scores64, labels),
        "concordance_pair_count": counts["positive"] * counts["negative"],
        "heldout_score_outcome_digest": score_outcome_digest(scores64, stop, branch),
        "raw_scores_exported": False,
    }


def aggregate_fold_direct_utility_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V6 aggregate requires rows")
    support_closed = all(bool(row["support_closed"]) for row in rows)
    stop_solutions = sum(int(row["stop_solution_count"]) for row in rows)
    routed_solutions = sum(int(row["routed_solution_count"]) for row in rows)
    stop_flops = sum(float(row["stop_total_accounted_flops"]) for row in rows)
    routed_flops = sum(float(row["cdd_routed_total_accounted_flops"]) for row in rows)
    selected = sum(int(row["route_k"]) for row in rows)
    rescues = sum(int(row["selected_rescues"]) for row in rows)
    harms = sum(int(row["selected_harms"]) for row in rows)
    neutral = sum(int(row["selected_neutral"]) for row in rows)
    rescue_count = sum(int(row["rescue_count"]) for row in rows)
    episodes = sum(int(row["episode_count"]) for row in rows)
    if selected != rescue_count or routed_solutions - stop_solutions != rescues - harms:
        raise RuntimeError("V6 aggregate accounting mismatch")

    stop_utility = stop_solutions / stop_flops
    routed_utility = routed_solutions / routed_flops
    pairs = sum(int(row["concordance_pair_count"]) for row in rows)
    auc = None
    if support_closed and pairs:
        auc = sum(float(row["roc_auc"]) * int(row["concordance_pair_count"]) for row in rows) / pairs
    ap_weight = sum(int(row["rescue_count"]) for row in rows if row["average_precision"] is not None)
    ap = None
    if ap_weight:
        ap = sum(
            float(row["average_precision"]) * int(row["rescue_count"])
            for row in rows
            if row["average_precision"] is not None
        ) / ap_weight
    return {
        "folds": len(rows),
        "support_closed": support_closed,
        "episode_count": episodes,
        "rescue_count": rescue_count,
        "route_k_total": selected,
        "selected_rescues": rescues,
        "selected_harms": harms,
        "selected_neutral": neutral,
        "selected_rescue_precision": rescues / selected if selected else None,
        "stop_solution_count": stop_solutions,
        "routed_solution_count": routed_solutions,
        "solution_count_delta": routed_solutions - stop_solutions,
        "stop_total_accounted_flops": stop_flops,
        "cdd_routed_total_accounted_flops": routed_flops,
        "stop_verified_utility": stop_utility,
        "cdd_routed_verified_utility": routed_utility,
        "relative_verified_utility_gain": (routed_utility - stop_utility) / abs(stop_utility) if stop_utility > 0 else None,
        "direct_utility_improved": support_closed and routed_utility > stop_utility,
        "cross_fitted_roc_auc": auc,
        "cross_fitted_average_precision": ap,
        "concordance_pair_count": pairs,
        "raw_scores_compared_across_folds": False,
    }
