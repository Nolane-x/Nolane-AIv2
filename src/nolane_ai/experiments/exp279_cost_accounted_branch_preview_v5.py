from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer

from .exp279_paired_runner import (
    _build_seeded_triplet,
    _functional_state_digest,
    _hybrid_forced_branch_decision_logits,
    _hybrid_stop_decision_logits,
    _train_step,
)
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import HybridRoutingArm, audit_matched_exp279_arm_triplet


SCHEMA = "NLM-EXP-279-COST-ACCOUNTED-BRANCH-PREVIEW-COURT-V5"
PREVIEW_FAMILIES = (
    "PREFIX1_STATE_LINEAR",
    "PREFIX2_STATE_LINEAR",
    "PREFIX3_STATE_LINEAR",
)
PROBE_ROOT_SUFFIX = "::independent-augmentation-preview-v5"
_PREVIEW_DEPTH = {
    "PREFIX1_STATE_LINEAR": 1,
    "PREFIX2_STATE_LINEAR": 2,
    "PREFIX3_STATE_LINEAR": 3,
}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _assign_probe_fold(replicate: int, *, folds: int) -> int:
    if replicate < 0:
        raise ValueError("probe replicate must be non-negative")
    if folds <= 1:
        raise ValueError("probe folds must be greater than one")
    return (replicate // len(STRATA)) % folds


def _validate_rank_inputs(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("rank scores and labels must be aligned rank-1 tensors")
    if scores.numel() == 0:
        raise ValueError("rank metrics require at least one example")
    labels_long = labels.to(dtype=torch.long)
    if bool(((labels_long != 0) & (labels_long != 1)).any().item()):
        raise ValueError("rank labels must be binary")
    return scores.to(dtype=torch.float64), labels_long


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores64, labels_long = _validate_rank_inputs(scores, labels)
    positive = scores64[labels_long == 1]
    negative = scores64[labels_long == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    wins = (differences > 0).to(torch.float64).sum()
    ties = (differences == 0).to(torch.float64).sum()
    pairs = positive.numel() * negative.numel()
    return float(((wins + 0.5 * ties) / pairs).item())


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores64, labels_long = _validate_rank_inputs(scores, labels)
    positives = int((labels_long == 1).sum().item())
    if positives == 0:
        return None
    order = torch.argsort(scores64, descending=True, stable=True)
    ranked = labels_long.index_select(0, order)
    cumulative = torch.cumsum(ranked.to(torch.float64), dim=0)
    ranks = torch.arange(1, ranked.numel() + 1, dtype=torch.float64)
    precision = cumulative / ranks
    return float(precision[ranked == 1].sum().item() / positives)


def _pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("pairwise selector scores and labels must be aligned rank-1 tensors")
    positive = scores[labels > 0]
    negative = scores[labels <= 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.sum() * 0.0
    differences = positive[:, None] - negative[None, :]
    return F.softplus(-differences).mean()


def _class_counts(labels: torch.Tensor) -> dict[str, int]:
    labels_long = labels.to(dtype=torch.long)
    positive = int((labels_long == 1).sum().item())
    total = int(labels_long.numel())
    return {"positive": positive, "negative": total - positive, "total": total}


def _preview_hidden(arm: HybridRoutingArm, events: torch.Tensor, *, depth: int) -> torch.Tensor:
    if events.ndim != 3 or events.shape[-1] != arm.hidden_size:
        raise ValueError("preview events must be [batch,time,hidden_size]")
    if depth <= 0 or depth >= events.shape[1]:
        raise ValueError("V5 preview depth must be within [1, timesteps-1]")
    _, hidden = arm.branch_gru(events[:, :depth, :])
    return hidden


def _resume_branch_hidden(
    arm: HybridRoutingArm,
    events: torch.Tensor,
    *,
    depth: int,
    prefix_hidden: torch.Tensor,
) -> torch.Tensor:
    if events.ndim != 3 or events.shape[-1] != arm.hidden_size:
        raise ValueError("resume events must be [batch,time,hidden_size]")
    if depth <= 0 or depth >= events.shape[1]:
        raise ValueError("V5 resume depth must be within [1, timesteps-1]")
    if prefix_hidden.shape != (1, events.shape[0], arm.hidden_size):
        raise ValueError("prefix_hidden shape mismatch")
    _, hidden = arm.branch_gru(events[:, depth:, :], prefix_hidden)
    return hidden


def _selector_costs(*, hidden_size: int, variables: int, preview_depth: int) -> dict[str, int]:
    if min(hidden_size, variables, preview_depth) <= 0:
        raise ValueError("V5 selector cost geometry must be positive")
    preview_gru = preview_depth * (12 * hidden_size * hidden_size + 20 * hidden_size)
    p_mean = variables * hidden_size
    selector_head = 4 * hidden_size + 1
    return {
        "preview_depth": preview_depth,
        "preview_gru_flops": int(preview_gru),
        "p_mean_flops": int(p_mean),
        "selector_head_flops": int(selector_head),
        "selector_total_flops": int(p_mean + selector_head),
    }


def _fold_preview_direct_utility_metrics(
    scores: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    *,
    stop_accounted_flops_per_episode: int | float,
    branch_accounted_flops_per_episode: int | float,
    preview_gru_flops_per_unselected_episode: int | float,
    selector_flops_per_episode: int | float,
) -> dict[str, Any]:
    if scores.ndim != 1 or stop_exact.ndim != 1 or branch_exact.ndim != 1:
        raise ValueError("V5 fold scores/outcomes must be rank-1")
    if not (scores.shape == stop_exact.shape == branch_exact.shape):
        raise ValueError("V5 fold scores/outcomes must be aligned")
    if scores.numel() == 0:
        raise ValueError("V5 fold must contain at least one episode")

    stop_cost = float(stop_accounted_flops_per_episode)
    branch_cost = float(branch_accounted_flops_per_episode)
    preview_cost = float(preview_gru_flops_per_unselected_episode)
    selector_cost = float(selector_flops_per_episode)
    if stop_cost <= 0.0 or branch_cost <= stop_cost:
        raise ValueError("V5 direct utility requires 0 < stop FLOPs < branch FLOPs")
    if preview_cost < 0.0 or selector_cost < 0.0:
        raise ValueError("V5 preview/selector FLOPs cannot be negative")

    scores64 = scores.detach().cpu().to(torch.float64)
    stop = stop_exact.detach().cpu().to(torch.bool)
    branch = branch_exact.detach().cpu().to(torch.bool)
    rescue = (~stop) & branch
    labels = rescue.to(torch.long)
    counts = _class_counts(labels)
    support_closed = counts["positive"] > 0 and counts["negative"] > 0
    route_k = counts["positive"]

    selected = torch.zeros(scores64.numel(), dtype=torch.bool)
    if route_k > 0:
        order = torch.argsort(scores64, descending=True, stable=True)
        selected[order[:route_k]] = True

    harms = stop & (~branch)
    selected_rescues = int((selected & rescue).sum().item())
    selected_harms = int((selected & harms).sum().item())
    selected_neutral = int(selected.sum().item()) - selected_rescues - selected_harms
    routed_exact = torch.where(selected, branch, stop)

    episode_count = int(scores64.numel())
    stop_solution_count = int(stop.sum().item())
    routed_solution_count = int(routed_exact.sum().item())
    if routed_solution_count - stop_solution_count != selected_rescues - selected_harms:
        raise RuntimeError("V5 rescue/harm accounting does not match solution delta")

    unselected_count = episode_count - route_k
    stop_total_flops = stop_cost * episode_count
    preview_routed_total_flops = (
        unselected_count * (stop_cost + preview_cost + selector_cost)
        + route_k * (branch_cost + selector_cost)
    )
    stop_utility = stop_solution_count / stop_total_flops
    preview_utility = routed_solution_count / preview_routed_total_flops
    relative_gain = (
        (preview_utility - stop_utility) / abs(stop_utility)
        if stop_utility > 0.0
        else None
    )
    auc = _roc_auc(scores64, labels)
    ap = _average_precision(scores64, labels)
    receipt_digest = canonical_sha256(
        {
            "scores": scores64.tolist(),
            "stop_exact": stop.to(torch.long).tolist(),
            "branch_exact": branch.to(torch.long).tolist(),
        }
    )

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
        "selected_rescue_precision": selected_rescues / route_k if route_k > 0 else None,
        "stop_solution_count": stop_solution_count,
        "routed_solution_count": routed_solution_count,
        "solution_count_delta": routed_solution_count - stop_solution_count,
        "stop_total_accounted_flops": stop_total_flops,
        "preview_routed_total_accounted_flops": preview_routed_total_flops,
        "stop_verified_utility": stop_utility,
        "preview_routed_verified_utility": preview_utility,
        "relative_verified_utility_gain": relative_gain,
        "direct_utility_improved": support_closed and preview_utility > stop_utility,
        "roc_auc": auc,
        "average_precision": ap,
        "concordance_pair_count": counts["positive"] * counts["negative"],
        "heldout_score_outcome_digest": receipt_digest,
        "raw_scores_exported": False,
    }


def _aggregate_fold_preview_direct_utility_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V5 aggregate requires at least one fold")
    support_closed = all(bool(row.get("support_closed")) for row in rows)
    stop_solutions = sum(int(row["stop_solution_count"]) for row in rows)
    routed_solutions = sum(int(row["routed_solution_count"]) for row in rows)
    stop_flops = sum(float(row["stop_total_accounted_flops"]) for row in rows)
    preview_flops = sum(float(row["preview_routed_total_accounted_flops"]) for row in rows)
    if stop_flops <= 0.0 or preview_flops <= 0.0:
        raise ValueError("V5 aggregate FLOPs must be positive")

    selected = sum(int(row["route_k"]) for row in rows)
    rescues = sum(int(row["selected_rescues"]) for row in rows)
    harms = sum(int(row["selected_harms"]) for row in rows)
    neutral = sum(int(row["selected_neutral"]) for row in rows)
    rescue_count = sum(int(row["rescue_count"]) for row in rows)
    episode_count = sum(int(row["episode_count"]) for row in rows)
    if selected != rescue_count:
        raise RuntimeError("V5 aggregate oracle route cardinality drift")
    if routed_solutions - stop_solutions != rescues - harms:
        raise RuntimeError("V5 aggregate solution delta does not match rescue/harm accounting")

    stop_utility = stop_solutions / stop_flops
    preview_utility = routed_solutions / preview_flops
    relative_gain = (
        (preview_utility - stop_utility) / abs(stop_utility)
        if stop_utility > 0.0
        else None
    )
    total_pairs = sum(int(row["concordance_pair_count"]) for row in rows)
    auc = None
    if support_closed and total_pairs > 0:
        auc = sum(
            float(row["roc_auc"]) * int(row["concordance_pair_count"])
            for row in rows
        ) / total_pairs
    ap_weight = sum(int(row["rescue_count"]) for row in rows if row.get("average_precision") is not None)
    ap = None
    if ap_weight > 0:
        ap = sum(
            float(row["average_precision"]) * int(row["rescue_count"])
            for row in rows
            if row.get("average_precision") is not None
        ) / ap_weight

    return {
        "folds": len(rows),
        "support_closed": support_closed,
        "episode_count": episode_count,
        "rescue_count": rescue_count,
        "route_k_total": selected,
        "selected_rescues": rescues,
        "selected_harms": harms,
        "selected_neutral": neutral,
        "selected_rescue_precision": rescues / selected if selected > 0 else None,
        "stop_solution_count": stop_solutions,
        "routed_solution_count": routed_solutions,
        "solution_count_delta": routed_solutions - stop_solutions,
        "stop_total_accounted_flops": stop_flops,
        "preview_routed_total_accounted_flops": preview_flops,
        "stop_verified_utility": stop_utility,
        "preview_routed_verified_utility": preview_utility,
        "relative_verified_utility_gain": relative_gain,
        "direct_utility_improved": support_closed and preview_utility > stop_utility,
        "cross_fitted_roc_auc": auc,
        "cross_fitted_average_precision": ap,
        "concordance_pair_count": total_pairs,
        "raw_scores_compared_across_folds": False,
    }


class _PreviewLinearSelector(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.readout = nn.Linear(2 * hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(x).squeeze(-1)


def _build_selector(*, hidden_size: int, seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _PreviewLinearSelector(hidden_size)


def _fit_preview_fold(
    *,
    kind: str,
    selector_inputs: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
    folds: int,
    hidden_size: int,
    probe_steps: int,
    probe_lr: float,
    probe_root_seed: str,
    stop_cost: float,
    branch_cost: float,
    preview_cost: float,
    selector_cost: float,
) -> dict[str, Any]:
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold
    train_x = selector_inputs[train_mask]
    train_stop = stop_exact[train_mask]
    train_branch = branch_exact[train_mask]
    train_labels = ((~train_stop) & train_branch).to(torch.long)
    train_counts = _class_counts(train_labels)
    train_supported = train_counts["positive"] > 0 and train_counts["negative"] > 0

    seed = derive_stream_seed(
        probe_root_seed,
        f"EXP-279-BRANCH-PREVIEW-V5-{kind}",
        fold,
        "model_init",
    )
    selector = _build_selector(hidden_size=hidden_size, seed=seed)
    optimizer = torch.optim.Adam(selector.parameters(), lr=probe_lr, weight_decay=0.0)
    initial_loss: float | None = None
    final_loss: float | None = None
    if train_supported:
        selector.train()
        for step in range(probe_steps):
            optimizer.zero_grad(set_to_none=True)
            scores = selector(train_x)
            loss = _pairwise_ranking_loss(scores, train_labels)
            if step == 0:
                initial_loss = float(loss.detach().item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    selector.eval()
    with torch.no_grad():
        test_scores = selector(selector_inputs[test_mask])
    metrics = _fold_preview_direct_utility_metrics(
        test_scores,
        stop_exact[test_mask],
        branch_exact[test_mask],
        stop_accounted_flops_per_episode=stop_cost,
        branch_accounted_flops_per_episode=branch_cost,
        preview_gru_flops_per_unselected_episode=preview_cost,
        selector_flops_per_episode=selector_cost,
    )
    test_supported = bool(metrics["support_closed"])
    metrics["support_closed"] = train_supported and test_supported
    metrics["direct_utility_improved"] = bool(
        metrics["support_closed"]
        and float(metrics["preview_routed_verified_utility"]) > float(metrics["stop_verified_utility"])
    )
    return {
        "fold": fold,
        "folds": folds,
        "seed": seed,
        "train_counts": train_counts,
        "train_supported": train_supported,
        "test_supported": test_supported,
        "initial_pairwise_loss": initial_loss,
        "final_pairwise_loss": final_loss,
        **metrics,
    }


def _cross_fitted_preview(
    *,
    kind: str,
    selector_inputs: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    fold_ids: torch.Tensor,
    folds: int,
    hidden_size: int,
    variables: int,
    probe_steps: int,
    probe_lr: float,
    probe_root_seed: str,
    stop_cost: float,
    branch_cost: float,
) -> dict[str, Any]:
    depth = _PREVIEW_DEPTH[kind]
    costs = _selector_costs(hidden_size=hidden_size, variables=variables, preview_depth=depth)
    rows = [
        _fit_preview_fold(
            kind=kind,
            selector_inputs=selector_inputs,
            stop_exact=stop_exact,
            branch_exact=branch_exact,
            fold_ids=fold_ids,
            fold=fold,
            folds=folds,
            hidden_size=hidden_size,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_root_seed=probe_root_seed,
            stop_cost=stop_cost,
            branch_cost=branch_cost,
            preview_cost=float(costs["preview_gru_flops"]),
            selector_cost=float(costs["selector_total_flops"]),
        )
        for fold in range(folds)
    ]
    aggregate = _aggregate_fold_preview_direct_utility_metrics(rows)
    seed = derive_stream_seed(
        probe_root_seed,
        f"EXP-279-BRANCH-PREVIEW-V5-{kind}",
        0,
        "model_init",
    )
    parameter_count = sum(p.numel() for p in _build_selector(hidden_size=hidden_size, seed=seed).parameters())
    return {
        "kind": kind,
        "preview_depth": depth,
        "selector_parameter_count": parameter_count,
        "costs": costs,
        "folds": rows,
        "aggregate": aggregate,
        "economically_routable": bool(aggregate["direct_utility_improved"]),
        "raw_scores_compared_across_folds": False,
    }


def _court_classification(probes: dict[str, dict[str, Any]]) -> str:
    if not all(bool(probe["aggregate"]["support_closed"]) for probe in probes.values()):
        return "INSUFFICIENT_RESCUE_SUPPORT"
    if bool(probes["PREFIX1_STATE_LINEAR"]["economically_routable"]):
        return "PREFIX1_DIRECT_UTILITY_POSITIVE"
    if bool(probes["PREFIX2_STATE_LINEAR"]["economically_routable"]):
        return "PREFIX2_DIRECT_UTILITY_POSITIVE"
    if bool(probes["PREFIX3_STATE_LINEAR"]["economically_routable"]):
        return "PREFIX3_DIRECT_UTILITY_POSITIVE"
    return "NO_DIRECTLY_ECONOMIC_BRANCH_PREVIEW"


def validate_exp279_cost_accounted_branch_preview_v5(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("V5 schema mismatch")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("V5 evidence boundary mismatch")
    if payload.get("scientific_evidence_eligible") is not False:
        errors.append("V5 cannot be scientific-evidence eligible")
    for key in (
        "fresh_evaluation_lineage_may_be_reserved",
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        if payload.get(key) is not False:
            errors.append(f"V5 {key} must remain false")

    boundary = payload.get("data_boundary") or {}
    if boundary.get("training_rng_stream") != "augmentation" or boundary.get("probe_rng_stream") != "augmentation":
        errors.append("V5 streams must remain augmentation-only")
    if boundary.get("evaluation_rng_stream_used") is not False or boundary.get("evaluation_targets_used") is not False:
        errors.append("V5 evaluation boundary violated")
    if boundary.get("probe_root_seed") == payload.get("root_seed"):
        errors.append("V5 probe root must be independent")

    semantics = payload.get("preview_semantics") or {}
    if semantics.get("branch_gru_used_for_preview") is not True:
        errors.append("V5 must use canonical branch GRU preview")
    if semantics.get("preview_hidden_reused_for_selected_branch") is not True:
        errors.append("V5 preview hidden must be reusable")
    if semantics.get("resume_equivalence_closed") is not True:
        errors.append("V5 resume equivalence must close")

    rule = payload.get("preview_cost_rule") or {}
    if rule.get("preview_flops_charged_to_primary_utility") is not True:
        errors.append("V5 preview FLOPs must be charged")
    if rule.get("selector_flops_charged_to_primary_utility") is not True:
        errors.append("V5 selector FLOPs must be charged")
    if rule.get("selected_preview_double_counted") is not False:
        errors.append("V5 selected preview cannot be double-counted")
    if rule.get("metric") != "direct_counterfactual_verified_solution_per_total_accounted_flop":
        errors.append("V5 primary metric changed")

    config = payload.get("probe_config") or {}
    if config.get("preview_families") != list(PREVIEW_FAMILIES):
        errors.append("V5 preview families changed")
    if config.get("folds") != 3:
        errors.append("V5 court is frozen to three folds")
    if config.get("fold_route_k_rule") != "heldout_fold_true_rescue_count":
        errors.append("V5 oracle route cardinality changed")
    if config.get("oracle_route_cardinality_is_deployable") is not False:
        errors.append("V5 oracle cardinality cannot be deployable")
    if config.get("raw_scores_compared_across_folds") is not False or config.get("raw_scores_exported") is not False:
        errors.append("V5 raw-score fold boundary violated")

    probes = payload.get("probes") or {}
    if set(probes) != set(PREVIEW_FAMILIES):
        errors.append("V5 preview receipts incomplete")
    else:
        for kind, probe in probes.items():
            aggregate = probe.get("aggregate") or {}
            expected = bool(
                aggregate.get("support_closed")
                and float(aggregate.get("preview_routed_verified_utility", 0.0))
                > float(aggregate.get("stop_verified_utility", 0.0))
            )
            if probe.get("economically_routable") is not expected:
                errors.append(f"V5 {kind} economic receipt mismatch")
            if probe.get("raw_scores_compared_across_folds") is not False:
                errors.append(f"V5 {kind} compared raw scores across folds")
            for fold in probe.get("folds") or []:
                if fold.get("raw_scores_exported") is not False:
                    errors.append(f"V5 {kind} exported raw scores")

    allowed = {
        "INSUFFICIENT_RESCUE_SUPPORT",
        "PREFIX1_DIRECT_UTILITY_POSITIVE",
        "PREFIX2_DIRECT_UTILITY_POSITIVE",
        "PREFIX3_DIRECT_UTILITY_POSITIVE",
        "NO_DIRECTLY_ECONOMIC_BRANCH_PREVIEW",
    }
    classification = payload.get("court_classification")
    if classification not in allowed:
        errors.append("V5 classification invalid")
    elif set(probes) == set(PREVIEW_FAMILIES) and classification != _court_classification(probes):
        errors.append("V5 classification receipt mismatch")
    return errors


def run_exp279_cost_accounted_branch_preview_v5_court(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
    train_replicates: int,
    probe_replicates: int,
    probe_folds: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    probe_steps: int,
    probe_lr: float,
    protocol_digest: str,
    code_digest: str,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    if not root_seed or not protocol_digest or not code_digest:
        raise ValueError("V5 root_seed, protocol_digest and code_digest are required")
    if min(
        d_model,
        hidden_size,
        target_parameters,
        train_replicates,
        probe_replicates,
        probe_folds,
        batch_size,
        timesteps,
        variables,
        constraints,
        probe_steps,
    ) <= 0:
        raise ValueError("V5 counts and dimensions must be positive")
    if timesteps != 4:
        raise ValueError("V5 court is frozen to four timesteps")
    if probe_folds != 3:
        raise ValueError("V5 court is frozen to three folds")
    if probe_replicates % (len(STRATA) * probe_folds) != 0:
        raise ValueError("V5 probe_replicates must balance strata across folds")
    if constraints > variables:
        raise ValueError("V5 constraints cannot exceed variables")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0,1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0 or probe_lr <= 0.0:
        raise ValueError("V5 optimizer/noise parameters invalid")

    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=route_threshold,
    )
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
        max_accounted_flops_per_episode=max_accounted_flops_per_episode,
    )
    required_audit = (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "compute_budget_closed",
    )
    if not all(pair_audit.get(key) is True for key in required_audit):
        raise RuntimeError("V5 matched resource contract did not close")

    generator = Exp279RoutingGenerator(root_seed=root_seed)
    optimizer = build_functional_optimizer(hybrid, lr=lr, weight_decay=weight_decay)
    training_batch_digests: list[str] = []
    training_losses: list[float] = []
    for replicate in range(train_replicates):
        stratum = STRATA[replicate % len(STRATA)]
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            variables=variables,
            constraints=constraints,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="augmentation",
            stratum=stratum,
            scope="synthetic-exp279-routing-development-training",
        )
        training_batch_digests.append(batch.digest)
        training_losses.append(
            _train_step(
                hybrid,
                optimizer,
                arm_id="hybrid",
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
                targets=batch.targets,
            )
        )

    frozen_hybrid_digest = _functional_state_digest(hybrid)
    for parameter in hybrid.parameters():
        parameter.requires_grad_(False)
    hybrid.eval()

    probe_root_seed = root_seed + PROBE_ROOT_SUFFIX
    probe_generator = Exp279RoutingGenerator(root_seed=probe_root_seed)
    p_mean_rows: list[torch.Tensor] = []
    preview_rows: dict[str, list[torch.Tensor]] = {kind: [] for kind in PREVIEW_FAMILIES}
    stop_rows: list[torch.Tensor] = []
    branch_rows: list[torch.Tensor] = []
    fold_rows: list[torch.Tensor] = []
    probe_batch_digests: list[str] = []
    probe_strata: list[str] = []
    resume_max_abs_error = {kind: 0.0 for kind in PREVIEW_FAMILIES}

    with torch.no_grad():
        for replicate in range(probe_replicates):
            stratum = STRATA[replicate % len(STRATA)]
            batch = probe_generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                constraints=constraints,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="augmentation",
                stratum=stratum,
                scope="synthetic-exp279-routing-development-cost-accounted-preview-v5-probe",
            )
            events, projected_variables = hybrid._validate_common(batch.surface_events, batch.variable_states)
            checked_incidence = hybrid._validate_incidence(batch.incidence, projected_variables)
            propagation_state = projected_variables + hybrid._propagate(projected_variables, checked_incidence)
            p_mean = propagation_state.mean(dim=1)
            _, full_hidden = hybrid.branch_gru(events)

            for kind in PREVIEW_FAMILIES:
                depth = _PREVIEW_DEPTH[kind]
                prefix = _preview_hidden(hybrid, events, depth=depth)
                resumed = _resume_branch_hidden(
                    hybrid,
                    events,
                    depth=depth,
                    prefix_hidden=prefix,
                )
                max_error = float((full_hidden - resumed).abs().max().item())
                resume_max_abs_error[kind] = max(resume_max_abs_error[kind], max_error)
                if not torch.allclose(full_hidden, resumed, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(f"V5 branch preview resume equivalence failed for {kind}")
                preview_rows[kind].append(prefix[-1].detach().cpu())

            stop_logits = _hybrid_stop_decision_logits(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            branch_logits = _hybrid_forced_branch_decision_logits(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            stop_exact = (stop_logits.argmax(dim=-1) == batch.targets).all(dim=-1)
            branch_exact = (branch_logits.argmax(dim=-1) == batch.targets).all(dim=-1)
            fold = _assign_probe_fold(replicate, folds=probe_folds)

            p_mean_rows.append(p_mean.detach().cpu())
            stop_rows.append(stop_exact.detach().cpu())
            branch_rows.append(branch_exact.detach().cpu())
            fold_rows.append(torch.full((batch_size,), fold, dtype=torch.long))
            probe_batch_digests.append(batch.digest)
            probe_strata.append(stratum)

    p_mean_all = torch.cat(p_mean_rows, dim=0)
    stop_exact_all = torch.cat(stop_rows, dim=0).to(torch.bool)
    branch_exact_all = torch.cat(branch_rows, dim=0).to(torch.bool)
    fold_ids = torch.cat(fold_rows, dim=0)
    rescue_labels = ((~stop_exact_all) & branch_exact_all).to(torch.long)
    pooled_counts = _class_counts(rescue_labels)

    hybrid_ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_cost = float(hybrid_ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(hybrid_ledger["branch_accounted_flops_per_episode"])
    probes: dict[str, dict[str, Any]] = {}
    for kind in PREVIEW_FAMILIES:
        preview_all = torch.cat(preview_rows[kind], dim=0)
        selector_inputs = torch.cat((p_mean_all, preview_all), dim=-1)
        probes[kind] = _cross_fitted_preview(
            kind=kind,
            selector_inputs=selector_inputs,
            stop_exact=stop_exact_all,
            branch_exact=branch_exact_all,
            fold_ids=fold_ids,
            folds=probe_folds,
            hidden_size=hidden_size,
            variables=variables,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_root_seed=probe_root_seed,
            stop_cost=stop_cost,
            branch_cost=branch_cost,
        )

    classification = _court_classification(probes)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "experiment_id": "EXP-279",
        "analysis_scope": "augmentation_only_frozen_canonical_cost_accounted_reusable_branch_preview",
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "data_boundary": {
            "training_rng_stream": "augmentation",
            "probe_rng_stream": "augmentation",
            "probe_root_seed": probe_root_seed,
            "probe_root_independent_from_training_root": True,
            "evaluation_rng_stream_used": False,
            "evaluation_targets_used": False,
            "confirmatory_examples_used": False,
            "external_examples_used": False,
        },
        "world_geometry": {
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "noise_std": float(noise_std),
        },
        "arm_geometry": {"hidden_size": hidden_size, "target_parameters": target_parameters},
        "route_config": {
            "canonical_threshold": float(route_threshold),
            "threshold_used_by_v5_oracle_selector": False,
            "threshold_tuned": False,
        },
        "canonical_training": {
            "replicates": train_replicates,
            "rng_stream": "augmentation",
            "batch_digests": training_batch_digests,
            "losses": training_losses,
            "frozen_hybrid_functional_state_digest": frozen_hybrid_digest,
            "canonical_parameters_frozen_before_probe_fit": True,
        },
        "preview_semantics": {
            "branch_gru_used_for_preview": True,
            "preview_hidden_reused_for_selected_branch": True,
            "resume_equivalence_closed": True,
            "resume_atol": 1e-6,
            "resume_rtol": 1e-6,
            "max_abs_error_by_family": resume_max_abs_error,
        },
        "probe_config": {
            "replicates": probe_replicates,
            "folds": probe_folds,
            "fold_assignment": "floor(replicate / number_of_strata) mod folds",
            "preview_families": list(PREVIEW_FAMILIES),
            "preview_depths": {kind: _PREVIEW_DEPTH[kind] for kind in PREVIEW_FAMILIES},
            "selector_architecture": "Linear(2H,1) over concat(p_mean,preview_hidden)",
            "pairwise_ranking_loss": "softplus(-(positive_score-negative_score))",
            "steps": probe_steps,
            "lr": float(probe_lr),
            "weight_decay": 0.0,
            "fold_route_k_rule": "heldout_fold_true_rescue_count",
            "oracle_route_cardinality_is_deployable": False,
            "heldout_targets_used_for_route_cardinality": True,
            "successor_must_replace_oracle_cardinality": True,
            "raw_scores_compared_across_folds": False,
            "raw_scores_exported": False,
            "decision_threshold_tuned": False,
        },
        "probe_dataset": {
            "batch_digests": probe_batch_digests,
            "strata": probe_strata,
            "episode_counts": pooled_counts,
            "natural_rescue_prevalence": (
                pooled_counts["positive"] / pooled_counts["total"] if pooled_counts["total"] else 0.0
            ),
            "fold_episode_counts": {
                str(fold): _class_counts(rescue_labels[fold_ids == fold]) for fold in range(probe_folds)
            },
        },
        "compute_economics": {
            "stop_accounted_flops_per_episode": int(stop_cost),
            "branch_accounted_flops_per_episode": int(branch_cost),
            "source": "sealed_matched_arm_compute_ledger",
            "family_costs": {kind: probes[kind]["costs"] for kind in PREVIEW_FAMILIES},
        },
        "preview_cost_rule": {
            "metric": "direct_counterfactual_verified_solution_per_total_accounted_flop",
            "cross_fold_aggregation": "sum_solution_counts_and_total_accounted_flops",
            "preview_flops_charged_to_primary_utility": True,
            "selector_flops_charged_to_primary_utility": True,
            "selected_preview_double_counted": False,
            "selected_path_cost": "C_branch + C_selector",
            "unselected_path_cost": "C_stop + G(prefix) + C_selector",
            "false_positive_harms_counted_directly": True,
        },
        "probes": probes,
        "court_classification": classification,
        "successor_design_may_be_considered": classification
        in {
            "PREFIX1_DIRECT_UTILITY_POSITIVE",
            "PREFIX2_DIRECT_UTILITY_POSITIVE",
            "PREFIX3_DIRECT_UTILITY_POSITIVE",
        },
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "analysis_boundary": (
            "augmentation-only DEVELOPMENT cost-accounted branch-preview upper-bound court; "
            "oracle route cardinality is non-deployable and fresh evaluation remains locked"
        ),
        "resource_pair_audit_digest": canonical_sha256(pair_audit),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp279_cost_accounted_branch_preview_v5(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 V5 artifact: " + "; ".join(errors))
    return payload
