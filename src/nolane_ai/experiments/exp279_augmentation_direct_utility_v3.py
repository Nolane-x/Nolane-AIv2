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


SCHEMA = "NLM-EXP-279-AUGMENTATION-DIRECT-UTILITY-COURT-V3"
PROBE_KINDS = ("LINEAR_MEAN", "MLP_MEAN", "DEEPSETS")
PROBE_ROOT_SUFFIX = "::independent-augmentation-probe-v3"


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
        raise ValueError("pairwise probe scores and labels must be aligned rank-1 tensors")
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


def _fold_direct_utility_metrics(
    scores: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    *,
    stop_accounted_flops_per_episode: int | float,
    branch_accounted_flops_per_episode: int | float,
) -> dict[str, Any]:
    if scores.ndim != 1 or stop_exact.ndim != 1 or branch_exact.ndim != 1:
        raise ValueError("V3 fold scores/outcomes must be rank-1")
    if not (scores.shape == stop_exact.shape == branch_exact.shape):
        raise ValueError("V3 fold scores/outcomes must be aligned")
    if scores.numel() == 0:
        raise ValueError("V3 fold must contain at least one episode")
    stop_cost = float(stop_accounted_flops_per_episode)
    branch_cost = float(branch_accounted_flops_per_episode)
    if stop_cost <= 0.0 or branch_cost <= stop_cost:
        raise ValueError("V3 direct utility requires 0 < stop FLOPs < branch FLOPs")

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
    expected_solution_delta = selected_rescues - selected_harms
    actual_solution_delta = routed_solution_count - stop_solution_count
    if actual_solution_delta != expected_solution_delta:
        raise RuntimeError("V3 rescue/harm accounting does not match counterfactual solution delta")

    stop_total_flops = stop_cost * episode_count
    routed_total_flops = stop_total_flops + route_k * (branch_cost - stop_cost)
    stop_utility = stop_solution_count / stop_total_flops
    routed_utility = routed_solution_count / routed_total_flops
    relative_gain = (
        (routed_utility - stop_utility) / abs(stop_utility)
        if stop_utility > 0.0
        else None
    )
    selected_precision = selected_rescues / route_k if route_k > 0 else None
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
        "selected_rescue_precision": selected_precision,
        "stop_solution_count": stop_solution_count,
        "routed_solution_count": routed_solution_count,
        "solution_count_delta": actual_solution_delta,
        "stop_total_accounted_flops": stop_total_flops,
        "routed_total_accounted_flops": routed_total_flops,
        "stop_verified_utility": stop_utility,
        "routed_verified_utility": routed_utility,
        "relative_verified_utility_gain": relative_gain,
        "direct_utility_improved": support_closed and routed_utility > stop_utility,
        "roc_auc": auc,
        "average_precision": ap,
        "concordance_pair_count": counts["positive"] * counts["negative"],
        "heldout_score_outcome_digest": receipt_digest,
        "raw_scores_exported": False,
    }


def _aggregate_fold_direct_utility_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V3 aggregate requires at least one fold")
    support_closed = all(bool(row.get("support_closed")) for row in rows)
    stop_solutions = sum(int(row["stop_solution_count"]) for row in rows)
    routed_solutions = sum(int(row["routed_solution_count"]) for row in rows)
    stop_flops = sum(float(row["stop_total_accounted_flops"]) for row in rows)
    routed_flops = sum(float(row["routed_total_accounted_flops"]) for row in rows)
    if stop_flops <= 0.0 or routed_flops <= 0.0:
        raise ValueError("V3 aggregate FLOPs must be positive")
    stop_utility = stop_solutions / stop_flops
    routed_utility = routed_solutions / routed_flops
    relative_gain = (
        (routed_utility - stop_utility) / abs(stop_utility)
        if stop_utility > 0.0
        else None
    )
    selected = sum(int(row["route_k"]) for row in rows)
    rescues = sum(int(row["selected_rescues"]) for row in rows)
    harms = sum(int(row["selected_harms"]) for row in rows)
    neutral = sum(int(row["selected_neutral"]) for row in rows)
    rescue_count = sum(int(row["rescue_count"]) for row in rows)
    episode_count = sum(int(row["episode_count"]) for row in rows)
    total_pairs = sum(int(row["concordance_pair_count"]) for row in rows)
    if selected != rescue_count:
        raise RuntimeError("V3 aggregate oracle route cardinality drift")
    if routed_solutions - stop_solutions != rescues - harms:
        raise RuntimeError("V3 aggregate solution delta does not match rescue/harm accounting")

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
        "routed_total_accounted_flops": routed_flops,
        "stop_verified_utility": stop_utility,
        "routed_verified_utility": routed_utility,
        "relative_verified_utility_gain": relative_gain,
        "direct_utility_improved": support_closed and routed_utility > stop_utility,
        "cross_fitted_roc_auc": auc,
        "cross_fitted_average_precision": ap,
        "concordance_pair_count": total_pairs,
        "raw_scores_compared_across_folds": False,
    }


def _propagation_state(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    _, variables = arm._validate_common(surface_events, variable_states)
    checked_incidence = arm._validate_incidence(incidence, variables)
    propagated = arm._propagate(variables, checked_incidence)
    return variables + propagated


class _LinearMeanProbe(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.readout(states.mean(dim=1)).squeeze(-1)


class _MlpMeanProbe(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states.mean(dim=1)).squeeze(-1)


class _DeepSetsProbe(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.rho = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.rho(self.phi(states).mean(dim=1)).squeeze(-1)


def _build_probe(kind: str, *, hidden_size: int, seed: int) -> nn.Module:
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown EXP-279 V3 probe kind: {kind}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "LINEAR_MEAN":
            return _LinearMeanProbe(hidden_size)
        if kind == "MLP_MEAN":
            return _MlpMeanProbe(hidden_size)
        return _DeepSetsProbe(hidden_size)


def _fit_probe_fold(
    *,
    kind: str,
    states: torch.Tensor,
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
) -> dict[str, Any]:
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold
    train_states = states[train_mask]
    train_stop = stop_exact[train_mask]
    train_branch = branch_exact[train_mask]
    train_labels = ((~train_stop) & train_branch).to(torch.long)
    train_counts = _class_counts(train_labels)
    train_supported = train_counts["positive"] > 0 and train_counts["negative"] > 0

    seed = derive_stream_seed(
        probe_root_seed,
        f"EXP-279-DIRECT-UTILITY-V3-{kind}",
        fold,
        "model_init",
    )
    probe = _build_probe(kind, hidden_size=hidden_size, seed=seed)
    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_lr, weight_decay=0.0)
    initial_loss: float | None = None
    final_loss: float | None = None
    if train_supported:
        probe.train()
        for step in range(probe_steps):
            optimizer.zero_grad(set_to_none=True)
            scores = probe(train_states)
            loss = _pairwise_ranking_loss(scores, train_labels)
            if step == 0:
                initial_loss = float(loss.detach().item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    probe.eval()
    with torch.no_grad():
        test_scores = probe(states[test_mask])
    metrics = _fold_direct_utility_metrics(
        test_scores,
        stop_exact[test_mask],
        branch_exact[test_mask],
        stop_accounted_flops_per_episode=stop_cost,
        branch_accounted_flops_per_episode=branch_cost,
    )
    test_supported = bool(metrics["support_closed"])
    metrics["support_closed"] = train_supported and test_supported
    metrics["direct_utility_improved"] = bool(
        metrics["support_closed"]
        and float(metrics["routed_verified_utility"]) > float(metrics["stop_verified_utility"])
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


def _cross_fitted_probe(
    *,
    kind: str,
    states: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    fold_ids: torch.Tensor,
    folds: int,
    hidden_size: int,
    probe_steps: int,
    probe_lr: float,
    probe_root_seed: str,
    stop_cost: float,
    branch_cost: float,
) -> dict[str, Any]:
    fold_rows = [
        _fit_probe_fold(
            kind=kind,
            states=states,
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
        )
        for fold in range(folds)
    ]
    aggregate = _aggregate_fold_direct_utility_metrics(fold_rows)
    seed = derive_stream_seed(
        probe_root_seed,
        f"EXP-279-DIRECT-UTILITY-V3-{kind}",
        0,
        "model_init",
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in _build_probe(kind, hidden_size=hidden_size, seed=seed).parameters()
    )
    return {
        "kind": kind,
        "parameter_count": parameter_count,
        "folds": fold_rows,
        "aggregate": aggregate,
        "economically_routable": bool(aggregate["direct_utility_improved"]),
        "raw_scores_compared_across_folds": False,
    }


def _court_classification(probes: dict[str, dict[str, Any]]) -> str:
    if not all(bool(probe["aggregate"]["support_closed"]) for probe in probes.values()):
        return "INSUFFICIENT_RESCUE_SUPPORT"
    if bool(probes["LINEAR_MEAN"]["economically_routable"]):
        return "LINEAR_DIRECT_UTILITY_POSITIVE"
    if any(bool(probes[kind]["economically_routable"]) for kind in ("MLP_MEAN", "DEEPSETS")):
        return "RICH_HEAD_DIRECT_UTILITY_POSITIVE"
    return "REPRESENTATION_NOT_DIRECTLY_ECONOMIC"


def validate_exp279_augmentation_direct_utility_v3(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("V3 direct-utility schema mismatch")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("V3 evidence boundary mismatch")
    if payload.get("scientific_evidence_eligible") is not False:
        errors.append("V3 cannot be scientific-evidence eligible")
    for key in (
        "fresh_evaluation_lineage_consumed",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
    ):
        if payload.get(key) is not False:
            errors.append(f"V3 {key} must remain false")

    boundary = payload.get("data_boundary") or {}
    if boundary.get("training_rng_stream") != "augmentation":
        errors.append("V3 canonical training must remain augmentation-only")
    if boundary.get("probe_rng_stream") != "augmentation":
        errors.append("V3 probe stream must remain augmentation-only")
    if boundary.get("evaluation_rng_stream_used") is not False:
        errors.append("V3 cannot use the evaluation RNG stream")
    if boundary.get("evaluation_targets_used") is not False:
        errors.append("V3 cannot use evaluation targets")
    if boundary.get("probe_root_seed") == payload.get("root_seed"):
        errors.append("V3 probe root must be independent from canonical training root")

    config = payload.get("probe_config") or {}
    if config.get("probe_kinds") != list(PROBE_KINDS):
        errors.append("V3 probe family changed")
    if config.get("folds") != 3:
        errors.append("V3 court is frozen to three folds")
    if config.get("fold_route_k_rule") != "heldout_fold_true_rescue_count":
        errors.append("V3 oracle route cardinality rule changed")
    if config.get("oracle_route_cardinality_is_deployable") is not False:
        errors.append("V3 oracle route cardinality cannot be deployable")
    if config.get("raw_scores_compared_across_folds") is not False:
        errors.append("V3 cannot compare raw scores across folds")
    if config.get("raw_scores_exported") is not False:
        errors.append("V3 cannot export raw held-out scores")

    rule = payload.get("economic_rule") or {}
    if rule.get("metric") != "direct_counterfactual_verified_solution_per_total_accounted_flop":
        errors.append("V3 direct economic metric changed")
    if rule.get("posterior_break_even_threshold_used") is not False:
        errors.append("V3 cannot use a posterior break-even threshold")
    if rule.get("auc_floor_used_for_decision") is not False:
        errors.append("V3 AUC cannot gate the economic decision")
    if rule.get("cross_fold_aggregation") != "sum_solution_counts_and_total_accounted_flops":
        errors.append("V3 economic aggregation changed")

    economics = payload.get("compute_economics") or {}
    stop_cost = float(economics.get("stop_accounted_flops_per_episode", 0.0) or 0.0)
    branch_cost = float(economics.get("branch_accounted_flops_per_episode", 0.0) or 0.0)
    if stop_cost <= 0.0 or branch_cost <= stop_cost:
        errors.append("V3 compute ledger costs invalid")

    probes = payload.get("probes") or {}
    if set(probes) != set(PROBE_KINDS):
        errors.append("V3 probe receipts incomplete")
    else:
        for kind, probe in probes.items():
            if probe.get("raw_scores_compared_across_folds") is not False:
                errors.append(f"V3 {kind} compared raw scores across folds")
            aggregate = probe.get("aggregate") or {}
            if aggregate.get("raw_scores_compared_across_folds") is not False:
                errors.append(f"V3 {kind} aggregate compared raw scores across folds")
            expected_pass = bool(
                aggregate.get("support_closed")
                and float(aggregate.get("routed_verified_utility", 0.0))
                > float(aggregate.get("stop_verified_utility", 0.0))
            )
            if probe.get("economically_routable") is not expected_pass:
                errors.append(f"V3 {kind} economic pass receipt mismatch")
            for fold in probe.get("folds") or []:
                if fold.get("raw_scores_exported") is not False:
                    errors.append(f"V3 {kind} exported raw fold scores")

    classification = payload.get("court_classification")
    allowed = {
        "INSUFFICIENT_RESCUE_SUPPORT",
        "LINEAR_DIRECT_UTILITY_POSITIVE",
        "RICH_HEAD_DIRECT_UTILITY_POSITIVE",
        "REPRESENTATION_NOT_DIRECTLY_ECONOMIC",
    }
    if classification not in allowed:
        errors.append("V3 court classification invalid")
    elif set(probes) == set(PROBE_KINDS) and classification != _court_classification(probes):
        errors.append("V3 court classification does not match receipts")
    expected_successor = classification in {
        "LINEAR_DIRECT_UTILITY_POSITIVE",
        "RICH_HEAD_DIRECT_UTILITY_POSITIVE",
    }
    if payload.get("successor_design_may_be_considered") is not expected_successor:
        errors.append("V3 successor design gate does not match classification")
    if payload.get("fresh_evaluation_lineage_may_be_reserved") is not False:
        errors.append("V3 cannot reserve fresh evaluation lineage directly")
    return errors


def run_exp279_augmentation_direct_utility_v3_court(
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
        raise ValueError("V3 root_seed, protocol_digest and code_digest are required")
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
        raise ValueError("V3 counts and dimensions must be positive")
    if probe_folds != 3:
        raise ValueError("V3 court is frozen to three folds")
    if probe_replicates % (len(STRATA) * probe_folds) != 0:
        raise ValueError("V3 probe_replicates must balance all strata across folds")
    if constraints > variables or timesteps < constraints:
        raise ValueError("V3 world geometry is invalid")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0, 1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0 or probe_lr <= 0.0:
        raise ValueError("V3 optimizer/noise parameters invalid")

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
        raise RuntimeError("V3 matched resource contract did not close")

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
    state_rows: list[torch.Tensor] = []
    stop_rows: list[torch.Tensor] = []
    branch_rows: list[torch.Tensor] = []
    fold_rows: list[torch.Tensor] = []
    probe_batch_digests: list[str] = []
    probe_strata: list[str] = []
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
                scope="synthetic-exp279-routing-development-direct-utility-v3-probe",
            )
            propagation_state = _propagation_state(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
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

            state_rows.append(propagation_state.detach().cpu())
            stop_rows.append(stop_exact.detach().cpu())
            branch_rows.append(branch_exact.detach().cpu())
            fold_rows.append(torch.full((batch_size,), fold, dtype=torch.long))
            probe_batch_digests.append(batch.digest)
            probe_strata.append(stratum)

    states = torch.cat(state_rows, dim=0)
    stop_exact = torch.cat(stop_rows, dim=0).to(torch.bool)
    branch_exact = torch.cat(branch_rows, dim=0).to(torch.bool)
    fold_ids = torch.cat(fold_rows, dim=0)
    rescue_labels = ((~stop_exact) & branch_exact).to(torch.long)
    pooled_counts = _class_counts(rescue_labels)

    hybrid_ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_cost = float(hybrid_ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(hybrid_ledger["branch_accounted_flops_per_episode"])
    probes = {
        kind: _cross_fitted_probe(
            kind=kind,
            states=states,
            stop_exact=stop_exact,
            branch_exact=branch_exact,
            fold_ids=fold_ids,
            folds=probe_folds,
            hidden_size=hidden_size,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_root_seed=probe_root_seed,
            stop_cost=stop_cost,
            branch_cost=branch_cost,
        )
        for kind in PROBE_KINDS
    }
    classification = _court_classification(probes)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "experiment_id": "EXP-279",
        "analysis_scope": "augmentation_only_frozen_representation_fold_local_direct_counterfactual_utility",
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
        "arm_geometry": {
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
        },
        "route_config": {
            "canonical_threshold": float(route_threshold),
            "threshold_used_by_v3_court": False,
            "threshold_tuned": False,
        },
        "canonical_training": {
            "replicates": train_replicates,
            "rng_stream": "augmentation",
            "paired_batch_digests": training_batch_digests,
            "losses": training_losses,
            "frozen_hybrid_functional_state_digest": frozen_hybrid_digest,
            "representation_frozen_before_probe_fit": True,
        },
        "probe_config": {
            "replicates": probe_replicates,
            "folds": probe_folds,
            "fold_assignment": "floor(replicate / number_of_strata) mod folds",
            "probe_kinds": list(PROBE_KINDS),
            "pairwise_ranking_loss": "softplus(-(positive_score-negative_score))",
            "steps": probe_steps,
            "lr": float(probe_lr),
            "weight_decay": 0.0,
            "fold_route_k_rule": "heldout_fold_true_rescue_count",
            "oracle_route_cardinality_is_deployable": False,
            "raw_scores_compared_across_folds": False,
            "raw_scores_exported": False,
            "decision_threshold_tuned": False,
        },
        "probe_dataset": {
            "batch_digests": probe_batch_digests,
            "strata": probe_strata,
            "episode_counts": pooled_counts,
            "natural_rescue_prevalence": (
                pooled_counts["positive"] / pooled_counts["total"]
                if pooled_counts["total"]
                else 0.0
            ),
            "fold_episode_counts": {
                str(fold): _class_counts(rescue_labels[fold_ids == fold])
                for fold in range(probe_folds)
            },
        },
        "compute_economics": {
            "stop_accounted_flops_per_episode": int(stop_cost),
            "branch_accounted_flops_per_episode": int(branch_cost),
            "incremental_branch_flops_per_selected_episode": int(branch_cost - stop_cost),
            "source": "sealed_matched_arm_compute_ledger",
        },
        "economic_rule": {
            "metric": "direct_counterfactual_verified_solution_per_total_accounted_flop",
            "cross_fold_aggregation": "sum_solution_counts_and_total_accounted_flops",
            "pass_condition": "support_closed_and_routed_verified_utility_strictly_greater_than_stop_verified_utility",
            "posterior_break_even_threshold_used": False,
            "auc_floor_used_for_decision": False,
            "fold_selection": "top_k_within_each_heldout_fold_only",
            "false_positive_harms_counted_directly": True,
            "path_dependent_flops_charged_directly": True,
        },
        "probes": probes,
        "court_classification": classification,
        "successor_design_may_be_considered": classification
        in {"LINEAR_DIRECT_UTILITY_POSITIVE", "RICH_HEAD_DIRECT_UTILITY_POSITIVE"},
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
        "analysis_boundary": (
            "augmentation-only DEVELOPMENT direct-utility diagnostic; no evaluation stream, "
            "no fresh evaluation lineage, and no promotion claim"
        ),
        "resource_pair_audit_digest": canonical_sha256(pair_audit),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp279_augmentation_direct_utility_v3(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 V3 direct-utility artifact: " + "; ".join(errors))
    return payload
