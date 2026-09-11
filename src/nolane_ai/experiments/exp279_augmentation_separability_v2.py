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


SCHEMA = "NLM-EXP-279-AUGMENTATION-SEPARABILITY-COURT-V2"
PROBE_KINDS = ("LINEAR_MEAN", "MLP_MEAN", "DEEPSETS")
MINIMUM_CROSS_FITTED_ROC_AUC = 0.75
PROBE_ROOT_SUFFIX = "::independent-augmentation-probe-v2"


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


def _economic_break_even_probability(
    *,
    stop_accounted_flops_per_episode: float,
    branch_accounted_flops_per_episode: float,
) -> float:
    stop_flops = float(stop_accounted_flops_per_episode)
    branch_flops = float(branch_accounted_flops_per_episode)
    if stop_flops <= 0.0 or branch_flops <= stop_flops:
        raise ValueError("economic break-even requires 0 < stop FLOPs < branch FLOPs")
    incremental_cost_ratio = (branch_flops - stop_flops) / branch_flops
    return incremental_cost_ratio / (1.0 + incremental_cost_ratio)


def _validate_rank_inputs(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("ranking scores and labels must be aligned rank-1 tensors")
    if scores.numel() == 0:
        raise ValueError("ranking metrics require at least one example")
    labels_long = labels.to(dtype=torch.long)
    if bool(((labels_long != 0) & (labels_long != 1)).any().item()):
        raise ValueError("ranking labels must be binary")
    return scores.to(dtype=torch.float64), labels_long


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores64, labels_long = _validate_rank_inputs(scores, labels)
    positive = scores64[labels_long == 1]
    negative = scores64[labels_long == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    delta = positive[:, None] - negative[None, :]
    wins = (delta > 0).to(torch.float64).sum()
    ties = (delta == 0).to(torch.float64).sum()
    pair_count = positive.numel() * negative.numel()
    return float(((wins + 0.5 * ties) / pair_count).item())


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


def _fold_ranking_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    scores64, labels_long = _validate_rank_inputs(scores, labels)
    positives = int((labels_long == 1).sum().item())
    negatives = int((labels_long == 0).sum().item())
    total = int(labels_long.numel())
    support_closed = positives > 0 and negatives > 0
    pair_count = positives * negatives
    if not support_closed:
        return {
            "support_closed": False,
            "positive_count": positives,
            "negative_count": negatives,
            "total_count": total,
            "concordance_pair_count": pair_count,
            "roc_auc": None,
            "average_precision": None,
            "tail_k": positives,
            "tail_selected_rescues": None,
            "tail_precision": None,
        }

    auc = _roc_auc(scores64, labels_long)
    ap = _average_precision(scores64, labels_long)
    k = positives
    order = torch.argsort(scores64, descending=True, stable=True)[:k]
    selected_rescues = int(labels_long.index_select(0, order).sum().item())
    return {
        "support_closed": True,
        "positive_count": positives,
        "negative_count": negatives,
        "total_count": total,
        "concordance_pair_count": pair_count,
        "roc_auc": auc,
        "average_precision": ap,
        "tail_k": k,
        "tail_selected_rescues": selected_rescues,
        "tail_precision": selected_rescues / k,
    }


def _aggregate_fold_ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one fold metric row is required")
    support_closed = all(row.get("support_closed") is True for row in rows)
    total_pairs = sum(int(row.get("concordance_pair_count", 0)) for row in rows)
    total_positive = sum(int(row.get("positive_count", 0)) for row in rows)
    total_negative = sum(int(row.get("negative_count", 0)) for row in rows)
    total_episodes = sum(int(row.get("total_count", 0)) for row in rows)
    if not support_closed or total_pairs <= 0 or total_positive <= 0:
        return {
            "support_closed": False,
            "cross_fitted_roc_auc": None,
            "cross_fitted_average_precision": None,
            "cross_fitted_tail_precision": None,
            "tail_selected_rescues": None,
            "tail_selected_episodes": total_positive,
            "concordance_pair_count": total_pairs,
            "positive_count": total_positive,
            "negative_count": total_negative,
            "total_count": total_episodes,
            "raw_scores_compared_across_folds": False,
        }

    weighted_auc = sum(
        float(row["roc_auc"]) * int(row["concordance_pair_count"])
        for row in rows
    ) / total_pairs
    weighted_ap = sum(
        float(row["average_precision"]) * int(row["positive_count"])
        for row in rows
    ) / total_positive
    selected_rescues = sum(int(row["tail_selected_rescues"]) for row in rows)
    selected_episodes = sum(int(row["tail_k"]) for row in rows)
    return {
        "support_closed": True,
        "cross_fitted_roc_auc": weighted_auc,
        "cross_fitted_average_precision": weighted_ap,
        "cross_fitted_tail_precision": selected_rescues / selected_episodes,
        "tail_selected_rescues": selected_rescues,
        "tail_selected_episodes": selected_episodes,
        "concordance_pair_count": total_pairs,
        "positive_count": total_positive,
        "negative_count": total_negative,
        "total_count": total_episodes,
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
        raise ValueError(f"unknown EXP-279 V2 probe kind: {kind}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "LINEAR_MEAN":
            return _LinearMeanProbe(hidden_size)
        if kind == "MLP_MEAN":
            return _MlpMeanProbe(hidden_size)
        return _DeepSetsProbe(hidden_size)


def _pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("probe scores and labels must be aligned rank-1 tensors")
    positive = scores[labels > 0]
    negative = scores[labels <= 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.sum() * 0.0
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def _class_counts(labels: torch.Tensor) -> dict[str, int]:
    labels_long = labels.to(dtype=torch.long)
    positive = int((labels_long == 1).sum().item())
    total = int(labels_long.numel())
    return {"positive": positive, "negative": total - positive, "total": total}


def _fit_probe_fold(
    *,
    kind: str,
    states: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
    hidden_size: int,
    probe_steps: int,
    probe_lr: float,
    probe_root_seed: str,
) -> dict[str, Any]:
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold
    train_states = states[train_mask]
    train_labels = labels[train_mask]
    test_states = states[test_mask]
    test_labels = labels[test_mask]
    train_counts = _class_counts(train_labels)
    test_counts = _class_counts(test_labels)
    train_supported = train_counts["positive"] > 0 and train_counts["negative"] > 0

    seed = derive_stream_seed(
        probe_root_seed,
        f"EXP-279-SEPARABILITY-V2-{kind}",
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
            loss = _pairwise_ranking_loss(probe(train_states), train_labels)
            if step == 0:
                initial_loss = float(loss.detach().item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    probe.eval()
    with torch.no_grad():
        test_scores = probe(test_states)
    metrics = _fold_ranking_metrics(test_scores.detach().cpu(), test_labels.detach().cpu())
    score_receipt = canonical_sha256(
        {
            "scores": test_scores.detach().cpu().to(torch.float64).tolist(),
            "labels": test_labels.detach().cpu().to(torch.long).tolist(),
        }
    )
    return {
        "fold": fold,
        "seed": seed,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_supported": train_supported,
        "initial_pairwise_loss": initial_loss,
        "final_pairwise_loss": final_loss,
        "heldout_score_label_digest": score_receipt,
        "heldout_metrics": metrics,
        "raw_scores_exported": False,
    }


def _cross_fitted_probe(
    *,
    kind: str,
    states: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    folds: int,
    hidden_size: int,
    probe_steps: int,
    probe_lr: float,
    probe_root_seed: str,
    economic_break_even_probability: float,
) -> dict[str, Any]:
    fold_rows = [
        _fit_probe_fold(
            kind=kind,
            states=states,
            labels=labels,
            fold_ids=fold_ids,
            fold=fold,
            hidden_size=hidden_size,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_root_seed=probe_root_seed,
        )
        for fold in range(folds)
    ]
    aggregate = _aggregate_fold_ranking_metrics(
        [row["heldout_metrics"] for row in fold_rows]
    )
    tail_separable = (
        aggregate["support_closed"]
        and aggregate["cross_fitted_roc_auc"] is not None
        and float(aggregate["cross_fitted_roc_auc"]) >= MINIMUM_CROSS_FITTED_ROC_AUC
        and aggregate["cross_fitted_tail_precision"] is not None
        and float(aggregate["cross_fitted_tail_precision"])
        > economic_break_even_probability
    )
    return {
        "kind": kind,
        "parameter_count": sum(
            parameter.numel()
            for parameter in _build_probe(kind, hidden_size=hidden_size, seed=0).parameters()
        ),
        "folds": fold_rows,
        "aggregate": aggregate,
        "economic_break_even_probability": economic_break_even_probability,
        "economically_tail_separable": bool(tail_separable),
        "raw_scores_compared_across_folds": False,
    }


def _court_classification(probes: dict[str, dict[str, Any]]) -> str:
    if not all(bool(probe["aggregate"]["support_closed"]) for probe in probes.values()):
        return "INSUFFICIENT_RESCUE_SUPPORT"
    if bool(probes["LINEAR_MEAN"]["economically_tail_separable"]):
        return "LINEAR_TAIL_SEPARABLE"
    if any(
        bool(probes[kind]["economically_tail_separable"])
        for kind in ("MLP_MEAN", "DEEPSETS")
    ):
        return "RICH_HEAD_TAIL_SEPARABLE"
    return "REPRESENTATION_NOT_ECONOMICALLY_SEPARABLE"


def validate_exp279_augmentation_separability_v2_court(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("V2 separability schema mismatch")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("V2 separability evidence boundary mismatch")
    if payload.get("scientific_evidence_eligible") is not False:
        errors.append("V2 court cannot be scientific-evidence eligible")
    for key in ("confirmatory_data_consumed", "challenge_materialized", "promotion_claimed"):
        if payload.get(key) is not False:
            errors.append(f"V2 separability {key} must remain false")

    boundary = payload.get("data_boundary") or {}
    if boundary.get("training_rng_stream") != "augmentation":
        errors.append("V2 canonical training must remain augmentation-only")
    if boundary.get("probe_rng_stream") != "augmentation":
        errors.append("V2 probe stream must remain augmentation-only")
    if boundary.get("evaluation_rng_stream_used") is not False:
        errors.append("V2 court cannot use evaluation RNG stream")
    if boundary.get("evaluation_targets_used") is not False:
        errors.append("V2 court cannot use evaluation targets")
    probe_root = boundary.get("probe_root_seed")
    if not isinstance(probe_root, str) or not probe_root.endswith(PROBE_ROOT_SUFFIX):
        errors.append("V2 probe root suffix mismatch")
    if probe_root == payload.get("root_seed"):
        errors.append("V2 probe root must be independent from canonical training root")

    config = payload.get("probe_config") or {}
    if config.get("probe_kinds") != list(PROBE_KINDS):
        errors.append("V2 probe family changed")
    if config.get("folds") != 3:
        errors.append("V2 court is frozen to three folds")
    if config.get("raw_scores_compared_across_folds") is not False:
        errors.append("V2 cannot compare raw scores across folds")
    if config.get("fold_tail_k_rule") != "heldout_fold_true_rescue_count":
        errors.append("V2 fold-local tail-k rule changed")
    if config.get("decision_threshold_tuned") is not False:
        errors.append("V2 cannot tune deployable threshold")

    rule = payload.get("survival_rule") or {}
    if float(rule.get("minimum_cross_fitted_roc_auc", -1.0)) != MINIMUM_CROSS_FITTED_ROC_AUC:
        errors.append("V2 AUC floor changed")
    if rule.get("auc_aggregation") != "within_fold_concordance_pairs_only":
        errors.append("V2 AUC aggregation changed")
    if rule.get("tail_aggregation") != "sum_fold_selected_rescues_over_sum_fold_true_rescue_count":
        errors.append("V2 tail aggregation changed")
    if rule.get("economic_interpretation") != "optimistic_compute_only_necessary_condition_false_positive_harms_unpriced":
        errors.append("V2 economic interpretation changed")
    if rule.get("requires_every_fold_positive_and_negative_support") is not True:
        errors.append("V2 support gate must remain fail-closed")

    probes = payload.get("probes") or {}
    if set(probes) != set(PROBE_KINDS):
        errors.append("V2 probe receipts incomplete")
    else:
        for kind, probe in probes.items():
            aggregate = probe.get("aggregate") or {}
            if probe.get("raw_scores_compared_across_folds") is not False:
                errors.append(f"V2 {kind} compared raw scores across folds")
            if aggregate.get("raw_scores_compared_across_folds") is not False:
                errors.append(f"V2 {kind} aggregate compared raw scores across folds")
            for fold in probe.get("folds") or []:
                if fold.get("raw_scores_exported") is not False:
                    errors.append(f"V2 {kind} exported raw heldout scores")

    classification = payload.get("court_classification")
    allowed = {
        "INSUFFICIENT_RESCUE_SUPPORT",
        "LINEAR_TAIL_SEPARABLE",
        "RICH_HEAD_TAIL_SEPARABLE",
        "REPRESENTATION_NOT_ECONOMICALLY_SEPARABLE",
    }
    if classification not in allowed:
        errors.append("V2 court classification invalid")
    elif set(probes) == set(PROBE_KINDS) and classification != _court_classification(probes):
        errors.append("V2 classification does not match probe receipts")

    may_reserve = classification in {"LINEAR_TAIL_SEPARABLE", "RICH_HEAD_TAIL_SEPARABLE"}
    if payload.get("successor_design_may_be_considered") is not may_reserve:
        errors.append("V2 successor-design gate does not match classification")
    if payload.get("fresh_evaluation_lineage_consumed") is not False:
        errors.append("V2 court cannot consume fresh evaluation lineage")
    return errors


def run_exp279_augmentation_separability_v2_court(
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
        raise ValueError("root_seed, protocol_digest and code_digest are required")
    counts = (
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
    )
    if min(counts) <= 0:
        raise ValueError("V2 counts and dimensions must be positive")
    if probe_folds != 3:
        raise ValueError("V2 separability court is frozen to three folds")
    if probe_replicates % (len(STRATA) * probe_folds) != 0:
        raise ValueError("V2 probe_replicates must balance strata across folds")
    if constraints > variables or timesteps < constraints:
        raise ValueError("V2 world geometry is invalid")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0,1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0 or probe_lr <= 0.0:
        raise ValueError("V2 noise/optimizer parameters are invalid")

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
        raise RuntimeError("V2 matched resource contract did not close")

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
    states_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
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
                scope="synthetic-exp279-routing-development-separability-v2-probe",
            )
            states = _propagation_state(
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
            rescue = ((~stop_exact) & branch_exact).to(torch.long)
            fold = _assign_probe_fold(replicate, folds=probe_folds)
            states_rows.append(states.detach().cpu())
            labels_rows.append(rescue.detach().cpu())
            fold_rows.append(torch.full((batch_size,), fold, dtype=torch.long))
            probe_batch_digests.append(batch.digest)
            probe_strata.append(stratum)

    states = torch.cat(states_rows, dim=0)
    labels = torch.cat(labels_rows, dim=0)
    fold_ids = torch.cat(fold_rows, dim=0)
    pooled_counts = _class_counts(labels)

    hybrid_ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_flops = float(hybrid_ledger["stop_accounted_flops_per_episode"])
    branch_flops = float(hybrid_ledger["branch_accounted_flops_per_episode"])
    break_even = _economic_break_even_probability(
        stop_accounted_flops_per_episode=stop_flops,
        branch_accounted_flops_per_episode=branch_flops,
    )

    probes = {
        kind: _cross_fitted_probe(
            kind=kind,
            states=states,
            labels=labels,
            fold_ids=fold_ids,
            folds=probe_folds,
            hidden_size=hidden_size,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_root_seed=probe_root_seed,
            economic_break_even_probability=break_even,
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
        "analysis_scope": "augmentation_only_fold_local_cross_fitted_rescue_separability_v2",
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
            "threshold": float(route_threshold),
            "threshold_used_by_court": False,
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
            "fold_tail_k_rule": "heldout_fold_true_rescue_count",
            "raw_scores_compared_across_folds": False,
            "raw_scores_exported": False,
            "decision_threshold_tuned": False,
            "oracle_cardinality_is_deployable": False,
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
                str(fold): _class_counts(labels[fold_ids == fold])
                for fold in range(probe_folds)
            },
        },
        "compute_economics": {
            "stop_accounted_flops_per_episode": int(stop_flops),
            "branch_accounted_flops_per_episode": int(branch_flops),
            "incremental_cost_ratio": (branch_flops - stop_flops) / branch_flops,
            "economic_break_even_probability": break_even,
            "source": "sealed_matched_arm_compute_ledger",
        },
        "survival_rule": {
            "minimum_cross_fitted_roc_auc": MINIMUM_CROSS_FITTED_ROC_AUC,
            "auc_aggregation": "within_fold_concordance_pairs_only",
            "tail_k_rule": "heldout_fold_true_rescue_count",
            "tail_aggregation": "sum_fold_selected_rescues_over_sum_fold_true_rescue_count",
            "tail_precision_threshold_source": "sealed_compute_ledger_break_even",
            "economic_break_even_probability": break_even,
            "economic_interpretation": "optimistic_compute_only_necessary_condition_false_positive_harms_unpriced",
            "requires_every_fold_positive_and_negative_support": True,
            "raw_scores_compared_across_folds": False,
            "decision_threshold_tuned": False,
        },
        "probes": probes,
        "court_classification": classification,
        "successor_design_may_be_considered": classification
        in {"LINEAR_TAIL_SEPARABLE", "RICH_HEAD_TAIL_SEPARABLE"},
        "fresh_evaluation_lineage_consumed": False,
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
        "analysis_boundary": (
            "augmentation-only DEVELOPMENT diagnostic; fold-local ranking only; "
            "no evaluation stream, no fresh evaluation lineage, no promotion claim"
        ),
        "resource_pair_audit_digest": canonical_sha256(pair_audit),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp279_augmentation_separability_v2_court(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 separability V2 artifact: " + "; ".join(errors))
    return payload
