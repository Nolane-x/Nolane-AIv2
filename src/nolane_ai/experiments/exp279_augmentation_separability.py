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


SCHEMA = "NLM-EXP-279-AUGMENTATION-SEPARABILITY-COURT-V1"
PROBE_KINDS = ("LINEAR_MEAN", "MLP_MEAN", "DEEPSETS")
MINIMUM_POOLED_ROC_AUC = 0.75
PROBE_ROOT_SUFFIX = "::independent-augmentation-probe-v1"


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
    ranked_labels = labels_long.index_select(0, order)
    cumulative = torch.cumsum(ranked_labels.to(torch.float64), dim=0)
    ranks = torch.arange(1, ranked_labels.numel() + 1, dtype=torch.float64)
    precision = cumulative / ranks
    return float(precision[ranked_labels == 1].sum().item() / positives)


def _top_k_precision(scores: torch.Tensor, labels: torch.Tensor, *, k: int) -> float | None:
    scores64, labels_long = _validate_rank_inputs(scores, labels)
    if k <= 0:
        return None
    k = min(int(k), int(scores64.numel()))
    order = torch.argsort(scores64, descending=True, stable=True)[:k]
    return float(labels_long.index_select(0, order).to(torch.float64).mean().item())


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
        self.phi = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        pooled = self.phi(states).mean(dim=1)
        return self.rho(pooled).squeeze(-1)


def _build_probe(kind: str, *, hidden_size: int, seed: int) -> nn.Module:
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown EXP-279 separability probe kind: {kind}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "LINEAR_MEAN":
            return _LinearMeanProbe(hidden_size)
        if kind == "MLP_MEAN":
            return _MlpMeanProbe(hidden_size)
        return _DeepSetsProbe(hidden_size)


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


def _fit_probe_fold(
    *,
    kind: str,
    states: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
    folds: int,
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
    test_supported = test_counts["positive"] > 0 and test_counts["negative"] > 0

    seed = derive_stream_seed(
        probe_root_seed,
        f"EXP-279-SEPARABILITY-{kind}",
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
        test_scores = probe(test_states)
    return {
        "fold": fold,
        "folds": folds,
        "seed": seed,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_supported": train_supported,
        "test_supported": test_supported,
        "support_closed": train_supported and test_supported,
        "initial_pairwise_loss": initial_loss,
        "final_pairwise_loss": final_loss,
        "test_indices": test_mask.nonzero(as_tuple=False).squeeze(-1).tolist(),
        "test_scores": test_scores.detach().cpu().tolist(),
        "test_labels": test_labels.detach().cpu().to(torch.long).tolist(),
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
            folds=folds,
            hidden_size=hidden_size,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_root_seed=probe_root_seed,
        )
        for fold in range(folds)
    ]
    oof_scores = torch.empty(labels.shape[0], dtype=torch.float64)
    oof_labels = labels.detach().cpu().to(torch.long).clone()
    for row in fold_rows:
        indices = torch.tensor(row["test_indices"], dtype=torch.long)
        scores = torch.tensor(row["test_scores"], dtype=torch.float64)
        oof_scores.index_copy_(0, indices, scores)

    counts = _class_counts(oof_labels)
    k = counts["positive"]
    auc = _roc_auc(oof_scores, oof_labels)
    average_precision = _average_precision(oof_scores, oof_labels)
    top_k_precision = _top_k_precision(oof_scores, oof_labels, k=k)
    support_closed = all(bool(row["support_closed"]) for row in fold_rows)
    tail_separable = (
        support_closed
        and auc is not None
        and auc >= MINIMUM_POOLED_ROC_AUC
        and top_k_precision is not None
        and top_k_precision > economic_break_even_probability
    )
    base_rate = counts["positive"] / counts["total"] if counts["total"] else 0.0
    return {
        "kind": kind,
        "parameter_count": sum(parameter.numel() for parameter in _build_probe(
            kind,
            hidden_size=hidden_size,
            seed=derive_stream_seed(probe_root_seed, f"EXP-279-SEPARABILITY-{kind}", 0, "parameter_audit"),
        ).parameters()),
        "folds": fold_rows,
        "support_closed": support_closed,
        "pooled_counts": counts,
        "natural_rescue_prevalence": base_rate,
        "pooled_roc_auc": auc,
        "pooled_average_precision": average_precision,
        "top_k": k,
        "top_k_precision": top_k_precision,
        "top_k_enrichment_over_prevalence": (
            top_k_precision / base_rate
            if top_k_precision is not None and base_rate > 0.0
            else None
        ),
        "economic_break_even_probability": economic_break_even_probability,
        "economically_tail_separable": tail_separable,
    }


def _court_classification(probes: dict[str, dict[str, Any]]) -> str:
    if not all(bool(probe["support_closed"]) for probe in probes.values()):
        return "INSUFFICIENT_RESCUE_SUPPORT"
    if bool(probes["LINEAR_MEAN"]["economically_tail_separable"]):
        return "LINEAR_TAIL_SEPARABLE"
    if any(
        bool(probes[kind]["economically_tail_separable"])
        for kind in ("MLP_MEAN", "DEEPSETS")
    ):
        return "RICH_HEAD_TAIL_SEPARABLE"
    return "REPRESENTATION_NOT_ECONOMICALLY_SEPARABLE"


def validate_exp279_augmentation_separability_court(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("augmentation separability schema mismatch")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("augmentation separability evidence boundary mismatch")
    if payload.get("scientific_evidence_eligible") is not False:
        errors.append("augmentation separability court cannot be scientific-evidence eligible")
    for key in ("confirmatory_data_consumed", "challenge_materialized", "promotion_claimed"):
        if payload.get(key) is not False:
            errors.append(f"augmentation separability {key} must remain false")

    boundary = payload.get("data_boundary") or {}
    if boundary.get("training_rng_stream") != "augmentation":
        errors.append("canonical training must remain augmentation-only")
    if boundary.get("probe_rng_stream") != "augmentation":
        errors.append("separability probe stream must remain augmentation-only")
    if boundary.get("evaluation_rng_stream_used") is not False:
        errors.append("separability court cannot use the evaluation RNG stream")
    if boundary.get("evaluation_targets_used") is not False:
        errors.append("separability court cannot use evaluation targets")
    if boundary.get("probe_root_seed") == payload.get("root_seed"):
        errors.append("probe root seed must be independent from canonical training root")

    config = payload.get("probe_config") or {}
    if config.get("folds") != 3:
        errors.append("separability court is frozen to three folds")
    if config.get("probe_kinds") != list(PROBE_KINDS):
        errors.append("separability probe family changed")
    if config.get("k_rule") != "pooled_true_rescue_count":
        errors.append("separability top-k rule changed")
    if config.get("decision_threshold_tuned") is not False:
        errors.append("separability court cannot tune a deployable threshold")

    rule = payload.get("survival_rule") or {}
    if float(rule.get("minimum_pooled_roc_auc", -1.0)) != MINIMUM_POOLED_ROC_AUC:
        errors.append("separability ROC-AUC survival floor changed")
    if rule.get("tail_precision_threshold_source") != "sealed_compute_ledger_break_even":
        errors.append("separability tail threshold must come from sealed compute economics")
    if rule.get("tail_k_rule") != "pooled_true_rescue_count":
        errors.append("separability tail k rule changed")
    if rule.get("requires_all_folds_positive_and_negative_support") is not True:
        errors.append("separability fold support gate must remain fail-closed")
    if rule.get("decision_threshold_tuned") is not False:
        errors.append("separability survival rule cannot tune routing threshold")

    probes = payload.get("probes") or {}
    if set(probes) != set(PROBE_KINDS):
        errors.append("separability probe receipts incomplete")
    classification = payload.get("court_classification")
    allowed = {
        "INSUFFICIENT_RESCUE_SUPPORT",
        "LINEAR_TAIL_SEPARABLE",
        "RICH_HEAD_TAIL_SEPARABLE",
        "REPRESENTATION_NOT_ECONOMICALLY_SEPARABLE",
    }
    if classification not in allowed:
        errors.append("separability court classification invalid")
    elif set(probes) == set(PROBE_KINDS) and classification != _court_classification(probes):
        errors.append("separability court classification does not match probe receipts")

    expected_fresh = classification in {"LINEAR_TAIL_SEPARABLE", "RICH_HEAD_TAIL_SEPARABLE"}
    if payload.get("fresh_evaluation_lineage_may_be_reserved") is not expected_fresh:
        errors.append("fresh-lineage reservation gate does not match court classification")
    if payload.get("fresh_evaluation_lineage_consumed") is not False:
        errors.append("augmentation separability court cannot consume a fresh evaluation lineage")
    return errors


def run_exp279_augmentation_separability_court(
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
        raise ValueError("EXP-279 separability counts and dimensions must be positive")
    if probe_folds != 3:
        raise ValueError("EXP-279 separability court is frozen to three folds")
    if probe_replicates % (len(STRATA) * probe_folds) != 0:
        raise ValueError("probe_replicates must balance all strata across all folds")
    if constraints > variables or timesteps < constraints:
        raise ValueError("EXP-279 separability world geometry is invalid")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0, 1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0 or probe_lr <= 0.0:
        raise ValueError("EXP-279 separability optimizer/noise parameters are invalid")

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
        raise RuntimeError("EXP-279 separability matched resource contract did not close")

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
    label_rows: list[torch.Tensor] = []
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
                scope="synthetic-exp279-routing-development-separability-probe",
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
            rescue = ((~stop_exact) & branch_exact).to(torch.long)
            fold = _assign_probe_fold(replicate, folds=probe_folds)

            state_rows.append(propagation_state.detach().cpu())
            label_rows.append(rescue.detach().cpu())
            fold_rows.append(torch.full((batch_size,), fold, dtype=torch.long))
            probe_batch_digests.append(batch.digest)
            probe_strata.append(stratum)

    states = torch.cat(state_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
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
        "analysis_scope": "augmentation_only_frozen_representation_cross_fitted_rescue_separability",
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
            "k_rule": "pooled_true_rescue_count",
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
            "minimum_pooled_roc_auc": MINIMUM_POOLED_ROC_AUC,
            "tail_precision_threshold_source": "sealed_compute_ledger_break_even",
            "tail_k_rule": "pooled_true_rescue_count",
            "requires_all_folds_positive_and_negative_support": True,
            "decision_threshold_tuned": False,
        },
        "probes": probes,
        "court_classification": classification,
        "fresh_evaluation_lineage_may_be_reserved": classification
        in {"LINEAR_TAIL_SEPARABLE", "RICH_HEAD_TAIL_SEPARABLE"},
        "fresh_evaluation_lineage_consumed": False,
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "promotion_claimed": False,
        "analysis_boundary": (
            "augmentation-only DEVELOPMENT separability diagnostic; no evaluation stream, "
            "no fresh evaluation lineage, and no promotion claim"
        ),
        "resource_pair_audit_digest": canonical_sha256(pair_audit),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp279_augmentation_separability_court(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 augmentation separability artifact: " + "; ".join(errors))
    return payload
