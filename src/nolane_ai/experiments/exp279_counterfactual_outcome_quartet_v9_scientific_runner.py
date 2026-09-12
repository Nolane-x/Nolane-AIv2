from __future__ import annotations

import re
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.training.optimizer import build_functional_optimizer

from .exp279_counterfactual_outcome_quartet_v9 import (
    DECISION_INDICES,
    DECISION_REPLICATES,
    FIT_INDICES,
    FIT_REPLICATES,
    FROZEN_PROTOCOL_DIGEST,
    PRIMARY_FAMILY,
    ROOT_PREFIX,
    SCHEMA_SHARD,
    STUDENT_GEOMETRY,
    STUDENT_OPTIMIZER,
    TRAIN_BUDGETS,
    _model_state_digest,
    apply_quartet_policy,
    canonical_root,
    cheap_prebranch_features,
    classify_quartet_root,
    decision_root,
    fit_quartet_student,
    fit_root,
    fit_stop_utility,
    heldout_policy_metrics,
    marginal_route_score,
    quartet_labels,
    student_inference_flops,
)
from .exp279_counterfactual_outcome_quartet_v9_runner_primitives import (
    combine_decision_root_metrics,
    fit_rescue_control,
)
from .exp279_paired_runner import (
    _build_seeded_triplet,
    _hybrid_forced_branch_decision_logits,
    _hybrid_stop_decision_logits,
    _train_step,
)
from .exp279_routing_worlds import Exp279RoutingGenerator
from .matched_routing_arms import audit_matched_exp279_arm_triplet

PROTOCOL_ID = "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
FROZEN_WORLD_GEOMETRY = {
    "batch_size": 8,
    "timesteps": 4,
    "variables": 6,
    "constraints": 3,
    "d_model": 64,
    "noise_std": 0.05,
}
FROZEN_ARM_GEOMETRY = {"hidden_size": 48, "target_parameters": 500_000}
FROZEN_CANONICAL_OPTIMIZER = {"lr": 0.002, "weight_decay": 0.0}
FROZEN_ROUTE_THRESHOLD = 0.5
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_commit(value: str, *, label: str) -> str:
    text = str(value)
    if _HEX40.fullmatch(text) is None:
        raise ValueError(f"V9 {label} must be an exact lowercase 40-hex commit SHA")
    return text


def _require_digest(value: str, *, label: str) -> str:
    text = str(value)
    if _HEX64.fullmatch(text) is None:
        raise ValueError(f"V9 {label} must be an exact lowercase 64-hex digest")
    return text


def _canonical_freeze_digest(model: torch.nn.Module) -> str:
    return _model_state_digest(model)


def exact_outcome_tensors(
    stop_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stop_logits.shape != branch_logits.shape:
        raise ValueError("V9 stop and forced-branch logits must have identical shape")
    if stop_logits.ndim != 3 or stop_logits.shape[-1] != 2:
        raise ValueError("V9 stop/branch logits must be [batch,variables,2]")
    if tuple(targets.shape) != tuple(stop_logits.shape[:-1]):
        raise ValueError("V9 target shape must match stop/branch logits")
    stop_exact = (stop_logits.argmax(dim=-1) == targets).all(dim=-1)
    branch_exact = (branch_logits.argmax(dim=-1) == targets).all(dim=-1)
    return stop_exact, branch_exact


def _train_canonical_hybrid(*, train_replicates: int, canonical_index: int) -> tuple[Any, dict[str, Any]]:
    root = canonical_root(train_replicates, canonical_index)
    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=root,
        d_model=FROZEN_WORLD_GEOMETRY["d_model"],
        hidden_size=FROZEN_ARM_GEOMETRY["hidden_size"],
        target_parameters=FROZEN_ARM_GEOMETRY["target_parameters"],
        route_threshold=FROZEN_ROUTE_THRESHOLD,
    )
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=FROZEN_WORLD_GEOMETRY["timesteps"],
        variables=FROZEN_WORLD_GEOMETRY["variables"],
        constraints=FROZEN_WORLD_GEOMETRY["constraints"],
        max_accounted_flops_per_episode=None,
    )
    required = (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "compute_budget_closed",
    )
    if not all(pair_audit.get(key) is True for key in required):
        raise RuntimeError("V9 matched-resource contract did not close")

    optimizer = build_functional_optimizer(
        hybrid,
        lr=FROZEN_CANONICAL_OPTIMIZER["lr"],
        weight_decay=FROZEN_CANONICAL_OPTIMIZER["weight_decay"],
    )
    generator = Exp279RoutingGenerator(root_seed=root)
    from .matched_routing_arms import STRATA

    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=FROZEN_WORLD_GEOMETRY["batch_size"],
            timesteps=FROZEN_WORLD_GEOMETRY["timesteps"],
            variables=FROZEN_WORLD_GEOMETRY["variables"],
            constraints=FROZEN_WORLD_GEOMETRY["constraints"],
            d_model=FROZEN_WORLD_GEOMETRY["d_model"],
            noise_std=FROZEN_WORLD_GEOMETRY["noise_std"],
            rng_stream="augmentation",
            stratum=STRATA[replicate % len(STRATA)],
        )
        _train_step(
            hybrid,
            optimizer,
            arm_id="hybrid",
            surface_events=batch.surface_events,
            variable_states=batch.variable_states,
            incidence=batch.incidence,
            targets=batch.targets,
        )

    final_digest = _canonical_freeze_digest(hybrid)
    hybrid.eval()
    for parameter in hybrid.parameters():
        parameter.requires_grad_(False)

    ledger = pair_audit["compute_ledger"]["hybrid"]
    training_receipt = {
        "canonical_root": root,
        "model_init_seed": model_init_seed,
        "final_canonical_digest": final_digest,
        "resource_pair_audit_digest": pair_audit.get("pair_audit_digest"),
        "stop_accounted_flops_per_episode": int(ledger["stop_accounted_flops_per_episode"]),
        "branch_accounted_flops_per_episode": int(ledger["branch_accounted_flops_per_episode"]),
    }
    return hybrid, training_receipt


def _fold_quartet_counts(labels: torch.Tensor, *, replicate: int) -> dict[str, int]:
    from .exp279_counterfactual_outcome_quartet_v9 import DIAGNOSTIC_FOLDS, QuartetClass, diagnostic_fold

    fold = diagnostic_fold(replicate)
    counts = {name: 0 for name in ("RESCUE", "HARM", "BOTH_SUCCESS", "BOTH_FAILURE")}
    for member in QuartetClass:
        counts[member.name] = int((labels == int(member)).sum().item())
    return {"fold": fold, "episodes": int(labels.numel()), **counts, "diagnostic_folds": DIAGNOSTIC_FOLDS}


def _collect_counterfactual_root(
    hybrid: Any,
    *,
    root: str,
    replicates: int,
) -> dict[str, Any]:
    from .exp279_counterfactual_outcome_quartet_v9 import DIAGNOSTIC_FOLDS
    from .matched_routing_arms import STRATA

    if replicates <= 0:
        raise ValueError("V9 root replicates must be positive")
    generator = Exp279RoutingGenerator(root_seed=root)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    stop_exact_values: list[torch.Tensor] = []
    branch_exact_values: list[torch.Tensor] = []
    fold_counts = [
        {"fold": fold, "episodes": 0, "RESCUE": 0, "HARM": 0, "BOTH_SUCCESS": 0, "BOTH_FAILURE": 0}
        for fold in range(DIAGNOSTIC_FOLDS)
    ]

    with torch.no_grad():
        for replicate in range(replicates):
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=FROZEN_WORLD_GEOMETRY["batch_size"],
                timesteps=FROZEN_WORLD_GEOMETRY["timesteps"],
                variables=FROZEN_WORLD_GEOMETRY["variables"],
                constraints=FROZEN_WORLD_GEOMETRY["constraints"],
                d_model=FROZEN_WORLD_GEOMETRY["d_model"],
                noise_std=FROZEN_WORLD_GEOMETRY["noise_std"],
                rng_stream="augmentation",
                stratum=STRATA[replicate % len(STRATA)],
            )
            x = cheap_prebranch_features(
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
            stop_exact, branch_exact = exact_outcome_tensors(stop_logits, branch_logits, batch.targets)
            y = quartet_labels(stop_exact, branch_exact)
            features.append(x.detach().cpu())
            labels.append(y.detach().cpu())
            stop_exact_values.append(stop_exact.detach().cpu())
            branch_exact_values.append(branch_exact.detach().cpu())

            batch_counts = _fold_quartet_counts(y.detach().cpu(), replicate=replicate)
            target = fold_counts[int(batch_counts["fold"])]
            target["episodes"] += int(batch_counts["episodes"])
            for key in ("RESCUE", "HARM", "BOTH_SUCCESS", "BOTH_FAILURE"):
                target[key] += int(batch_counts[key])

    all_features = torch.cat(features, dim=0)
    all_labels = torch.cat(labels, dim=0)
    all_stop = torch.cat(stop_exact_values, dim=0)
    all_branch = torch.cat(branch_exact_values, dim=0)
    episodes = int(all_labels.numel())
    if episodes != replicates * FROZEN_WORLD_GEOMETRY["batch_size"]:
        raise RuntimeError("V9 root episode geometry did not close")
    if sum(int(item["episodes"]) for item in fold_counts) != episodes:
        raise RuntimeError("V9 fold episode partition did not close")
    return {
        "root": root,
        "features": all_features,
        "labels": all_labels,
        "stop_exact": all_stop,
        "branch_exact": all_branch,
        "fold_counts": fold_counts,
        "episodes": episodes,
    }


def _public_root_summary(root_data: dict[str, Any]) -> dict[str, Any]:
    labels = root_data["labels"]
    from .exp279_counterfactual_outcome_quartet_v9 import QuartetClass

    counts = {member.name: int((labels == int(member)).sum().item()) for member in QuartetClass}
    return {
        "root": root_data["root"],
        "episodes": int(root_data["episodes"]),
        "quartet_counts": counts,
        "fold_counts": root_data["fold_counts"],
    }


def run_exp279_counterfactual_outcome_quartet_v9_shard(
    *,
    train_replicates: int,
    canonical_index: int,
    protocol_digest: str,
    code_digest: str,
    scientific_branch_head: str,
    executed_commit: str,
) -> dict[str, Any]:
    budget = int(train_replicates)
    canonical = int(canonical_index)
    if budget not in TRAIN_BUDGETS:
        raise ValueError(f"V9 train budget must be one of {TRAIN_BUDGETS}")
    if canonical not in (0, 1, 2, 3):
        raise ValueError("V9 canonical index must be one of 0..3")
    if protocol_digest != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("V9 scientific runner requires the exact frozen Stage-A protocol digest")
    code = _require_digest(code_digest, label="code digest")
    branch_head = _require_commit(scientific_branch_head, label="scientific branch head")
    executed = _require_commit(executed_commit, label="executed commit")

    hybrid, canonical_training = _train_canonical_hybrid(
        train_replicates=budget,
        canonical_index=canonical,
    )
    frozen_digest = str(canonical_training["final_canonical_digest"])
    stop_cost = int(canonical_training["stop_accounted_flops_per_episode"])
    branch_cost = int(canonical_training["branch_accounted_flops_per_episode"])
    student_cost = student_inference_flops(
        hidden_size=FROZEN_ARM_GEOMETRY["hidden_size"],
        variables=FROZEN_WORLD_GEOMETRY["variables"],
        timesteps=FROZEN_WORLD_GEOMETRY["timesteps"],
    )

    fit_roots = [fit_root(budget, canonical, index) for index in FIT_INDICES]
    fit_sets = [
        _collect_counterfactual_root(hybrid, root=root, replicates=FIT_REPLICATES)
        for root in fit_roots
    ]
    if _canonical_freeze_digest(hybrid) != frozen_digest:
        raise RuntimeError("V9 canonical model changed while collecting fit roots")
    fit_features = torch.cat([item["features"] for item in fit_sets], dim=0)
    fit_labels = torch.cat([item["labels"] for item in fit_sets], dim=0)
    fit_stop = torch.cat([item["stop_exact"] for item in fit_sets], dim=0)

    primary_fit = fit_quartet_student(
        fit_features,
        fit_labels,
        train_replicates=budget,
        canonical_index=canonical,
        fit_roots=fit_roots,
    )
    control_fit = fit_rescue_control(
        fit_features,
        fit_labels,
        train_replicates=budget,
        canonical_index=canonical,
        fit_roots=fit_roots,
    )
    if _canonical_freeze_digest(hybrid) != frozen_digest:
        raise RuntimeError("V9 canonical model changed while fitting students")

    fit_stop_metrics = fit_stop_utility(
        fit_stop,
        stop_accounted_flops=stop_cost,
        student_accounted_flops=student_cost,
    )
    quartet_model = primary_fit["model"]
    quartet_model.eval()

    decision_roots = [decision_root(budget, canonical, index) for index in DECISION_INDICES]
    decision_sets = [
        _collect_counterfactual_root(hybrid, root=root, replicates=DECISION_REPLICATES)
        for root in decision_roots
    ]
    if _canonical_freeze_digest(hybrid) != frozen_digest:
        raise RuntimeError("V9 canonical model changed while collecting decision roots")

    decision_receipts: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in decision_sets:
            probabilities = F.softmax(quartet_model(item["features"]), dim=-1)
            scores = marginal_route_score(
                probabilities,
                fit_stop_utility=float(fit_stop_metrics["utility"]),
                stop_accounted_flops=stop_cost,
                branch_accounted_flops=branch_cost,
            )
            chosen_branch = apply_quartet_policy(scores)
            metrics = heldout_policy_metrics(
                chosen_branch,
                item["stop_exact"],
                item["branch_exact"],
                stop_accounted_flops=stop_cost,
                branch_accounted_flops=branch_cost,
                student_accounted_flops=student_cost,
            )
            decision_receipts.append(
                {
                    **_public_root_summary(item),
                    "metrics": metrics,
                    "score_positive_count": int((scores > 0).sum().item()),
                }
            )

    root_metrics = combine_decision_root_metrics([item["metrics"] for item in decision_receipts])
    root_classification = classify_quartet_root(root_metrics)
    if _canonical_freeze_digest(hybrid) != frozen_digest:
        raise RuntimeError("V9 canonical model changed after decision evaluation")

    receipt: dict[str, Any] = {
        "schema": SCHEMA_SHARD,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol_digest,
        "code_digest": code,
        "scientific_branch_head": branch_head,
        "executed_commit": executed,
        "root_prefix": ROOT_PREFIX,
        "train_replicates": budget,
        "canonical_index": canonical,
        "canonical_root": canonical_training["canonical_root"],
        "fit_roots": fit_roots,
        "decision_roots": decision_roots,
        "rng_streams": {
            "canonical_training": "augmentation",
            "fit": "augmentation",
            "decision": "augmentation",
        },
        "evaluation_rng_stream_used": False,
        "evaluation_targets_used": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
        "threshold_tuned": False,
        "world_geometry": dict(FROZEN_WORLD_GEOMETRY),
        "arm_geometry": dict(FROZEN_ARM_GEOMETRY),
        "canonical_optimizer": dict(FROZEN_CANONICAL_OPTIMIZER),
        "canonical_route_threshold": FROZEN_ROUTE_THRESHOLD,
        "student_family": PRIMARY_FAMILY,
        "student_geometry": dict(STUDENT_GEOMETRY),
        "student_optimizer": dict(STUDENT_OPTIMIZER),
        "fit_replicates_per_root": FIT_REPLICATES,
        "decision_replicates_per_root": DECISION_REPLICATES,
        "student_accounted_flops_per_episode": student_cost,
        "stop_accounted_flops_per_episode": stop_cost,
        "branch_accounted_flops_per_episode": branch_cost,
        "model_init_seed": canonical_training["model_init_seed"],
        "final_canonical_digest": frozen_digest,
        "resource_pair_audit_digest": canonical_training["resource_pair_audit_digest"],
        "primary_fit": {
            key: value
            for key, value in primary_fit.items()
            if key != "model"
        },
        "control_fit": {
            key: value
            for key, value in control_fit.items()
            if key != "model"
        },
        "fit_stop_utility": fit_stop_metrics,
        "fit_root_summaries": [_public_root_summary(item) for item in fit_sets],
        "decision_root_summaries": decision_receipts,
        "root_metrics": root_metrics,
        "root_classification": root_classification,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "artifact_digest"}
    )
    return receipt


__all__ = [
    "FROZEN_WORLD_GEOMETRY",
    "FROZEN_ARM_GEOMETRY",
    "FROZEN_CANONICAL_OPTIMIZER",
    "FROZEN_ROUTE_THRESHOLD",
    "_canonical_freeze_digest",
    "exact_outcome_tensors",
    "run_exp279_counterfactual_outcome_quartet_v9_shard",
]
