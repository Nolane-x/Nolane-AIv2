from __future__ import annotations

import re
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.training.optimizer import build_functional_optimizer

from .exp279_counterfactual_representation_identifiability_v10 import (
    REPRESENTATION_DIMENSIONS,
    REPRESENTATION_VIEWS,
    classify_representation_root,
    counterfactual_action_labels,
    direct_policy_metrics,
    exact_1nn_predict,
    extract_representation_views,
)
from .exp279_paired_runner import (
    _build_seeded_triplet,
    _functional_state_digest,
    _hybrid_forced_branch_decision_logits,
    _hybrid_stop_decision_logits,
    _train_step,
)
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import audit_matched_exp279_arm_triplet

SCHEMA = "NLM-EXP-279-V10-CRIC-SHARD-V1"
PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
ROOT_PREFIX = "20260912-exp279-v10-cric-dev"
FIT_REPLICATES_PER_ROOT = 128
HELDOUT_REPLICATES_PER_ROOT = 128
FROZEN_GEOMETRY = {
    "batch_size": 8,
    "timesteps": 4,
    "variables": 6,
    "constraints": 3,
    "d_model": 64,
    "hidden_size": 48,
    "target_parameters": 500000,
    "noise_std": 0.05,
    "route_threshold": 0.5,
}
FROZEN_OPTIMIZER = {"lr": 0.002, "weight_decay": 0.0}
FROZEN_NN = {
    "metric": "squared_euclidean_after_fit_zscore",
    "k": 1,
    "query_chunk_size": 256,
    "tie_break": "first_fit_row",
    "router_flops_charged": False,
}
_MATCHED_AUDIT_KEYS = (
    "parameter_match",
    "functional_parameter_match",
    "active_functional_parameter_match",
    "optimizer_visible_parameter_match",
    "reclaimed_parameter_assignment_closed",
    "compute_budget_closed",
)


def build_root_schedule(train_replicates: int, canonical_index: int) -> dict[str, Any]:
    if train_replicates not in {60, 120}:
        raise ValueError("V10 train budget must be exactly 60 or 120")
    if canonical_index not in {0, 1, 2, 3}:
        raise ValueError("V10 canonical index must be in 0..3")
    base = f"{ROOT_PREFIX}/train-{train_replicates}/canonical-{canonical_index}"
    return {
        "training_root": f"{base}/train",
        "fit_roots": [f"{base}/fit-0", f"{base}/fit-1"],
        "heldout_roots": [f"{base}/heldout-0", f"{base}/heldout-1"],
    }


def _require_identity(
    *,
    protocol_digest: str,
    code_digest: str,
    scientific_branch_head: str,
    executed_commit: str,
) -> None:
    if protocol_digest != PROTOCOL_DIGEST:
        raise ValueError("V10 protocol digest does not match frozen Stage-A protocol")
    if re.fullmatch(r"[0-9a-f]{64}", code_digest) is None:
        raise ValueError("V10 code_digest must be lowercase sha256")
    for field, value in (
        ("scientific_branch_head", scientific_branch_head),
        ("executed_commit", executed_commit),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"V10 {field} must be a 40-hex Git SHA")


def _batch_digest(root: str, digests: list[str]) -> str:
    return canonical_sha256({"root": root, "batch_digests": digests})


def _make_batch(
    generator: Exp279RoutingGenerator,
    *,
    replicate: int,
) -> Any:
    return generator.make_batch(
        replicate=replicate,
        batch_size=FROZEN_GEOMETRY["batch_size"],
        timesteps=FROZEN_GEOMETRY["timesteps"],
        variables=FROZEN_GEOMETRY["variables"],
        constraints=FROZEN_GEOMETRY["constraints"],
        d_model=FROZEN_GEOMETRY["d_model"],
        noise_std=FROZEN_GEOMETRY["noise_std"],
        rng_stream="augmentation",
        stratum=STRATA[replicate % len(STRATA)],
    )


def _collect_fit(
    hybrid: Any,
    roots: list[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, str], dict[str, int]]:
    features: dict[str, list[torch.Tensor]] = {name: [] for name in REPRESENTATION_VIEWS}
    labels: list[torch.Tensor] = []
    root_digests: dict[str, str] = {}
    outcome_counts = {"RESCUE": 0, "HARM": 0, "BOTH_SUCCESS": 0, "BOTH_FAILURE": 0}
    for root in sorted(roots):
        generator = Exp279RoutingGenerator(root_seed=root)
        batch_digests: list[str] = []
        for replicate in range(FIT_REPLICATES_PER_ROOT):
            batch = _make_batch(generator, replicate=replicate)
            batch_digests.append(batch.digest)
            views = extract_representation_views(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            for name in REPRESENTATION_VIEWS:
                features[name].append(views[name].detach().cpu())
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
            batch_labels, outcomes = counterfactual_action_labels(
                stop_logits,
                branch_logits,
                batch.targets,
            )
            labels.append(batch_labels.detach().cpu())
            for code, name in enumerate(("RESCUE", "HARM", "BOTH_SUCCESS", "BOTH_FAILURE")):
                outcome_counts[name] += int((outcomes == code).to(torch.int64).sum().item())
        root_digests[root] = _batch_digest(root, batch_digests)
    return (
        {name: torch.cat(rows, dim=0) for name, rows in features.items()},
        torch.cat(labels, dim=0),
        root_digests,
        outcome_counts,
    )


def _collect_heldout(
    hybrid: Any,
    roots: list[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, str]]:
    features: dict[str, list[torch.Tensor]] = {name: [] for name in REPRESENTATION_VIEWS}
    outcomes_all: list[torch.Tensor] = []
    root_digests: dict[str, str] = {}
    for root in sorted(roots):
        generator = Exp279RoutingGenerator(root_seed=root)
        batch_digests: list[str] = []
        for replicate in range(HELDOUT_REPLICATES_PER_ROOT):
            batch = _make_batch(generator, replicate=replicate)
            batch_digests.append(batch.digest)
            views = extract_representation_views(
                hybrid,
                surface_events=batch.surface_events,
                variable_states=batch.variable_states,
                incidence=batch.incidence,
            )
            for name in REPRESENTATION_VIEWS:
                features[name].append(views[name].detach().cpu())
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
            _, outcomes = counterfactual_action_labels(stop_logits, branch_logits, batch.targets)
            outcomes_all.append(outcomes.detach().cpu())
        root_digests[root] = _batch_digest(root, batch_digests)
    return (
        {name: torch.cat(rows, dim=0) for name, rows in features.items()},
        torch.cat(outcomes_all, dim=0),
        root_digests,
    )


def run_v10_cric_shard(
    *,
    train_replicates: int,
    canonical_index: int,
    protocol_digest: str,
    code_digest: str,
    scientific_branch_head: str,
    executed_commit: str,
) -> dict[str, Any]:
    _require_identity(
        protocol_digest=protocol_digest,
        code_digest=code_digest,
        scientific_branch_head=scientific_branch_head,
        executed_commit=executed_commit,
    )
    roots = build_root_schedule(train_replicates, canonical_index)
    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=roots["training_root"],
        d_model=FROZEN_GEOMETRY["d_model"],
        hidden_size=FROZEN_GEOMETRY["hidden_size"],
        target_parameters=FROZEN_GEOMETRY["target_parameters"],
        route_threshold=FROZEN_GEOMETRY["route_threshold"],
    )
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=FROZEN_GEOMETRY["timesteps"],
        variables=FROZEN_GEOMETRY["variables"],
        constraints=FROZEN_GEOMETRY["constraints"],
    )
    if not all(pair_audit.get(key) is True for key in _MATCHED_AUDIT_KEYS):
        raise RuntimeError("V10 matched-resource contract did not close")

    hybrid_ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_cost = float(hybrid_ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(hybrid_ledger["branch_accounted_flops_per_episode"])
    if stop_cost <= 0.0 or branch_cost < stop_cost:
        raise RuntimeError("V10 hybrid compute ledger is invalid")

    optimizer = build_functional_optimizer(
        hybrid,
        lr=FROZEN_OPTIMIZER["lr"],
        weight_decay=FROZEN_OPTIMIZER["weight_decay"],
    )
    training_generator = Exp279RoutingGenerator(root_seed=roots["training_root"])
    training_digests: list[str] = []
    losses: list[float] = []
    for replicate in range(train_replicates):
        batch = _make_batch(training_generator, replicate=replicate)
        training_digests.append(batch.digest)
        losses.append(
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

    canonical_digest = _functional_state_digest(hybrid)
    hybrid.eval()
    with torch.no_grad():
        fit_features, fit_labels, fit_root_digests, fit_outcomes = _collect_fit(
            hybrid, roots["fit_roots"]
        )
        if _functional_state_digest(hybrid) != canonical_digest:
            raise RuntimeError("V10 canonical model changed while collecting fit roots")
        heldout_features, heldout_outcomes, heldout_root_digests = _collect_heldout(
            hybrid, roots["heldout_roots"]
        )
        if _functional_state_digest(hybrid) != canonical_digest:
            raise RuntimeError("V10 canonical model changed while collecting heldout roots")

    representation_results: dict[str, Any] = {}
    for name in REPRESENTATION_VIEWS:
        predictions = exact_1nn_predict(
            fit_features[name],
            fit_labels,
            heldout_features[name],
            chunk_size=FROZEN_NN["query_chunk_size"],
        )
        metrics = direct_policy_metrics(
            predictions=predictions,
            outcomes=heldout_outcomes,
            stop_cost=stop_cost,
            branch_cost=branch_cost,
        )
        representation_results[name] = {
            "dimension": REPRESENTATION_DIMENSIONS[name],
            "fit_branch_labels": int(fit_labels.sum().item()),
            "fit_stop_labels": int(fit_labels.numel() - fit_labels.sum().item()),
            "heldout_metrics": metrics,
            "root_classification": classify_representation_root(metrics),
        }

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "analysis_scope": "development_counterfactual_representation_identifiability_upper_bound",
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "scientific_branch_head": scientific_branch_head,
        "executed_commit": executed_commit,
        "train_replicates": train_replicates,
        "canonical_index": canonical_index,
        "roots": roots,
        "model_init_seed": model_init_seed,
        "world_model_geometry": dict(FROZEN_GEOMETRY),
        "optimizer": dict(FROZEN_OPTIMIZER),
        "nearest_neighbor_contract": dict(FROZEN_NN),
        "fit_replicates_per_root": FIT_REPLICATES_PER_ROOT,
        "heldout_replicates_per_root": HELDOUT_REPLICATES_PER_ROOT,
        "training_batches_digest": _batch_digest(roots["training_root"], training_digests),
        "fit_root_batch_digests": fit_root_digests,
        "heldout_root_batch_digests": heldout_root_digests,
        "training_loss_summary": {
            "count": len(losses),
            "mean": sum(losses) / len(losses),
            "final": losses[-1],
        },
        "canonical_model_digest": canonical_digest,
        "canonical_model_digest_after_court": _functional_state_digest(hybrid),
        "matched_resource_audit": {key: bool(pair_audit[key]) for key in _MATCHED_AUDIT_KEYS},
        "compute_ledger": {
            "stop_accounted_flops_per_episode": stop_cost,
            "branch_accounted_flops_per_episode": branch_cost,
        },
        "fit_outcome_counts": fit_outcomes,
        "representations": representation_results,
        "evaluation_rng_used": False,
        "evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "mechanism_successor_authorized": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
    }
    receipt["artifact_digest"] = canonical_sha256(receipt)
    return receipt
