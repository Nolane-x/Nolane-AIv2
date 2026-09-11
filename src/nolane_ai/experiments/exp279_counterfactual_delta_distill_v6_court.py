from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.training.optimizer import build_functional_optimizer

from .exp279_counterfactual_delta_distill_v6_fit import fit_cdd_fold
from .exp279_counterfactual_delta_distill_v6_primitives import (
    CONTROL_FAMILY,
    PRIMARY_FAMILY,
    PROBE_ROOT_SUFFIX,
    SCHEMA,
    cdd_costs,
    control_costs,
)
from .exp279_counterfactual_delta_distill_v6_utility import aggregate_fold_direct_utility_metrics
from .exp279_counterfactual_delta_distill_v6_validation import court_classification, validate_artifact
from .exp279_paired_runner import _build_seeded_triplet, _functional_state_digest, _train_step
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import audit_matched_exp279_arm_triplet


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _assign_probe_fold(replicate: int, folds: int) -> int:
    if replicate < 0 or folds <= 1:
        raise ValueError("invalid V6 fold geometry")
    return (replicate // len(STRATA)) % folds


def run_court(
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
    distill_steps: int,
    distill_lr: float,
    selector_steps: int,
    selector_lr: float,
    protocol_digest: str,
    code_digest: str,
    max_accounted_flops_per_episode: int | None = None,
) -> dict[str, Any]:
    if not root_seed or not protocol_digest or not code_digest:
        raise ValueError("V6 root_seed, protocol_digest and code_digest are required")
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
        distill_steps,
        selector_steps,
    ) <= 0:
        raise ValueError("V6 counts and dimensions must be positive")
    if probe_folds != 3:
        raise ValueError("V6 is frozen to three folds")
    if probe_replicates % (len(STRATA) * probe_folds) != 0:
        raise ValueError("V6 probe replicates must balance strata across folds")
    if constraints > variables or not 0.0 <= route_threshold <= 1.0:
        raise ValueError("invalid V6 geometry or threshold")
    if noise_std < 0 or lr <= 0 or weight_decay < 0 or distill_lr <= 0 or selector_lr <= 0:
        raise ValueError("invalid V6 optimizer/noise settings")

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
    required = (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "compute_budget_closed",
    )
    if not all(pair_audit.get(key) is True for key in required):
        raise RuntimeError("V6 matched resource contract did not close")

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

    frozen_digest = _functional_state_digest(hybrid)
    for parameter in hybrid.parameters():
        parameter.requires_grad_(False)
    hybrid.eval()

    probe_root = root_seed + PROBE_ROOT_SUFFIX
    probe_generator = Exp279RoutingGenerator(root_seed=probe_root)
    event_rows: list[torch.Tensor] = []
    variable_rows: list[torch.Tensor] = []
    incidence_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    fold_rows: list[torch.Tensor] = []
    batch_digests: list[str] = []
    strata: list[str] = []
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
            scope="synthetic-exp279-routing-development-counterfactual-delta-distill-v6-probe",
        )
        fold = _assign_probe_fold(replicate, probe_folds)
        event_rows.append(batch.surface_events.detach().cpu())
        variable_rows.append(batch.variable_states.detach().cpu())
        incidence_rows.append(batch.incidence.detach().cpu())
        target_rows.append(batch.targets.detach().cpu())
        fold_rows.append(torch.full((batch_size,), fold, dtype=torch.long))
        batch_digests.append(batch.digest)
        strata.append(stratum)

    events = torch.cat(event_rows, dim=0)
    variable_states = torch.cat(variable_rows, dim=0)
    incidence = torch.cat(incidence_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    fold_ids = torch.cat(fold_rows, dim=0)

    ledger = pair_audit["compute_ledger"]["hybrid"]
    stop_cost = float(ledger["stop_accounted_flops_per_episode"])
    branch_cost = float(ledger["branch_accounted_flops_per_episode"])
    primary_costs = cdd_costs(hidden_size=hidden_size, variables=variables, timesteps=timesteps)
    raw_costs = control_costs(hidden_size=hidden_size, variables=variables, timesteps=timesteps)

    fold_receipts = [
        fit_cdd_fold(
            arm=hybrid,
            surface_events=events,
            variable_states=variable_states,
            incidence=incidence,
            targets=targets,
            fold_ids=fold_ids,
            fold=fold,
            folds=probe_folds,
            hidden_size=hidden_size,
            distill_steps=distill_steps,
            distill_lr=distill_lr,
            selector_steps=selector_steps,
            selector_lr=selector_lr,
            probe_root_seed=probe_root,
            stop_cost=stop_cost,
            branch_cost=branch_cost,
            cdd_cost=float(primary_costs["total_cdd_inference_flops"]),
            control_cost=float(raw_costs["total_control_inference_flops"]),
        )
        for fold in range(probe_folds)
    ]

    primary_aggregate = aggregate_fold_direct_utility_metrics([row["primary"] for row in fold_receipts])
    control_aggregate = aggregate_fold_direct_utility_metrics([row["control"] for row in fold_receipts])
    primary = {
        "kind": PRIMARY_FAMILY,
        "costs": primary_costs,
        "folds": [{**row["primary"], "fit_supported": row["primary_fit"]["supported"], "distiller_frozen_before_selector": row["distiller_frozen_before_selector"]} for row in fold_receipts],
        "aggregate": primary_aggregate,
        "economically_routable": bool(primary_aggregate["direct_utility_improved"]),
        "authorizes_successor": bool(primary_aggregate["direct_utility_improved"]),
        "raw_scores_compared_across_folds": False,
    }
    control = {
        "kind": CONTROL_FAMILY,
        "costs": raw_costs,
        "folds": [{**row["control"], "fit_supported": row["control_fit"]["supported"]} for row in fold_receipts],
        "aggregate": control_aggregate,
        "economically_routable": bool(control_aggregate["direct_utility_improved"]),
        "authorizes_successor": False,
        "raw_scores_compared_across_folds": False,
    }
    probes = {PRIMARY_FAMILY: primary, CONTROL_FAMILY: control}
    classification = court_classification(primary)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "experiment_id": "EXP-279",
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "canonical_model_frozen_for_cdd": True,
        "data_boundary": {
            "training_rng_stream": "augmentation",
            "probe_rng_stream": "augmentation",
            "probe_root_seed": probe_root,
            "probe_root_independent_from_training_root": True,
            "evaluation_rng_stream_used": False,
            "evaluation_targets_used": False,
            "heldout_branch_hidden_used_for_features": False,
            "confirmatory_examples_used": False,
            "external_examples_used": False,
        },
        "world_geometry": {"batch_size": batch_size, "timesteps": timesteps, "variables": variables, "constraints": constraints, "d_model": d_model, "noise_std": float(noise_std)},
        "arm_geometry": {"hidden_size": hidden_size, "target_parameters": target_parameters},
        "route_config": {"canonical_threshold": float(route_threshold), "threshold_tuned": False},
        "canonical_training": {
            "replicates": train_replicates,
            "rng_stream": "augmentation",
            "batch_digests": training_batch_digests,
            "losses": training_losses,
            "frozen_hybrid_functional_state_digest": frozen_digest,
            "canonical_parameters_frozen_before_cdd": True,
        },
        "teacher_rule": {
            "teacher_target": "mean_detached_branch_state_minus_stop_state_over_variables",
            "teacher_uses_fit_partition_only": True,
            "heldout_teacher_delta_materialized_for_features": False,
            "teacher_training_flops_excluded_from_deployment_utility": True,
            "training_compute_match_claimed": False,
        },
        "probe_config": {
            "replicates": probe_replicates,
            "folds": probe_folds,
            "primary_family": PRIMARY_FAMILY,
            "descriptive_control_family": CONTROL_FAMILY,
            "control_can_authorize_successor": False,
            "distill_steps": distill_steps,
            "distill_lr": float(distill_lr),
            "selector_steps": selector_steps,
            "selector_lr": float(selector_lr),
            "fold_route_k_rule": "heldout_fold_true_rescue_count",
            "oracle_route_cardinality_is_deployable": False,
            "raw_scores_compared_across_folds": False,
            "raw_scores_exported": False,
        },
        "probe_dataset": {
            "batch_digests": batch_digests,
            "strata": strata,
            "episodes": int(events.shape[0]),
            "fold_episode_counts": {str(fold): int((fold_ids == fold).sum().item()) for fold in range(probe_folds)},
        },
        "compute_economics": {
            "stop_accounted_flops_per_episode": int(stop_cost),
            "branch_accounted_flops_per_episode": int(branch_cost),
            "source": "sealed_matched_arm_compute_ledger",
            "primary_cdd_costs": primary_costs,
            "descriptive_control_costs": raw_costs,
        },
        "cdd_cost_rule": {
            "metric": "direct_counterfactual_verified_solution_per_total_accounted_flop",
            "cross_fold_aggregation": "sum_solution_counts_and_total_accounted_flops",
            "cdd_inference_flops_charged_to_primary_utility": True,
            "cdd_cost_charged_on_stop_and_branch_paths": True,
            "false_positive_harms_counted_directly": True,
        },
        "probes": probes,
        "court_classification": classification,
        "successor_design_may_be_considered": classification == "CDD_DIRECT_UTILITY_POSITIVE",
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "resource_pair_audit_digest": canonical_sha256(pair_audit),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_artifact(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 V6 artifact: " + "; ".join(errors))
    return payload
