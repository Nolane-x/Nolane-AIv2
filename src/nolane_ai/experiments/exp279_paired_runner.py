from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer, functional_trainable_named_parameters
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes
from .exp279_routing_worlds import Exp279RoutingGenerator, STRATA
from .matched_routing_arms import (
    BranchOnlyArm,
    HybridRoutingArm,
    PropagationOnlyArm,
    audit_matched_exp279_arm_triplet,
    build_matched_exp279_arm_triplet,
)

SCHEMA = "NLM-EXP-279-PAIRED-DEV-EVAL-V1"
ARM_ORDER = ("propagation_only", "branch_only", "hybrid")
PRIMARY_METRIC = "verified_utility_per_accounted_flop_on_structure_dense_stratum"
PROTECTED_FLOOR = "hybrid >= best_simple - 0.01"
MULTIPLICITY_FAMILY = "PROPAGATION_ROUTING"
REPLAY_CONTRACT_SCHEMA = "NLM-EXP-279-DEVELOPMENT-REPLAY-CONTRACT-V1"
RESIDUAL_STATISTIC = "mean_predicted_episode_stop_failure_probability"
ROUTING_SUPERVISION = {
    "loss": "binary_cross_entropy",
    "weight": 1.0,
    "gradient_scope": "routing_head_only",
    "shared_backbone_receives_routing_loss_gradient": False,
    "decision_head_receives_routing_loss_gradient": False,
    "episode_targets": {
        "propagation_only": "arm_exact_failure",
        "branch_only": "arm_exact_failure",
        "hybrid": "branch_rescue_required",
    },
    "hybrid_route_teacher": {
        "positive": "stop_exact_failure_and_forced_branch_exact_success",
        "negative": "otherwise",
        "class_balance": "equal_positive_negative_mass_when_both_present",
        "decision_threshold_changed": False,
        "evaluation_targets_used_for_routing": False,
    },
    "hybrid_stop_path_supervision": {
        "loss": "cross_entropy",
        "final_path_weight": 0.5,
        "stop_path_weight": 0.5,
        "normalization": "weights_sum_to_one",
        "uses_same_paired_training_targets": True,
        "external_examples_added": False,
        "training_compute_match_claimed": False,
    },
    "hybrid_forced_branch_supervision": {
        "loss": "cross_entropy",
        "weight": 0.5,
        "purpose": "train_branch_expert_independent_of_current_route_gate",
        "uses_same_paired_training_targets": True,
        "external_examples_added": False,
        "training_only": True,
        "training_compute_match_claimed": False,
    },
    "development_targets_used": True,
    "evaluation_targets_used_for_routing": False,
}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _functional_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-279-FUNCTIONAL-STATE-V1\0")
    for name, parameter in functional_trainable_named_parameters(model):
        cpu = parameter.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "byteorder": tensor_byteorder(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(header)
        hasher.update(b"\0")
        hasher.update(tensor_raw_bytes(cpu))
    return hasher.hexdigest()


def _build_seeded_triplet(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
) -> tuple[PropagationOnlyArm, BranchOnlyArm, HybridRoutingArm, int]:
    seed = derive_stream_seed(root_seed, "EXP-279", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        propagation, branch, hybrid = build_matched_exp279_arm_triplet(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            route_threshold=route_threshold,
            device="cpu",
        )
    return propagation, branch, hybrid, seed


def _episode_failure_target(
    decision_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    predictions = decision_logits.detach().argmax(dim=-1)
    exact = (predictions == targets).all(dim=-1)
    return (~exact).to(dtype=decision_logits.dtype)


def _hybrid_branch_rescue_target(
    stop_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    stop_predictions = stop_logits.detach().argmax(dim=-1)
    branch_predictions = branch_logits.detach().argmax(dim=-1)
    stop_exact = (stop_predictions == targets).all(dim=-1)
    branch_exact = (branch_predictions == targets).all(dim=-1)
    return ((~stop_exact) & branch_exact).to(dtype=stop_logits.dtype)


def _balanced_binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if probabilities.shape != targets.shape:
        raise ValueError("routing probabilities and targets must have identical shape")
    positive = targets > 0.5
    negative = ~positive
    if bool(positive.any().item()) and bool(negative.any().item()):
        positive_loss = F.binary_cross_entropy(
            probabilities[positive], targets[positive]
        )
        negative_loss = F.binary_cross_entropy(
            probabilities[negative], targets[negative]
        )
        return 0.5 * (positive_loss + negative_loss)
    return F.binary_cross_entropy(probabilities, targets)


def _hybrid_stop_decision_logits(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    events, variables = arm._validate_common(surface_events, variable_states)
    checked_incidence = arm._validate_incidence(incidence, variables)
    propagated = arm._propagate(variables, checked_incidence)
    propagation_state = variables + propagated
    reclaimed = torch.tanh(arm.reclaimed_projection(propagation_state))
    stop_state = propagation_state + reclaimed
    return arm.decision_head(stop_state)


def _hybrid_forced_branch_decision_logits(
    arm: HybridRoutingArm,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    events, variables = arm._validate_common(surface_events, variable_states)
    checked_incidence = arm._validate_incidence(incidence, variables)
    propagated = arm._propagate(variables, checked_incidence)
    propagation_state = variables + propagated
    reclaimed = torch.tanh(arm.reclaimed_projection(propagation_state))
    branch_context = arm._branch_context(events).unsqueeze(1)
    branch_context = branch_context.expand(-1, variables.shape[1], -1)
    branch_state = propagation_state + torch.sigmoid(arm.mix_gate) * (
        branch_context + reclaimed
    )
    return arm.decision_head(branch_state)


def _train_step(
    arm: PropagationOnlyArm | BranchOnlyArm | HybridRoutingArm,
    optimizer: torch.optim.Optimizer,
    *,
    arm_id: str,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    arm.train()
    optimizer.zero_grad(set_to_none=True)
    if arm_id == "branch_only":
        output = arm(surface_events, variable_states)
    else:
        output = arm(surface_events, variable_states, incidence)

    final_decision_loss = F.cross_entropy(
        output.decision_logits.reshape(-1, 2),
        targets.reshape(-1),
    )
    decision_loss = final_decision_loss
    if arm_id == "hybrid":
        stop_logits = _hybrid_stop_decision_logits(
            arm,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
        )
        forced_branch_logits = _hybrid_forced_branch_decision_logits(
            arm,
            surface_events=surface_events,
            variable_states=variable_states,
            incidence=incidence,
        )
        stop_path = ROUTING_SUPERVISION["hybrid_stop_path_supervision"]
        final_path_weight = float(stop_path["final_path_weight"])
        stop_path_weight = float(stop_path["stop_path_weight"])
        stop_decision_loss = F.cross_entropy(
            stop_logits.reshape(-1, 2),
            targets.reshape(-1),
        )
        forced_branch = ROUTING_SUPERVISION["hybrid_forced_branch_supervision"]
        forced_branch_weight = float(forced_branch["weight"])
        forced_branch_decision_loss = F.cross_entropy(
            forced_branch_logits.reshape(-1, 2),
            targets.reshape(-1),
        )
        decision_loss = (
            final_path_weight * final_decision_loss
            + stop_path_weight * stop_decision_loss
            + forced_branch_weight * forced_branch_decision_loss
        )
        routing_target = _hybrid_branch_rescue_target(
            stop_logits,
            forced_branch_logits,
            targets,
        )
    else:
        routing_target = _episode_failure_target(output.decision_logits, targets)
    if arm_id == "hybrid":
        routing_loss = _balanced_binary_cross_entropy(
            output.residual_uncertainty,
            routing_target,
        )
    else:
        routing_loss = F.binary_cross_entropy(
            output.residual_uncertainty,
            routing_target,
        )
    routing_weight = float(ROUTING_SUPERVISION["weight"])
    routing_parameters = tuple(arm.routing_head.parameters())
    routing_gradients = torch.autograd.grad(
        routing_weight * routing_loss,
        routing_parameters,
        retain_graph=True,
    )
    decision_loss.backward()
    for parameter, routing_gradient in zip(routing_parameters, routing_gradients):
        if parameter.grad is None:
            parameter.grad = routing_gradient.detach().clone()
        else:
            parameter.grad.add_(routing_gradient)
    optimizer.step()
    loss_value = decision_loss.detach() + routing_weight * routing_loss.detach()
    return float(loss_value.item())


def _external_metrics(
    output: Any,
    targets: torch.Tensor,
    *,
    accounted_flops_per_episode: float,
) -> dict[str, float]:
    predictions = output.decision_logits.argmax(dim=-1)
    exact_per_episode = (predictions == targets).all(dim=-1)
    solution_rate = float(exact_per_episode.to(torch.float32).mean().item())
    accuracy = float((predictions == targets).to(torch.float32).mean().item())
    flops = max(float(accounted_flops_per_episode), 1.0)
    return {
        "verified_solution_rate": solution_rate,
        "verified_decision_accuracy": accuracy,
        "mean_verifier_confidence": float(output.verifier_confidence.mean().item()),
        "accounted_flops_per_episode": float(accounted_flops_per_episode),
        PRIMARY_METRIC: solution_rate / flops,
    }


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return sum(float(row[arm][metric]) for row in rows) / len(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_utilities = {arm: _mean(rows, arm, PRIMARY_METRIC) for arm in ARM_ORDER}
    mean_solutions = {arm: _mean(rows, arm, "verified_solution_rate") for arm in ARM_ORDER}
    best_simple_arm = (
        "propagation_only"
        if mean_utilities["propagation_only"] >= mean_utilities["branch_only"]
        else "branch_only"
    )
    best_simple_utility = mean_utilities[best_simple_arm]
    best_simple_solution = mean_solutions[best_simple_arm]
    relative_gain = (
        mean_utilities["hybrid"] - best_simple_utility
    ) / max(abs(best_simple_utility), 1e-12)

    by_stratum: dict[str, Any] = {}
    for stratum in STRATA:
        selected = [row for row in rows if row["stratum"] == stratum]
        if not selected:
            continue
        by_stratum[stratum] = {
            "n": len(selected),
            "mean_propagation_utility": _mean(selected, "propagation_only", PRIMARY_METRIC),
            "mean_branch_utility": _mean(selected, "branch_only", PRIMARY_METRIC),
            "mean_hybrid_utility": _mean(selected, "hybrid", PRIMARY_METRIC),
            "mean_propagation_solution_rate": _mean(
                selected, "propagation_only", "verified_solution_rate"
            ),
            "mean_branch_solution_rate": _mean(selected, "branch_only", "verified_solution_rate"),
            "mean_hybrid_solution_rate": _mean(selected, "hybrid", "verified_solution_rate"),
        }

    return {
        "n": len(rows),
        "structure_fit_strata": list(STRATA),
        "mean_propagation_utility": mean_utilities["propagation_only"],
        "mean_branch_utility": mean_utilities["branch_only"],
        "mean_hybrid_utility": mean_utilities["hybrid"],
        "mean_propagation_solution_rate": mean_solutions["propagation_only"],
        "mean_branch_solution_rate": mean_solutions["branch_only"],
        "mean_hybrid_solution_rate": mean_solutions["hybrid"],
        "best_simple_arm": best_simple_arm,
        "best_simple_mean_utility": best_simple_utility,
        "best_simple_mean_solution_rate": best_simple_solution,
        "hybrid_relative_utility_gain": relative_gain,
        "hybrid_minus_best_simple_verified_solution_rate": (
            mean_solutions["hybrid"] - best_simple_solution
        ),
        "by_stratum": by_stratum,
        "analysis_boundary": (
            "descriptive DEVELOPMENT aggregates only; no blocked/Holm confirmatory inference executed"
        ),
    }


def _replay_contract(payload: dict[str, Any]) -> dict[str, Any]:
    training = payload.get("training") or {}
    return {
        "schema": REPLAY_CONTRACT_SCHEMA,
        "protocol_digest": payload.get("protocol_digest"),
        "development_code_digest": payload.get("code_digest"),
        "root_seed": payload.get("root_seed"),
        "model_init_seed": payload.get("model_init_seed"),
        "arm_geometry": deepcopy(payload.get("arm_geometry") or {}),
        "world_geometry": deepcopy(payload.get("world_geometry") or {}),
        "route_config": deepcopy(payload.get("route_config") or {}),
        "training": {
            "rng_stream": training.get("rng_stream"),
            "start_replicate": training.get("start_replicate"),
            "replicates": training.get("replicates"),
            "strata": deepcopy(training.get("strata") or []),
            "paired_batch_digests": deepcopy(training.get("paired_batch_digests") or []),
            "optimizer": deepcopy(training.get("optimizer") or {}),
            "routing_supervision": deepcopy(training.get("routing_supervision") or {}),
            "losses": deepcopy(training.get("losses") or {}),
        },
        "final_state": deepcopy(payload.get("final_state") or {}),
        "resource_pair_audit_digest": (payload.get("resource_match") or {}).get(
            "pair_audit_digest"
        ),
        "declared_max_accounted_flops_per_episode": (payload.get("resource_match") or {}).get(
            "declared_max_accounted_flops_per_episode"
        ),
    }


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def validate_exp279_paired_development(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-279 paired development schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-279 paired development artifact cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 paired development artifact cannot promote a claim")
    for key in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(key) is not False:
            errors.append(f"EXP-279 development boundary requires {key}=false")
    if payload.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1":
        errors.append("EXP-279 protocol authority drift")
    if not payload.get("protocol_digest") or not payload.get("code_digest"):
        errors.append("EXP-279 paired development provenance is incomplete")
    if payload.get("arm_order") != list(ARM_ORDER):
        errors.append("EXP-279 arm ordering drift")

    root_seed = payload.get("root_seed")
    if not isinstance(root_seed, str) or not root_seed:
        errors.append("EXP-279 replay root seed is missing")
        expected_model_seed = None
    else:
        expected_model_seed = derive_stream_seed(root_seed, "EXP-279", 0, "model_init")
    if expected_model_seed is None or payload.get("model_init_seed") != expected_model_seed:
        errors.append("EXP-279 model-init seed does not match replay root seed")

    arm_geometry = payload.get("arm_geometry") or {}
    for key in ("d_model", "hidden_size", "target_parameters"):
        value = arm_geometry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"EXP-279 replay arm geometry {key} is invalid")
    try:
        arm_route_threshold = float(arm_geometry.get("route_threshold"))
    except (TypeError, ValueError):
        arm_route_threshold = -1.0
    if not 0.0 <= arm_route_threshold <= 1.0:
        errors.append("EXP-279 replay arm route threshold is invalid")

    world_geometry = payload.get("world_geometry") or {}
    for key in ("batch_size", "timesteps", "variables", "constraints", "d_model"):
        value = world_geometry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"EXP-279 replay world geometry {key} is invalid")
    if arm_geometry.get("d_model") != world_geometry.get("d_model"):
        errors.append("EXP-279 replay d_model geometry mismatch")
    try:
        noise_std = float(world_geometry.get("noise_std"))
    except (TypeError, ValueError):
        noise_std = -1.0
    if noise_std < 0.0 or not math.isfinite(noise_std):
        errors.append("EXP-279 replay noise geometry is invalid")
    if (
        isinstance(world_geometry.get("constraints"), int)
        and isinstance(world_geometry.get("variables"), int)
        and world_geometry["constraints"] > world_geometry["variables"]
    ):
        errors.append("EXP-279 replay constraints exceed variables")
    if (
        isinstance(world_geometry.get("timesteps"), int)
        and isinstance(world_geometry.get("constraints"), int)
        and world_geometry["timesteps"] < world_geometry["constraints"]
    ):
        errors.append("EXP-279 replay timesteps are shorter than constraints")

    initial = payload.get("initial_state") or {}
    initial_digests = [
        initial.get("propagation_only_digest"),
        initial.get("branch_only_digest"),
        initial.get("hybrid_digest"),
    ]
    if (
        initial.get("functional_digest_match") is not True
        or None in initial_digests
        or len(set(initial_digests)) != 1
    ):
        errors.append("EXP-279 matched arms must share identical functional initialization")

    resource = payload.get("resource_match") or {}
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "same_world_lineage",
        "compute_budget_closed",
    ):
        if resource.get(key) is not True:
            errors.append(f"EXP-279 resource match {key} is not closed")
    if resource.get("structure_fit_strata") != list(STRATA):
        errors.append("EXP-279 structure-fit strata drift")
    pair_audit = resource.get("pair_audit") or {}
    if pair_audit.get("schema") != "NLM-EXP-279-MATCHED-ROUTING-ARMS-DEV-V1":
        errors.append("EXP-279 matched triplet audit schema drift")
    if resource.get("pair_audit_digest") != canonical_sha256(pair_audit):
        errors.append("EXP-279 matched triplet audit digest mismatch")
    if pair_audit.get("inactive_excluded_reclaimed_parameters") != 0:
        errors.append("EXP-279 reclaimed capacity cannot be hidden in excluded reserve")
    if pair_audit.get("optimizer_visible_parameter_match") is not True:
        errors.append("EXP-279 optimizer-visible parameter match is not closed")
    declared_budget = float(
        resource.get("declared_max_accounted_flops_per_episode", 0.0) or 0.0
    )
    if declared_budget <= 0.0:
        errors.append("EXP-279 declared compute ceiling is missing")

    expected_receipt = {
        "artifact": "constraint_variable_incidence",
        "ground_truth": True,
        "delivered_to": ["propagation_only", "hybrid"],
        "withheld_from": ["branch_only"],
        "branch_only_received_incidence": False,
    }
    if payload.get("information_receipt") != expected_receipt:
        errors.append("EXP-279 information receipt drift")

    route_config = payload.get("route_config") or {}
    if route_config.get("strategy") != "propagation_then_branch_on_residual_uncertainty":
        errors.append("EXP-279 routing strategy drift")
    if route_config.get("residual_statistic") != RESIDUAL_STATISTIC:
        errors.append("EXP-279 residual routing statistic drift")
    if route_config.get("frozen_before_evaluation") is not True:
        errors.append("EXP-279 route threshold must be frozen before evaluation")
    try:
        threshold = float(route_config.get("threshold"))
    except (TypeError, ValueError):
        threshold = -1.0
    if not 0.0 <= threshold <= 1.0:
        errors.append("EXP-279 route threshold is invalid")
    if abs(arm_route_threshold - threshold) > 1e-12:
        errors.append("EXP-279 replay arm route threshold does not match route threshold")
    if pair_audit and abs(float(pair_audit.get("route_threshold", -2.0)) - threshold) > 1e-12:
        errors.append("EXP-279 route threshold does not match sealed pair audit")

    if payload.get("primary_endpoint") != {
        "metric": PRIMARY_METRIC,
        "direction": "higher",
        "mesi_relative_gain": 0.08,
    }:
        errors.append("EXP-279 primary endpoint or MESI drift")
    protected = payload.get("protected_endpoints") or {}
    if protected.get("verified_solution_rate_floor") != PROTECTED_FLOOR:
        errors.append("EXP-279 protected solution-rate floor drift")
    if payload.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        errors.append("EXP-279 multiplicity family drift")

    training = payload.get("training") or {}
    if training.get("rng_stream") != "augmentation":
        errors.append("EXP-279 training stream must be augmentation")
    if int(training.get("start_replicate", -1)) != 0:
        errors.append("EXP-279 training replicate lineage must start at 0")
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = list(training.get("paired_batch_digests") or [])
    if training.get("strata") != list(STRATA):
        errors.append("EXP-279 training strata drift")
    if training.get("routing_supervision") != ROUTING_SUPERVISION:
        errors.append("EXP-279 calibrated routing supervision contract drift")
    if (
        train_count <= 0
        or len(train_digests) != train_count
        or len(set(train_digests)) != len(train_digests)
    ):
        errors.append("EXP-279 training batch lineage is incomplete or non-unique")
    optimizer = training.get("optimizer") or {}
    if optimizer.get("type") != "AdamW":
        errors.append("EXP-279 replay optimizer type drift")
    try:
        optimizer_lr = float(optimizer.get("lr"))
        optimizer_weight_decay = float(optimizer.get("weight_decay"))
    except (TypeError, ValueError):
        optimizer_lr = -1.0
        optimizer_weight_decay = -1.0
    if optimizer_lr <= 0.0 or optimizer_weight_decay < 0.0:
        errors.append("EXP-279 replay optimizer contract invalid")
    losses = training.get("losses") or {}
    mean_losses = training.get("mean_losses") or {}
    if set(losses) != set(ARM_ORDER):
        errors.append("EXP-279 training loss lineage arm set mismatch")
    else:
        for arm in ARM_ORDER:
            values = list(losses.get(arm) or [])
            if len(values) != train_count or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in values
            ):
                errors.append(f"EXP-279 {arm} training loss lineage invalid")
                continue
            expected_mean = sum(float(value) for value in values) / len(values)
            try:
                observed_mean = float(mean_losses.get(arm))
            except (TypeError, ValueError):
                observed_mean = float("nan")
            if not math.isfinite(observed_mean) or abs(observed_mean - expected_mean) > 1e-12:
                errors.append(f"EXP-279 {arm} training loss mean mismatch")

    final_state = payload.get("final_state") or {}
    expected_final_keys = {
        "propagation_only_digest",
        "branch_only_digest",
        "hybrid_digest",
    }
    if set(final_state) != expected_final_keys or any(
        not _valid_digest(final_state.get(key)) for key in expected_final_keys
    ):
        errors.append("EXP-279 final state functional digests invalid")
    if payload.get("final_state_digest") != canonical_sha256(final_state):
        errors.append("EXP-279 final state digest mismatch")
    expected_replay_contract_digest = canonical_sha256(_replay_contract(payload))
    if payload.get("replay_contract_digest") != expected_replay_contract_digest:
        errors.append("EXP-279 replay contract digest mismatch")

    evaluation = payload.get("evaluation") or {}
    if evaluation.get("rng_stream") != "evaluation":
        errors.append("EXP-279 evaluation stream must be evaluation")
    eval_start = int(evaluation.get("start_replicate", -1))
    eval_count = int(evaluation.get("replicates", 0) or 0)
    rows = list(evaluation.get("per_replicate") or [])
    if eval_count < len(STRATA):
        errors.append("EXP-279 evaluation requires at least 3 replicates to cover all predeclared strata")
    if len(rows) != eval_count:
        errors.append("EXP-279 evaluation replicate count does not match raw lineage")
    if eval_start < train_count and eval_start + eval_count > 0:
        errors.append("EXP-279 training and evaluation replicate lineages overlap")

    if rows:
        expected_indexes = list(range(eval_start, eval_start + len(rows)))
        if [row.get("replicate") for row in rows] != expected_indexes:
            errors.append("EXP-279 evaluation replicate lineage is reordered or incomplete")
        expected_strata = [STRATA[offset % len(STRATA)] for offset in range(len(rows))]
        if [row.get("stratum") for row in rows] != expected_strata:
            errors.append("EXP-279 evaluation stratum ordering or membership drift")
        digests = [row.get("paired_batch_digest") for row in rows]
        if any(not digest for digest in digests) or len(set(digests)) != len(digests):
            errors.append("EXP-279 evaluation paired batch digests must be non-empty and unique")

        ledger = pair_audit.get("compute_ledger") or {}
        hybrid_ledger = ledger.get("hybrid") or {}
        stop_cost = float(
            hybrid_ledger.get("stop_accounted_flops_per_episode", 0.0) or 0.0
        )
        branch_cost = float(
            hybrid_ledger.get("branch_accounted_flops_per_episode", 0.0) or 0.0
        )
        fixed_costs = {
            "propagation_only": float(
                (ledger.get("propagation_only") or {}).get(
                    "accounted_flops_per_episode", 0.0
                )
                or 0.0
            ),
            "branch_only": float(
                (ledger.get("branch_only") or {}).get(
                    "accounted_flops_per_episode", 0.0
                )
                or 0.0
            ),
        }
        for row in rows:
            if row.get("world_pairing_closed") is not True:
                errors.append("EXP-279 per-replicate world pairing is not closed")
            receipt = row.get("hybrid_route_receipt") or {}
            mask = list(receipt.get("branch_route_mask") or [])
            batch_size = int(receipt.get("batch_size", 0) or 0)
            routed = int(receipt.get("routed_episodes", -1))
            route_fraction = float(receipt.get("route_fraction", -1.0) or 0.0)
            if abs(float(receipt.get("threshold", -2.0)) - threshold) > 1e-12:
                errors.append("EXP-279 per-replicate route threshold drift")
            if batch_size <= 0 or len(mask) != batch_size:
                errors.append("EXP-279 hybrid route receipt batch shape mismatch")
            if routed != sum(bool(item) for item in mask):
                errors.append("EXP-279 hybrid route receipt routed count mismatch")
            expected_fraction = routed / batch_size if batch_size > 0 else -1.0
            if abs(route_fraction - expected_fraction) > 1e-12:
                errors.append("EXP-279 hybrid route fraction mismatch")
            expected_hybrid_cost = stop_cost + expected_fraction * (
                branch_cost - stop_cost
            )
            charged = float(
                receipt.get("charged_accounted_flops_per_episode", -1.0) or -1.0
            )
            if abs(charged - expected_hybrid_cost) > 1e-9:
                errors.append("EXP-279 hybrid path-dependent cost receipt mismatch")

            for arm in ARM_ORDER:
                metrics = row.get(arm) or {}
                flops = float(metrics.get("accounted_flops_per_episode", 0.0) or 0.0)
                if flops <= 0.0 or (
                    declared_budget > 0.0 and flops > declared_budget + 1e-9
                ):
                    errors.append(f"EXP-279 {arm} accounted FLOPs violate declared ceiling")
                if arm in fixed_costs and abs(flops - fixed_costs[arm]) > 1e-9:
                    errors.append(f"EXP-279 {arm} charged FLOPs drift from matched audit")
                if arm == "hybrid" and abs(flops - expected_hybrid_cost) > 1e-9:
                    errors.append("EXP-279 hybrid metric cost does not match routing receipt")
                if PRIMARY_METRIC not in metrics:
                    errors.append(f"EXP-279 {arm} primary utility is missing")
                else:
                    solution = float(metrics.get("verified_solution_rate", 0.0))
                    expected_utility = solution / max(flops, 1.0)
                    if abs(float(metrics[PRIMARY_METRIC]) - expected_utility) > 1e-12:
                        errors.append(
                            f"EXP-279 {arm} primary utility is inconsistent with charged FLOPs"
                        )

        aggregate = evaluation.get("aggregate") or {}
        if int(aggregate.get("n", 0) or 0) != len(rows):
            errors.append("EXP-279 aggregate n mismatch")
        if aggregate.get("structure_fit_strata") != list(STRATA):
            errors.append("EXP-279 aggregate strata drift")
        if set((aggregate.get("by_stratum") or {}).keys()) != set(STRATA):
            errors.append("EXP-279 aggregate strata coverage is incomplete")
        expected = _aggregate_rows(rows)
        numeric_keys = (
            "mean_propagation_utility",
            "mean_branch_utility",
            "mean_hybrid_utility",
            "mean_propagation_solution_rate",
            "mean_branch_solution_rate",
            "mean_hybrid_solution_rate",
            "best_simple_mean_utility",
            "best_simple_mean_solution_rate",
            "hybrid_relative_utility_gain",
            "hybrid_minus_best_simple_verified_solution_rate",
        )
        if aggregate.get("best_simple_arm") != expected["best_simple_arm"]:
            errors.append("EXP-279 best simpler arm aggregate mismatch")
        for key in numeric_keys:
            if abs(float(aggregate.get(key, 0.0)) - float(expected[key])) > 1e-12:
                errors.append(f"EXP-279 aggregate {key} mismatch")
        for stratum in STRATA:
            observed_stratum = (aggregate.get("by_stratum") or {}).get(stratum) or {}
            expected_stratum = expected["by_stratum"].get(stratum) or {}
            if observed_stratum.keys() != expected_stratum.keys():
                errors.append(f"EXP-279 aggregate {stratum} fields mismatch")
                continue
            for key, expected_value in expected_stratum.items():
                observed_value = observed_stratum.get(key)
                if isinstance(expected_value, float):
                    if abs(float(observed_value) - expected_value) > 1e-12:
                        errors.append(f"EXP-279 aggregate {stratum} {key} mismatch")
                elif observed_value != expected_value:
                    errors.append(f"EXP-279 aggregate {stratum} {key} mismatch")

    if (
        payload.get("artifact_digest") not in (None, "")
        and payload.get("artifact_digest") != _artifact_digest(payload)
    ):
        errors.append("EXP-279 paired artifact digest mismatch")
    return errors


def run_exp279_paired_development(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
    train_replicates: int,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
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
        eval_replicates,
        batch_size,
        timesteps,
        variables,
        constraints,
    ) <= 0:
        raise ValueError("EXP-279 execution counts and dimensions must be positive")
    if eval_replicates < len(STRATA):
        raise ValueError(
            "EXP-279 development evaluation requires at least 3 replicates to cover all predeclared strata"
        )
    if constraints > variables or timesteps < constraints:
        raise ValueError("EXP-279 world geometry is invalid")
    if eval_start_replicate < train_replicates:
        raise ValueError("training and evaluation replicate lineages must be disjoint")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("route_threshold must be within [0, 1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("EXP-279 noise/optimizer parameters are invalid")

    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=route_threshold,
    )
    initial = {
        "propagation_only_digest": _functional_state_digest(propagation),
        "branch_only_digest": _functional_state_digest(branch),
        "hybrid_digest": _functional_state_digest(hybrid),
    }
    initial["functional_digest_match"] = len(set(initial.values())) == 1
    if initial["functional_digest_match"] is not True:
        raise RuntimeError(
            "EXP-279 matched triplet does not share identical functional initialization"
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
    if not all(
        pair_audit.get(key) is True
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "reclaimed_parameter_assignment_closed",
            "compute_budget_closed",
        )
    ):
        raise RuntimeError(
            "EXP-279 matched resource contract did not close before paired execution"
        )

    ledgers = pair_audit["compute_ledger"]
    prop_cost = float(
        ledgers["propagation_only"]["accounted_flops_per_episode"]
    )
    branch_cost = float(ledgers["branch_only"]["accounted_flops_per_episode"])
    hybrid_stop_cost = float(
        ledgers["hybrid"]["stop_accounted_flops_per_episode"]
    )
    hybrid_branch_cost = float(
        ledgers["hybrid"]["branch_accounted_flops_per_episode"]
    )

    generator = Exp279RoutingGenerator(root_seed=root_seed)
    optimizers = {
        "propagation_only": build_functional_optimizer(
            propagation, lr=lr, weight_decay=weight_decay
        ),
        "branch_only": build_functional_optimizer(
            branch, lr=lr, weight_decay=weight_decay
        ),
        "hybrid": build_functional_optimizer(
            hybrid, lr=lr, weight_decay=weight_decay
        ),
    }
    arms = {
        "propagation_only": propagation,
        "branch_only": branch,
        "hybrid": hybrid,
    }

    training_digests: list[str] = []
    training_losses: dict[str, list[float]] = {arm: [] for arm in ARM_ORDER}
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
        )
        training_digests.append(batch.digest)
        for arm_id in ARM_ORDER:
            training_losses[arm_id].append(
                _train_step(
                    arms[arm_id],
                    optimizers[arm_id],
                    arm_id=arm_id,
                    surface_events=batch.surface_events,
                    variable_states=batch.variable_states,
                    incidence=batch.incidence,
                    targets=batch.targets,
                )
            )

    final_state = {
        "propagation_only_digest": _functional_state_digest(propagation),
        "branch_only_digest": _functional_state_digest(branch),
        "hybrid_digest": _functional_state_digest(hybrid),
    }

    for arm in arms.values():
        arm.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(eval_replicates):
            replicate = eval_start_replicate + offset
            stratum = STRATA[offset % len(STRATA)]
            batch = generator.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                timesteps=timesteps,
                variables=variables,
                constraints=constraints,
                d_model=d_model,
                noise_std=noise_std,
                rng_stream="evaluation",
                stratum=stratum,
            )
            prop_output = propagation(
                batch.surface_events, batch.variable_states, batch.incidence
            )
            branch_output = branch(batch.surface_events, batch.variable_states)
            hybrid_output = hybrid(
                batch.surface_events, batch.variable_states, batch.incidence
            )

            route_mask = [
                bool(item)
                for item in hybrid_output.branch_route_mask.detach().cpu().tolist()
            ]
            routed = sum(route_mask)
            route_fraction = routed / batch_size
            hybrid_cost = hybrid_stop_cost + route_fraction * (
                hybrid_branch_cost - hybrid_stop_cost
            )
            route_receipt = {
                "strategy": "propagation_then_branch_on_residual_uncertainty",
                "threshold": float(route_threshold),
                "batch_size": batch_size,
                "branch_route_mask": route_mask,
                "routed_episodes": routed,
                "route_fraction": route_fraction,
                "mean_residual_uncertainty": float(
                    hybrid_output.residual_uncertainty.mean().item()
                ),
                "stop_accounted_flops_per_episode": hybrid_stop_cost,
                "branch_accounted_flops_per_episode": hybrid_branch_cost,
                "charged_accounted_flops_per_episode": hybrid_cost,
            }
            rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "paired_batch_digest": batch.digest,
                    "world_pairing_closed": True,
                    "propagation_only": _external_metrics(
                        prop_output,
                        batch.targets,
                        accounted_flops_per_episode=prop_cost,
                    ),
                    "branch_only": _external_metrics(
                        branch_output,
                        batch.targets,
                        accounted_flops_per_episode=branch_cost,
                    ),
                    "hybrid": _external_metrics(
                        hybrid_output,
                        batch.targets,
                        accounted_flops_per_episode=hybrid_cost,
                    ),
                    "hybrid_route_receipt": route_receipt,
                }
            )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "experiment_id": "EXP-279",
        "arm_order": list(ARM_ORDER),
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "arm_geometry": {
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
            "route_threshold": float(route_threshold),
        },
        "world_geometry": {
            "batch_size": batch_size,
            "timesteps": timesteps,
            "variables": variables,
            "constraints": constraints,
            "d_model": d_model,
            "noise_std": float(noise_std),
        },
        "initial_state": initial,
        "final_state": final_state,
        "final_state_digest": canonical_sha256(final_state),
        "information_receipt": {
            "artifact": "constraint_variable_incidence",
            "ground_truth": True,
            "delivered_to": ["propagation_only", "hybrid"],
            "withheld_from": ["branch_only"],
            "branch_only_received_incidence": False,
        },
        "route_config": {
            "strategy": "propagation_then_branch_on_residual_uncertainty",
            "threshold": float(route_threshold),
            "residual_statistic": RESIDUAL_STATISTIC,
            "frozen_before_evaluation": True,
        },
        "primary_endpoint": {
            "metric": PRIMARY_METRIC,
            "direction": "higher",
            "mesi_relative_gain": 0.08,
        },
        "protected_endpoints": {
            "verified_solution_rate_floor": PROTECTED_FLOOR,
        },
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "analysis_method_boundary": (
            "blocked paired/Holm confirmatory inference intentionally not executed in DEVELOPMENT"
        ),
        "resource_match": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "active_functional_parameter_match": True,
            "reclaimed_parameter_assignment_closed": True,
            "same_world_lineage": True,
            "compute_budget_closed": True,
            "structure_fit_strata": list(STRATA),
            "declared_max_accounted_flops_per_episode": int(
                pair_audit["declared_max_accounted_flops_per_episode"]
            ),
            "pair_audit": pair_audit,
            "pair_audit_digest": canonical_sha256(pair_audit),
        },
        "training": {
            "rng_stream": "augmentation",
            "start_replicate": 0,
            "replicates": train_replicates,
            "strata": list(STRATA),
            "paired_batch_digests": training_digests,
            "optimizer": {
                "type": "AdamW",
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            },
            "routing_supervision": deepcopy(ROUTING_SUPERVISION),
            "losses": training_losses,
            "mean_losses": {
                arm: sum(loss_values) / len(loss_values)
                for arm, loss_values in training_losses.items()
            },
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": eval_start_replicate,
            "replicates": eval_replicates,
            "per_replicate": rows,
            "aggregate": _aggregate_rows(rows),
        },
        "remaining_blockers": [
            "confirmatory sample-size and blocked/Holm analysis freeze remain open",
            "confirmatory-open execution remains unrun",
            "post-freeze challenge randomness remains unmaterialized",
        ],
        "replay_contract_digest": "",
        "artifact_digest": "",
    }
    payload["replay_contract_digest"] = canonical_sha256(_replay_contract(payload))
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp279_paired_development(payload)
    if errors:
        raise RuntimeError(
            "invalid EXP-279 paired development artifact: " + "; ".join(errors)
        )
    return payload