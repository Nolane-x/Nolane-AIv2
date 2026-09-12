from __future__ import annotations

from typing import Any

import torch

V9_MEAN_BASELINE = "V9_MEAN_BASELINE"
FULL_PREBRANCH_STATE = "FULL_PREBRANCH_STATE"
EXECUTION_FRONTIER_STATE = "EXECUTION_FRONTIER_STATE"
REPRESENTATION_VIEWS = (
    V9_MEAN_BASELINE,
    FULL_PREBRANCH_STATE,
    EXECUTION_FRONTIER_STATE,
)
REPRESENTATION_DIMENSIONS = {
    V9_MEAN_BASELINE: 144,
    FULL_PREBRANCH_STATE: 480,
    EXECUTION_FRONTIER_STATE: 312,
}

RESCUE = 0
HARM = 1
BOTH_SUCCESS = 2
BOTH_FAILURE = 3

REPRESENTATION_ECONOMICALLY_IDENTIFIABLE = "REPRESENTATION_ECONOMICALLY_IDENTIFIABLE"
REPRESENTATION_PARTIAL = "REPRESENTATION_PARTIAL"
REPRESENTATION_NOT_IDENTIFIABLE = "REPRESENTATION_NOT_IDENTIFIABLE"
REPRESENTATION_RECURRENTLY_IDENTIFIABLE = "REPRESENTATION_RECURRENTLY_IDENTIFIABLE"
REPRESENTATION_INTERMITTENT = "REPRESENTATION_INTERMITTENT"
CROSS_BUDGET_IDENTIFIABLE = "CROSS_BUDGET_IDENTIFIABLE"
CROSS_BUDGET_INTERMITTENT = "CROSS_BUDGET_INTERMITTENT"
CROSS_BUDGET_NOT_IDENTIFIABLE = "CROSS_BUDGET_NOT_IDENTIFIABLE"


def extract_representation_views(
    arm: Any,
    *,
    surface_events: torch.Tensor,
    variable_states: torch.Tensor,
    incidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    events, variables = arm._validate_common(surface_events, variable_states)
    checked_incidence = arm._validate_incidence(incidence, variables)
    propagated = arm._propagate(variables, checked_incidence)
    propagation_state = variables + propagated
    reclaimed = torch.tanh(arm.reclaimed_projection(propagation_state))
    stop_state = propagation_state + reclaimed
    routing_logits = arm.routing_head(propagation_state).squeeze(-1)
    stop_logits = arm.decision_head(stop_state)
    stop_verifier_logits = arm.verifier_head(stop_state).squeeze(-1)

    mean_baseline = torch.cat(
        (
            propagation_state.mean(dim=1),
            events.mean(dim=1),
            events[:, -1] - events[:, 0],
        ),
        dim=1,
    )
    full_prebranch = torch.cat(
        (events.flatten(start_dim=1), propagation_state.flatten(start_dim=1)),
        dim=1,
    )
    frontier = torch.cat(
        (
            stop_state.flatten(start_dim=1),
            stop_logits.flatten(start_dim=1),
            stop_verifier_logits,
            routing_logits,
        ),
        dim=1,
    )
    views = {
        V9_MEAN_BASELINE: mean_baseline,
        FULL_PREBRANCH_STATE: full_prebranch,
        EXECUTION_FRONTIER_STATE: frontier,
    }
    for name, features in views.items():
        expected = REPRESENTATION_DIMENSIONS[name]
        if features.ndim != 2 or features.shape[1] != expected:
            raise RuntimeError(
                f"V10 representation {name} must be [batch,{expected}], got {tuple(features.shape)}"
            )
    return views


def counterfactual_action_labels(
    stop_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stop_logits.shape != branch_logits.shape:
        raise ValueError("V10 stop and branch logits must have identical shape")
    if stop_logits.ndim != 3 or stop_logits.shape[-1] != 2:
        raise ValueError("V10 logits must be [batch,variables,2]")
    if tuple(targets.shape) != tuple(stop_logits.shape[:-1]):
        raise ValueError("V10 target shape mismatch")

    stop_exact = (stop_logits.argmax(dim=-1) == targets).all(dim=-1)
    branch_exact = (branch_logits.argmax(dim=-1) == targets).all(dim=-1)
    rescue = (~stop_exact) & branch_exact
    harm = stop_exact & (~branch_exact)
    both_success = stop_exact & branch_exact
    both_failure = (~stop_exact) & (~branch_exact)

    outcomes = torch.empty(stop_exact.shape[0], dtype=torch.int64, device=stop_logits.device)
    outcomes[rescue] = RESCUE
    outcomes[harm] = HARM
    outcomes[both_success] = BOTH_SUCCESS
    outcomes[both_failure] = BOTH_FAILURE
    if not bool((rescue | harm | both_success | both_failure).all().item()):
        raise RuntimeError("V10 counterfactual outcome partition did not close")
    labels = rescue.to(dtype=torch.int64)
    return labels, outcomes


def fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError("V10 fit features must be a non-empty rank-2 tensor")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def exact_1nn_predict(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    query_features: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    if fit_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("V10 1-NN features must be rank-2")
    if fit_features.shape[0] <= 0 or query_features.shape[0] <= 0:
        raise ValueError("V10 1-NN requires non-empty fit and query sets")
    if fit_features.shape[1] != query_features.shape[1]:
        raise ValueError("V10 1-NN feature dimensions must match")
    if fit_labels.ndim != 1 or fit_labels.shape[0] != fit_features.shape[0]:
        raise ValueError("V10 1-NN labels must align with fit rows")
    if chunk_size <= 0:
        raise ValueError("V10 1-NN chunk_size must be positive")
    if not bool(((fit_labels == 0) | (fit_labels == 1)).all().item()):
        raise ValueError("V10 1-NN labels must be binary")

    mean, std = fit_standardizer(fit_features)
    fit = (fit_features - mean) / std
    query = (query_features - mean) / std
    fit_sq = (fit * fit).sum(dim=1).unsqueeze(0)
    fit_t = fit.transpose(0, 1)
    predictions: list[torch.Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start : start + chunk_size]
        q_sq = (q * q).sum(dim=1, keepdim=True)
        distances = q_sq + fit_sq - 2.0 * (q @ fit_t)
        nearest = distances.argmin(dim=1)
        predictions.append(fit_labels.index_select(0, nearest))
    return torch.cat(predictions, dim=0)


def direct_policy_metrics(
    *,
    predictions: torch.Tensor,
    outcomes: torch.Tensor,
    stop_cost: float,
    branch_cost: float,
) -> dict[str, Any]:
    if predictions.ndim != 1 or outcomes.ndim != 1 or predictions.shape != outcomes.shape:
        raise ValueError("V10 policy predictions/outcomes must be aligned rank-1 tensors")
    if predictions.numel() <= 0:
        raise ValueError("V10 policy metrics require episodes")
    if stop_cost <= 0.0 or branch_cost < stop_cost:
        raise ValueError("V10 requires branch_cost >= stop_cost > 0")
    if not bool(((predictions == 0) | (predictions == 1)).all().item()):
        raise ValueError("V10 policy predictions must be binary")
    if not bool(((outcomes >= RESCUE) & (outcomes <= BOTH_FAILURE)).all().item()):
        raise ValueError("V10 outcome code outside frozen partition")

    routed = predictions == 1
    stop_success = (outcomes == HARM) | (outcomes == BOTH_SUCCESS)
    branch_success = (outcomes == RESCUE) | (outcomes == BOTH_SUCCESS)
    realized_success = torch.where(routed, branch_success, stop_success)

    def count(mask: torch.Tensor) -> int:
        return int(mask.to(torch.int64).sum().item())

    episodes = int(predictions.numel())
    route_count = count(routed)
    selected_rescues = count(routed & (outcomes == RESCUE))
    selected_harms = count(routed & (outcomes == HARM))
    selected_both_success = count(routed & (outcomes == BOTH_SUCCESS))
    selected_both_failure = count(routed & (outcomes == BOTH_FAILURE))
    raw_rescues = count(outcomes == RESCUE)
    policy_solutions = count(realized_success)
    stop_successes = count(stop_success)
    branch_successes = count(branch_success)
    policy_flops = (episodes - route_count) * float(stop_cost) + route_count * float(branch_cost)
    stop_flops = episodes * float(stop_cost)
    branch_flops = episodes * float(branch_cost)
    raw_prevalence = raw_rescues / episodes
    selected_prevalence = selected_rescues / route_count if route_count else 0.0
    enrichment = (
        selected_prevalence / raw_prevalence
        if raw_prevalence > 0.0
        else 0.0
    )
    return {
        "episodes": episodes,
        "route_count": route_count,
        "route_fraction": route_count / episodes,
        "policy_solutions": policy_solutions,
        "policy_flops": policy_flops,
        "stop_successes": stop_successes,
        "branch_successes": branch_successes,
        "stop_flops": stop_flops,
        "branch_flops": branch_flops,
        "policy_utility": policy_solutions / policy_flops,
        "stop_utility": stop_successes / stop_flops,
        "branch_utility": branch_successes / branch_flops,
        "selected_rescues": selected_rescues,
        "selected_harms": selected_harms,
        "selected_both_success": selected_both_success,
        "selected_both_failure": selected_both_failure,
        "raw_rescues": raw_rescues,
        "raw_rescue_prevalence": raw_prevalence,
        "selected_rescue_prevalence": selected_prevalence,
        "rescue_enrichment": enrichment,
        "provenance_closed": True,
    }


def _derived_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    episodes = int(metrics["episodes"])
    route_count = int(metrics["route_count"])
    if episodes <= 0 or not 0 <= route_count <= episodes:
        raise ValueError("V10 invalid episode/route counts")
    policy_flops = float(metrics["policy_flops"])
    stop_flops = float(metrics["stop_flops"])
    branch_flops = float(metrics["branch_flops"])
    if min(policy_flops, stop_flops, branch_flops) <= 0.0:
        raise ValueError("V10 utility denominators must be positive")
    raw_rescues = int(metrics["raw_rescues"])
    selected_rescues = int(metrics["selected_rescues"])
    raw_prevalence = raw_rescues / episodes
    selected_prevalence = selected_rescues / route_count if route_count else 0.0
    return {
        "route_fraction": route_count / episodes,
        "policy_utility": int(metrics["policy_solutions"]) / policy_flops,
        "stop_utility": int(metrics["stop_successes"]) / stop_flops,
        "branch_utility": int(metrics["branch_successes"]) / branch_flops,
        "raw_rescue_prevalence": raw_prevalence,
        "selected_rescue_prevalence": selected_prevalence,
    }


def classify_representation_root(metrics: dict[str, Any]) -> str:
    if metrics.get("provenance_closed") is not True:
        return REPRESENTATION_NOT_IDENTIFIABLE
    derived = _derived_metrics(metrics)
    beats_baselines = (
        derived["policy_utility"] > derived["stop_utility"]
        and derived["policy_utility"] > derived["branch_utility"]
    )
    route_is_selective = 0.0 < derived["route_fraction"] < 1.0
    rescue_dominance = int(metrics["selected_rescues"]) > int(metrics["selected_harms"])
    enriched = derived["selected_rescue_prevalence"] > derived["raw_rescue_prevalence"]
    if beats_baselines and route_is_selective and rescue_dominance and enriched:
        return REPRESENTATION_ECONOMICALLY_IDENTIFIABLE
    if beats_baselines:
        return REPRESENTATION_PARTIAL
    return REPRESENTATION_NOT_IDENTIFIABLE


def _pool_root_metrics(root_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(root_metrics) != 4:
        raise ValueError("V10 budget reduction requires exactly four canonical roots")
    additive = (
        "episodes",
        "route_count",
        "policy_solutions",
        "policy_flops",
        "stop_successes",
        "branch_successes",
        "stop_flops",
        "branch_flops",
        "selected_rescues",
        "selected_harms",
        "selected_both_success",
        "selected_both_failure",
        "raw_rescues",
    )
    pooled: dict[str, Any] = {key: sum(row[key] for row in root_metrics) for key in additive}
    pooled["provenance_closed"] = all(row.get("provenance_closed") is True for row in root_metrics)
    pooled.update(_derived_metrics(pooled))
    return pooled


def classify_representation_budget(
    root_metrics: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    pooled = _pool_root_metrics(root_metrics)
    root_classes = [classify_representation_root(row) for row in root_metrics]
    derived = _derived_metrics(pooled)
    pooled_beats = (
        derived["policy_utility"] > derived["stop_utility"]
        and derived["policy_utility"] > derived["branch_utility"]
    )
    pooled_dominance = int(pooled["selected_rescues"]) > int(pooled["selected_harms"])
    pooled_enriched = derived["selected_rescue_prevalence"] > derived["raw_rescue_prevalence"]
    all_roots = all(cls == REPRESENTATION_ECONOMICALLY_IDENTIFIABLE for cls in root_classes)
    if all_roots and pooled_beats and pooled_dominance and pooled_enriched and pooled["provenance_closed"]:
        classification = REPRESENTATION_RECURRENTLY_IDENTIFIABLE
    elif pooled_beats:
        classification = REPRESENTATION_INTERMITTENT
    else:
        classification = REPRESENTATION_NOT_IDENTIFIABLE
    pooled["root_classifications"] = root_classes
    return classification, pooled


def classify_cross_budget(
    budget_classifications: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if set(budget_classifications) != {"60", "120"}:
        raise ValueError("V10 cross classification requires string budgets '60' and '120'")
    cross: dict[str, str] = {}
    for view in REPRESENTATION_VIEWS:
        classes = [budget_classifications[budget][view] for budget in ("60", "120")]
        if all(cls == REPRESENTATION_RECURRENTLY_IDENTIFIABLE for cls in classes):
            cross[view] = CROSS_BUDGET_IDENTIFIABLE
        elif all(
            cls in {REPRESENTATION_RECURRENTLY_IDENTIFIABLE, REPRESENTATION_INTERMITTENT}
            for cls in classes
        ) and any(cls == REPRESENTATION_INTERMITTENT for cls in classes):
            cross[view] = CROSS_BUDGET_INTERMITTENT
        else:
            cross[view] = CROSS_BUDGET_NOT_IDENTIFIABLE

    if cross[EXECUTION_FRONTIER_STATE] == CROSS_BUDGET_IDENTIFIABLE:
        decision = "FRONTIER_SIGNAL_IDENTIFIED"
        successor = True
        authorized = EXECUTION_FRONTIER_STATE
        scope = "DESIGN_FRONTIER_ROUTER_MECHANISM_COURT_ONLY"
    elif cross[FULL_PREBRANCH_STATE] == CROSS_BUDGET_IDENTIFIABLE:
        decision = "FULL_PREBRANCH_SIGNAL_IDENTIFIED"
        successor = True
        authorized = FULL_PREBRANCH_STATE
        scope = "DESIGN_PREBRANCH_ROUTER_MECHANISM_COURT_ONLY"
    elif cross[V9_MEAN_BASELINE] == CROSS_BUDGET_IDENTIFIABLE:
        decision = "MEAN_CONTROL_ONLY_SIGNAL"
        successor = False
        authorized = "NONE"
        scope = "REVIEW_CONTROL_ANOMALY_ONLY"
    else:
        decision = "REPRESENTATION_SIGNAL_NOT_ESTABLISHED"
        successor = False
        authorized = "NONE"
        scope = "DESIGN_REPRESENTATION_OBJECTIVE_RESEARCH_ONLY"

    return {
        "representation_cross_classifications": cross,
        "decision": decision,
        "successor_design_authorized": successor,
        "authorized_representation_view": authorized,
        "authorization_scope": scope,
        "mechanism_successor_authorized": False,
    }
