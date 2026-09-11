from __future__ import annotations

from typing import Any

import torch

from .exp279_counterfactual_delta_distill_v6_court import run_court
from .exp279_counterfactual_delta_distill_v6_fit import fit_cdd_fold as _fit_cdd_fold
from .exp279_counterfactual_delta_distill_v6_primitives import (
    CONTROL_FAMILY,
    PRIMARY_FAMILY,
    PROBE_ROOT_SUFFIX,
    SCHEMA,
    cdd_costs as _cdd_costs,
    cheap_features as _cheap_features,
    pairwise_ranking_loss as _pairwise_ranking_loss,
    stop_and_branch_states as _stop_and_branch_states,
    teacher_delta as _teacher_delta,
)
from .exp279_counterfactual_delta_distill_v6_utility import (
    aggregate_fold_direct_utility_metrics as _aggregate_fold_direct_utility_metrics,
    fold_direct_utility_metrics,
)
from .exp279_counterfactual_delta_distill_v6_validation import validate_artifact


def _fold_direct_utility_metrics(
    scores: torch.Tensor,
    stop_exact: torch.Tensor,
    branch_exact: torch.Tensor,
    *,
    stop_accounted_flops_per_episode: int | float,
    branch_accounted_flops_per_episode: int | float,
    cdd_inference_flops_per_episode: int | float,
) -> dict[str, Any]:
    return fold_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=stop_accounted_flops_per_episode,
        branch_accounted_flops_per_episode=branch_accounted_flops_per_episode,
        inference_flops_per_episode=cdd_inference_flops_per_episode,
    )


def validate_exp279_counterfactual_delta_distill_v6(payload: dict[str, Any]) -> list[str]:
    return validate_artifact(payload)


def run_exp279_counterfactual_delta_distill_v6_court(**kwargs: Any) -> dict[str, Any]:
    return run_court(**kwargs)


__all__ = [
    "SCHEMA",
    "PRIMARY_FAMILY",
    "CONTROL_FAMILY",
    "PROBE_ROOT_SUFFIX",
    "_cheap_features",
    "_stop_and_branch_states",
    "_teacher_delta",
    "_cdd_costs",
    "_pairwise_ranking_loss",
    "_fold_direct_utility_metrics",
    "_aggregate_fold_direct_utility_metrics",
    "_fit_cdd_fold",
    "validate_exp279_counterfactual_delta_distill_v6",
    "run_exp279_counterfactual_delta_distill_v6_court",
]
