from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_representation_identifiability_v10_runner import (
    FIT_REPLICATES_PER_ROOT,
    FROZEN_GEOMETRY,
    FROZEN_OPTIMIZER,
    HELDOUT_REPLICATES_PER_ROOT,
    ROOT_PREFIX,
    build_root_schedule,
    run_v10_cric_shard,
)


def test_public_runner_signature_exposes_identity_only():
    assert list(inspect.signature(run_v10_cric_shard).parameters) == [
        "train_replicates",
        "canonical_index",
        "protocol_digest",
        "code_digest",
        "scientific_branch_head",
        "executed_commit",
    ]


def test_frozen_geometry_and_sample_contract():
    assert ROOT_PREFIX == "20260912-exp279-v10-cric-dev"
    assert FIT_REPLICATES_PER_ROOT == 128
    assert HELDOUT_REPLICATES_PER_ROOT == 128
    assert FROZEN_GEOMETRY == {
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
    assert FROZEN_OPTIMIZER == {"lr": 0.002, "weight_decay": 0.0}


def test_root_schedule_is_exact_and_fresh_per_cell():
    roots = build_root_schedule(60, 2)
    prefix = "20260912-exp279-v10-cric-dev/train-60/canonical-2"
    assert roots == {
        "training_root": f"{prefix}/train",
        "fit_roots": [f"{prefix}/fit-0", f"{prefix}/fit-1"],
        "heldout_roots": [f"{prefix}/heldout-0", f"{prefix}/heldout-1"],
    }
    assert set(roots["fit_roots"]).isdisjoint(roots["heldout_roots"])


@pytest.mark.parametrize("budget", [0, 59, 61, 119, 121])
def test_root_schedule_rejects_noncanonical_budget(budget):
    with pytest.raises(ValueError, match="train budget"):
        build_root_schedule(budget, 0)


@pytest.mark.parametrize("canonical_index", [-1, 4, 9])
def test_root_schedule_rejects_noncanonical_index(canonical_index):
    with pytest.raises(ValueError, match="canonical index"):
        build_root_schedule(60, canonical_index)


def test_runner_has_no_tuning_or_scientific_boundary_knobs():
    forbidden = {
        "root_seed",
        "fit_replicates",
        "heldout_replicates",
        "eval_replicates",
        "evaluation_start_replicate",
        "batch_size",
        "timesteps",
        "variables",
        "constraints",
        "d_model",
        "hidden_size",
        "route_threshold",
        "lr",
        "weight_decay",
        "k",
        "chunk_size",
        "threshold",
        "representation",
        "confirmatory",
        "challenge",
        "promotion",
    }
    assert forbidden.isdisjoint(inspect.signature(run_v10_cric_shard).parameters)
