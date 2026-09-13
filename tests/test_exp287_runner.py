from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from nolane_ai.experiments.exp287_conflict_worlds import Exp287ConflictGenerator
from nolane_ai.experiments.matched_conflict_localizer_arms import ORACLE_MODE
from nolane_ai.experiments.exp287_learned_conflict_localization import (
    ESTABLISHED,
    NOT_ESTABLISHED,
    ORACLE_NOT_REPLICATED,
    build_seeded_exp287_model,
    classify_exp287_root_metrics,
    compute_exp287_training_loss,
    evaluate_exp287_batch_same_weights,
    model_state_digest,
)


ROOT_PREFIX = "20260913-exp287-learned-conflict-localization-v1-dev"


def _batch(root: str = f"{ROOT_PREFIX}::0", replicate: int = 10000):
    return Exp287ConflictGenerator(root_seed=root).make_batch(
        replicate=replicate,
        batch_size=2,
        timesteps=4,
        variables=8,
        decoys=3,
        d_model=64,
        noise_std=0.05,
        rng_stream="evaluation",
    )


def test_seeded_model_is_deterministic_per_canonical_root() -> None:
    first, first_seed = build_seeded_exp287_model(
        root_seed=f"{ROOT_PREFIX}::2",
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    second, second_seed = build_seeded_exp287_model(
        root_seed=f"{ROOT_PREFIX}::2",
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    assert first_seed == second_seed
    assert model_state_digest(first) == model_state_digest(second)


def test_training_loss_is_exact_unweighted_sum_of_three_preregistered_terms() -> None:
    model, _ = build_seeded_exp287_model(
        root_seed=f"{ROOT_PREFIX}::0",
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    batch = Exp287ConflictGenerator(root_seed=f"{ROOT_PREFIX}::0").make_batch(
        replicate=0,
        batch_size=2,
        timesteps=4,
        variables=8,
        decoys=3,
        d_model=64,
        noise_std=0.05,
        rng_stream="augmentation",
    )
    output = model(
        batch.surface_events,
        batch.variable_states,
        mode=ORACLE_MODE,
        contradiction_observed=True,
        oracle_core_mask=batch.core_masks,
    )
    conflict_targets = torch.tensor(
        [int(item["conflict_variable"]) for item in batch.metadata["episodes"]],
        dtype=torch.long,
    )
    observed = compute_exp287_training_loss(
        output,
        conflict_targets=conflict_targets,
        solution_targets=batch.solution_targets,
        core_masks=batch.core_masks,
    )
    expected = (
        F.cross_entropy(output.rollback_logits, conflict_targets)
        + F.binary_cross_entropy(
            output.verifier_confidence,
            batch.solution_targets.to(dtype=output.verifier_confidence.dtype),
        )
        + F.binary_cross_entropy_with_logits(output.localizer_logits, batch.core_masks)
    )
    assert torch.equal(observed, expected)


def test_same_weights_evaluation_preserves_model_digest_and_oracle_boundary() -> None:
    model, _ = build_seeded_exp287_model(
        root_seed=f"{ROOT_PREFIX}::1",
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    before = model_state_digest(model)
    receipt = evaluate_exp287_batch_same_weights(
        model,
        _batch(f"{ROOT_PREFIX}::1"),
        per_step_flops=1000,
        ceiling=16000,
        max_search_steps=16,
    )
    after = model_state_digest(model)

    assert before == after == receipt["model_state_digest"]
    assert set(receipt["modes"]) == {
        "NULL_CORE_CONTROL",
        "LEARNED_CORE_TOP2",
        "ORACLE_CORE_UPPER_BOUND",
    }
    assert receipt["modes"]["LEARNED_CORE_TOP2"]["oracle_information_delivered"] is False
    assert receipt["modes"]["NULL_CORE_CONTROL"]["oracle_information_delivered"] is False
    assert receipt["modes"]["ORACLE_CORE_UPPER_BOUND"]["oracle_information_delivered"] is True
    assert receipt["precontradiction_conflict_delivery"] is False
    assert receipt["localization"]["evaluated_variables"] == 16
    assert 0.0 <= receipt["localization"]["top2_core_precision"] <= 1.0


def test_root_classifier_requires_fresh_oracle_headroom_first() -> None:
    metrics = {
        "mean_control_cost": 100.0,
        "mean_learned_cost": 80.0,
        "mean_oracle_cost": 101.0,
        "control_solution_rate": 1.0,
        "learned_solution_rate": 1.0,
        "top2_core_precision": 1.0,
        "learned_oracle_information_delivered": False,
        "precontradiction_conflict_delivery": False,
    }
    result = classify_exp287_root_metrics(metrics)
    assert result["classification"] == ORACLE_NOT_REPLICATED
    assert result["oracle_headroom"] == -1.0
    assert result["capture"] is None


def test_root_classifier_establishes_value_only_when_all_fixed_predicates_close() -> None:
    metrics = {
        "mean_control_cost": 100.0,
        "mean_learned_cost": 70.0,
        "mean_oracle_cost": 50.0,
        "control_solution_rate": 0.90,
        "learned_solution_rate": 0.90,
        "top2_core_precision": 0.75,
        "learned_oracle_information_delivered": False,
        "precontradiction_conflict_delivery": False,
    }
    result = classify_exp287_root_metrics(metrics)
    assert result["classification"] == ESTABLISHED
    assert result["oracle_headroom"] == 50.0
    assert result["learned_headroom"] == 30.0
    assert result["capture"] == pytest.approx(0.6)
    assert result["predicates"] == {
        "oracle_headroom_positive": True,
        "learned_headroom_positive": True,
        "capture_at_least_half": True,
        "solution_floor_preserved": True,
        "top2_precision_above_half": True,
        "learned_oracle_boundary_closed": True,
        "precontradiction_boundary_closed": True,
    }


def test_root_classifier_fails_closed_on_exact_half_precision() -> None:
    metrics = {
        "mean_control_cost": 100.0,
        "mean_learned_cost": 70.0,
        "mean_oracle_cost": 50.0,
        "control_solution_rate": 0.90,
        "learned_solution_rate": 0.90,
        "top2_core_precision": 0.50,
        "learned_oracle_information_delivered": False,
        "precontradiction_conflict_delivery": False,
    }
    result = classify_exp287_root_metrics(metrics)
    assert result["classification"] == NOT_ESTABLISHED
    assert result["predicates"]["top2_precision_above_half"] is False
