from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect

import pytest

torch = pytest.importorskip("torch")


def test_exp289_arm_action_path_does_not_read_evaluator_solution_targets() -> None:
    from nolane_ai.experiments import exp289_paired_runner

    source = inspect.getsource(exp289_paired_runner._run_episode)
    assert "solution_targets" not in source


def test_exp289_public_contradiction_observation_uses_problem_constraints() -> None:
    from nolane_ai.experiments.exp289_nogood_worlds import Exp289NogoodGenerator, problem_from_exp289_episode
    from nolane_ai.experiments.exp289_paired_runner import _observed_public_contradiction

    batch = Exp289NogoodGenerator(root_seed="exp289-ground-truth-boundary").make_batch(
        replicate=0,
        batch_size=1,
        timesteps=2,
        restarts=2,
        variables=3,
        decoys=1,
        d_model=4,
        noise_std=0.0,
        rng_stream="evaluation",
    )
    episode = batch.metadata["episodes"][0]
    problem = problem_from_exp289_episode(episode)
    assert problem.constraints
    assert all(len(constraint.scope) == 1 for constraint in problem.constraints)

    first = problem.constraints[0]
    variable = first.scope[0]
    allowed = int(first.allowed[0][0])
    wrong = 1 - allowed

    assert _observed_public_contradiction(problem, {variable: wrong}) is True
    assert _observed_public_contradiction(problem, {variable: allowed}) is False
    # Incomplete assignments must not be misclassified by constraints whose scope
    # has not yet been fully assigned.
    assert _observed_public_contradiction(problem, {}) is False


def _causal_projection(result: dict) -> dict:
    return {
        "raw_dead_end_signatures": result["raw_dead_end_signatures"],
        "per_restart_dead_end_lineage": result["per_restart_dead_end_lineage"],
        "candidate_solution": result["candidate_solution"],
        "repeated_dead_end_reentries": result["repeated_dead_end_reentries"],
        "prevented_repeat_count": result["prevented_repeat_count"],
        "nogood_hits": result["nogood_hits"],
        "memory_insertion_count": result["memory_insertion_count"],
        "memory_query_count": result["memory_query_count"],
        "memory_comparison_count": result["memory_comparison_count"],
        "memory_canonicalization_operations": result[
            "memory_canonicalization_operations"
        ],
        "step_receipts": result["step_receipts"],
        "insertion_actions": [
            (
                receipt["restart_index"],
                receipt["canonical_key"],
                receipt["inserted"],
            )
            for receipt in result["store_insertion_receipts"]
        ],
        "prune_actions": [
            (
                receipt["restart_index"],
                receipt["canonical_key"],
                receipt["prevented_repeat"],
            )
            for receipt in result["prune_receipts"]
        ],
        "accounted_reasoning_cost": result["accounted_reasoning_cost"],
        "censored_at_max_cost": result["censored_at_max_cost"],
        "external_solution_verified": result["external_solution_verified"],
    }


def test_exp289_evaluator_only_truth_cannot_change_causal_search_or_store_actions() -> None:
    from nolane_ai.experiments.exp289_nogood_worlds import Exp289NogoodGenerator
    from nolane_ai.experiments.exp289_paired_runner import _run_episode
    from nolane_ai.experiments.matched_nogood_arms import build_matched_exp289_arm_pair

    batch = Exp289NogoodGenerator(root_seed="exp289-evaluator-poison").make_batch(
        replicate=11,
        batch_size=1,
        timesteps=3,
        restarts=3,
        variables=4,
        decoys=2,
        d_model=6,
        noise_std=0.0,
        rng_stream="evaluation",
    )
    poisoned_metadata = deepcopy(batch.metadata)
    for episode in poisoned_metadata["episodes"]:
        # These fields are evaluator/training truth only. Deliberately make them
        # nonsensical while keeping the public CSP payload unchanged.
        episode["valid_solutions"] = []
    poisoned = replace(
        batch,
        solution_targets=1 - batch.solution_targets,
        metadata=poisoned_metadata,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(289)
        no_nogood, local_nogood = build_matched_exp289_arm_pair(
            d_model=6,
            hidden_size=5,
            target_parameters=5_000,
            device="cpu",
        )

    common = dict(
        episode_index=0,
        max_search_steps=12,
        neural_flops_per_step=1,
        common_ceiling=10**9,
    )
    original_episode = batch.metadata["episodes"][0]
    poisoned_episode = poisoned.metadata["episodes"][0]

    original_no = _run_episode(
        arm_name="no_nogood",
        model=no_nogood,
        batch=batch,
        episode_metadata=original_episode,
        **common,
    )
    poisoned_no = _run_episode(
        arm_name="no_nogood",
        model=no_nogood,
        batch=poisoned,
        episode_metadata=poisoned_episode,
        **common,
    )
    assert _causal_projection(poisoned_no) == _causal_projection(original_no)

    original_local = _run_episode(
        arm_name="local_nogood",
        model=local_nogood,
        batch=batch,
        episode_metadata=original_episode,
        **common,
    )
    poisoned_local = _run_episode(
        arm_name="local_nogood",
        model=local_nogood,
        batch=poisoned,
        episode_metadata=poisoned_episode,
        **common,
    )
    assert _causal_projection(poisoned_local) == _causal_projection(original_local)

    # Post-hoc audit truth is allowed to change when evaluator metadata is poisoned;
    # those audit fields are intentionally excluded from the causal projection.
    assert poisoned_local["store_soundness_violation_count"] != original_local[
        "store_soundness_violation_count"
    ] or poisoned_local["valid_state_overprune_rate"] != original_local[
        "valid_state_overprune_rate"
    ]
