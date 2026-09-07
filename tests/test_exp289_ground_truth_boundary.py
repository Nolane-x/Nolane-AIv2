from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")


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
    first = problem.constraints[0]
    variable = first.scope[0]
    allowed = int(first.allowed[0][0])
    wrong = 1 - allowed

    assert _observed_public_contradiction(problem, {variable: wrong}) is True
    assert _observed_public_contradiction(problem, {variable: allowed}) is False
    # Incomplete assignments must not be misclassified by constraints whose scope
    # has not yet been fully assigned.
    assert _observed_public_contradiction(problem, {}) is False
