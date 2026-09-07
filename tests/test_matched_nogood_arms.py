from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")


def _pair():
    from nolane_ai.experiments.matched_nogood_arms import build_matched_exp289_arm_pair

    return build_matched_exp289_arm_pair(
        d_model=8,
        hidden_size=10,
        target_parameters=5_000,
    )


def test_matched_exp289_arms_close_parameter_and_output_contract() -> None:
    from nolane_ai.experiments.matched_nogood_arms import audit_matched_exp289_arm_pair

    no_nogood, local = _pair()
    no_state = no_nogood.state_dict()
    local_state = local.state_dict()
    assert tuple(no_state) == tuple(local_state)
    for name in no_state:
        assert torch.equal(no_state[name], local_state[name]), name

    surface = torch.randn(2, 4, 8)
    variables = torch.randn(2, 5, 8)
    no_out = no_nogood(surface, variables)
    local_out = local(surface, variables, memory_hit=torch.tensor([1.0, 0.0]))

    for output in (no_out, local_out):
        assert output.branch_logits.shape == (2, 5)
        assert output.verifier_confidence.shape == (2, 5)
        assert output.memory_query_logit.shape == (2, 5)
        assert isinstance(output.memory_conditioning_used, bool)
        assert isinstance(output.representation_semantics, str)
        assert output.representation_semantics

    assert no_out.memory_conditioning_used is False
    assert local_out.memory_conditioning_used is True

    audit = audit_matched_exp289_arm_pair(
        no_nogood,
        local,
        restarts=3,
        variables=5,
        max_search_steps=8,
    )
    assert audit["schema"] == "NLM-EXP-289-MATCHED-NOGOOD-ARMS-DEV-V1"
    assert audit["evidence_level"] == "EV-E2"
    assert audit["decision"] == "UNVERIFIED"
    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["active_functional_parameter_match"] is True
    assert audit["optimizer_visible_parameter_match"] is True
    assert audit["memory_scope_closed"] is True
    assert audit["compute_budget_closed"] is True
    assert audit["hardware_profiler_flops_claimed"] is False
    assert audit["no_nogood"]["total_parameters"] == 5_000
    assert audit["local_nogood"]["total_parameters"] == 5_000
    assert audit["no_nogood"]["reserved_parameters"] == audit["local_nogood"]["reserved_parameters"]
    assert audit["no_nogood"]["optimizer_visible_parameters"] == audit["local_nogood"]["optimizer_visible_parameters"]
    assert audit["compute_ledger"]["local_nogood"]["max_memory_accounted_operations_per_episode"] > 0
    assert audit["compute_ledger"]["no_nogood"]["max_memory_accounted_operations_per_episode"] == 0
    assert audit["compute_ledger"]["local_nogood"]["hardware_profiler_flops_claimed"] is False
    assert audit["compute_ledger"]["no_nogood"]["hardware_profiler_flops_claimed"] is False


def test_episode_scoped_store_uses_exact_nonempty_subset_semantics() -> None:
    from nolane_ai.experiments.matched_nogood_arms import EpisodeScopedNogoodStore

    store = EpisodeScopedNogoodStore(episode_digest="episode-a", problem_digest="problem-a")
    inserted = store.add({"x": 0, "y": 1}, dead_end_observed=True)
    assert inserted is True
    assert len(store) == 1
    assert store.matches({"x": 0, "y": 1, "z": 0}) is True
    assert store.matches({"x": 0, "y": 0}) is False
    assert store.matches({"x": 0}) is False

    duplicate = store.add({"y": 1, "x": 0}, dead_end_observed=True)
    assert duplicate is False
    assert len(store) == 1

    counters = store.snapshot_counters()
    assert counters["insertion_count"] == 1
    assert counters["query_count"] == 3
    assert counters["comparison_count"] >= 3
    assert counters["hit_count"] == 1
    assert counters["canonicalization_operations"] > 0


def test_episode_scoped_store_rejects_root_unobserved_and_scope_drift() -> None:
    from nolane_ai.experiments.matched_nogood_arms import EpisodeScopedNogoodStore

    store = EpisodeScopedNogoodStore(episode_digest="episode-a", problem_digest="problem-a")
    with pytest.raises(ValueError, match="non-empty"):
        store.add({}, dead_end_observed=True)
    with pytest.raises(ValueError, match="dead end"):
        store.add({"x": 0}, dead_end_observed=False)
    with pytest.raises(ValueError, match="episode scope"):
        store.add(
            {"x": 0},
            dead_end_observed=True,
            episode_digest="episode-b",
            problem_digest="problem-a",
        )
    with pytest.raises(ValueError, match="problem scope"):
        store.matches(
            {"x": 0},
            episode_digest="episode-a",
            problem_digest="problem-b",
        )


def test_store_api_has_no_ground_truth_or_oracle_admission_surface() -> None:
    from nolane_ai.experiments.matched_nogood_arms import EpisodeScopedNogoodStore

    add_parameters = set(inspect.signature(EpisodeScopedNogoodStore.add).parameters)
    assert "ground_truth_valid" not in add_parameters
    assert "no_valid_completion" not in add_parameters
    assert "oracle_conflict_core" not in add_parameters
    assert "future_solution" not in add_parameters

    store = EpisodeScopedNogoodStore(episode_digest="episode-a", problem_digest="problem-a")
    assert not hasattr(store, "similarity")
    assert not hasattr(store, "nearest")
    assert not hasattr(store, "fuzzy_matches")


def test_no_nogood_arm_has_no_real_memory_hit_input_surface() -> None:
    no_nogood, _ = _pair()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)

    with pytest.raises(TypeError):
        no_nogood(surface, variables, memory_hit=torch.tensor([1.0]))


def test_local_nogood_arm_requires_binary_current_query_hit() -> None:
    _, local = _pair()
    surface = torch.randn(2, 3, 8)
    variables = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match="memory_hit"):
        local(surface, variables, memory_hit=torch.ones(2, 1))
    with pytest.raises(ValueError, match="memory_hit"):
        local(surface, variables, memory_hit=torch.tensor([0.5, 1.0]))


def test_exp289_arms_require_rank_three_common_inputs() -> None:
    no_nogood, local = _pair()
    valid_surface = torch.randn(1, 3, 8)
    valid_variables = torch.randn(1, 4, 8)

    with pytest.raises(ValueError, match="surface_events"):
        no_nogood(torch.randn(1, 8), valid_variables)
    with pytest.raises(ValueError, match="variable_states"):
        local(valid_surface, torch.randn(1, 8), memory_hit=torch.tensor([0.0]))


def test_exp289_compute_budget_fails_closed() -> None:
    from nolane_ai.experiments.matched_nogood_arms import audit_matched_exp289_arm_pair

    no_nogood, local = _pair()
    baseline = audit_matched_exp289_arm_pair(
        no_nogood,
        local,
        restarts=3,
        variables=5,
        max_search_steps=8,
    )
    required = baseline["declared_max_accounted_cost_per_episode"]

    with pytest.raises(ValueError, match="compute budget"):
        audit_matched_exp289_arm_pair(
            no_nogood,
            local,
            restarts=3,
            variables=5,
            max_search_steps=8,
            max_accounted_cost_per_episode=required - 1,
        )
