from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _pair():
    from nolane_ai.experiments.matched_conflict_arms import build_matched_exp286_arm_pair

    return build_matched_exp286_arm_pair(
        d_model=8,
        hidden_size=10,
        target_parameters=5_000,
    )


def test_matched_exp286_arms_close_parameter_information_and_output_contract() -> None:
    from nolane_ai.experiments.matched_conflict_arms import audit_matched_exp286_arm_pair

    chronological, oracle = _pair()

    chronological_state = chronological.state_dict()
    oracle_state = oracle.state_dict()
    assert tuple(chronological_state) == tuple(oracle_state)
    for name in chronological_state:
        assert torch.equal(chronological_state[name], oracle_state[name]), name

    surface = torch.randn(2, 4, 8)
    variables = torch.randn(2, 5, 8)
    core_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    chronological_out = chronological(
        surface,
        variables,
        contradiction_observed=True,
    )
    oracle_out = oracle(
        surface,
        variables,
        conflict_core_mask=core_mask,
        contradiction_observed=True,
    )

    for output in (chronological_out, oracle_out):
        assert output.rollback_logits.shape == (2, 5)
        assert output.verifier_confidence.shape == (2, 5)
        assert isinstance(output.conflict_conditioning_used, bool)
        assert isinstance(output.representation_semantics, str)
        assert output.representation_semantics

    assert chronological_out.conflict_conditioning_used is False
    assert oracle_out.conflict_conditioning_used is True

    audit = audit_matched_exp286_arm_pair(
        chronological,
        oracle,
        timesteps=4,
        variables=5,
        max_search_steps=8,
    )
    assert audit["schema"] == "NLM-EXP-286-MATCHED-CONFLICT-ARMS-DEV-V1"
    assert audit["evidence_level"] == "EV-E2"
    assert audit["decision"] == "UNVERIFIED"
    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["active_functional_parameter_match"] is True
    assert audit["optimizer_visible_parameter_match"] is True
    assert audit["oracle_information_separation"] is True
    assert audit["compute_budget_closed"] is True
    assert audit["chronological_failure"]["total_parameters"] == 5_000
    assert audit["oracle_conflict_core"]["total_parameters"] == 5_000
    assert (
        audit["chronological_failure"]["reserved_parameters"]
        == audit["oracle_conflict_core"]["reserved_parameters"]
    )
    assert (
        audit["chronological_failure"]["optimizer_visible_parameters"]
        == audit["oracle_conflict_core"]["optimizer_visible_parameters"]
    )
    assert audit["compute_ledger"]["chronological_failure"]["hardware_profiler_flops_claimed"] is False
    assert audit["compute_ledger"]["oracle_conflict_core"]["hardware_profiler_flops_claimed"] is False
    assert (
        audit["compute_ledger"]["chronological_failure"]["max_accounted_flops_per_episode"]
        <= audit["declared_max_accounted_flops_per_episode"]
    )
    assert (
        audit["compute_ledger"]["oracle_conflict_core"]["max_accounted_flops_per_episode"]
        <= audit["declared_max_accounted_flops_per_episode"]
    )


def test_chronological_failure_has_no_oracle_core_input_surface() -> None:
    chronological, _ = _pair()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)
    core_mask = torch.ones(1, 4)

    with pytest.raises(TypeError):
        chronological(
            surface,
            variables,
            conflict_core_mask=core_mask,
            contradiction_observed=True,
        )


def test_oracle_core_is_forbidden_before_current_contradiction() -> None:
    _, oracle = _pair()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)
    core_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match="contradiction"):
        oracle(
            surface,
            variables,
            conflict_core_mask=core_mask,
            contradiction_observed=False,
        )


def test_oracle_requires_valid_current_core_shape_and_membership() -> None:
    _, oracle = _pair()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)

    with pytest.raises(ValueError, match="conflict_core_mask"):
        oracle(
            surface,
            variables,
            conflict_core_mask=torch.ones(1, 3),
            contradiction_observed=True,
        )
    with pytest.raises(ValueError, match="conflict_core_mask"):
        oracle(
            surface,
            variables,
            conflict_core_mask=torch.zeros(1, 4),
            contradiction_observed=True,
        )


def test_exp286_arms_require_rank_three_common_inputs() -> None:
    chronological, oracle = _pair()
    valid_surface = torch.randn(1, 3, 8)
    valid_variables = torch.randn(1, 4, 8)
    valid_core = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="surface_events"):
        chronological(
            torch.randn(1, 8),
            valid_variables,
            contradiction_observed=True,
        )
    with pytest.raises(ValueError, match="variable_states"):
        oracle(
            valid_surface,
            torch.randn(1, 8),
            conflict_core_mask=valid_core,
            contradiction_observed=True,
        )


def test_exp286_compute_budget_fails_closed() -> None:
    from nolane_ai.experiments.matched_conflict_arms import audit_matched_exp286_arm_pair

    chronological, oracle = _pair()
    baseline = audit_matched_exp286_arm_pair(
        chronological,
        oracle,
        timesteps=4,
        variables=5,
        max_search_steps=8,
    )
    required = baseline["declared_max_accounted_flops_per_episode"]

    with pytest.raises(ValueError, match="compute budget"):
        audit_matched_exp286_arm_pair(
            chronological,
            oracle,
            timesteps=4,
            variables=5,
            max_search_steps=8,
            max_accounted_flops_per_episode=required - 1,
        )
