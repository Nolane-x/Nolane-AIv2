from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _triplet(*, route_threshold: float = 0.5):
    from nolane_ai.experiments.matched_routing_arms import build_matched_exp279_arm_triplet

    return build_matched_exp279_arm_triplet(
        d_model=8,
        hidden_size=10,
        target_parameters=10_000,
        route_threshold=route_threshold,
    )


def test_matched_exp279_arms_close_parameter_reclaim_and_output_contract() -> None:
    from nolane_ai.experiments.matched_routing_arms import audit_matched_exp279_arm_triplet

    propagation, branch, hybrid = _triplet()
    surface = torch.randn(2, 4, 8)
    variables = torch.randn(2, 5, 8)
    incidence = torch.ones(2, 3, 5)

    propagation_out = propagation(surface, variables, incidence)
    branch_out = branch(surface, variables)
    hybrid_out = hybrid(surface, variables, incidence)

    for output in (propagation_out, branch_out, hybrid_out):
        assert output.decision_logits.shape == (2, 5, 2)
        assert output.verifier_confidence.shape == (2, 5)
        assert output.residual_uncertainty.shape == (2,)
        assert output.branch_route_mask.shape == (2,)

    audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=4,
        variables=5,
        constraints=3,
    )
    assert audit["schema"] == "NLM-EXP-279-MATCHED-ROUTING-ARMS-DEV-V1"
    assert audit["evidence_level"] == "EV-E2"
    assert audit["decision"] == "UNVERIFIED"
    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["active_functional_parameter_match"] is True
    assert audit["reclaimed_parameter_assignment_closed"] is True
    assert audit["compute_budget_closed"] is True
    assert audit["structure_fit_strata"] == ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]

    for arm in ("propagation_only", "branch_only", "hybrid"):
        assert audit[arm]["total_parameters"] == 10_000
        assert audit[arm]["functional_parameters"] > 0
        assert audit[arm]["active_functional_parameters"] == audit[arm]["functional_parameters"]
        assert audit["reclaimed_parameter_assignment"][arm]
        assert audit["compute_ledger"][arm]["hardware_profiler_flops_claimed"] is False
        assert audit["compute_ledger"][arm]["max_accounted_flops_per_episode"] <= audit["declared_max_accounted_flops_per_episode"]


def test_branch_only_cannot_receive_compiled_incidence() -> None:
    _, branch, _ = _triplet()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)
    incidence = torch.ones(1, 2, 4)
    with pytest.raises(TypeError):
        branch(surface, variables, incidence=incidence)


def test_propagation_and_hybrid_require_valid_rank_three_incidence() -> None:
    propagation, _, hybrid = _triplet()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)
    for arm in (propagation, hybrid):
        with pytest.raises(ValueError, match="incidence"):
            arm(surface, variables, torch.ones(1, 4))
        with pytest.raises(ValueError, match="incidence"):
            arm(surface, variables, torch.ones(1, 2, 5))


def test_hybrid_route_threshold_is_explicit_and_controls_branch_execution() -> None:
    surface = torch.randn(3, 3, 8)
    variables = torch.randn(3, 4, 8)
    incidence = torch.ones(3, 2, 4)

    _, _, always_route = _triplet(route_threshold=0.0)
    always_output = always_route(surface, variables, incidence)
    assert always_output.branch_route_mask.dtype == torch.bool
    assert always_output.branch_route_mask.tolist() == [True, True, True]

    _, _, never_route = _triplet(route_threshold=1.0)
    never_output = never_route(surface, variables, incidence)
    assert never_output.branch_route_mask.tolist() == [False, False, False]


def test_exp279_compute_budget_fails_closed() -> None:
    from nolane_ai.experiments.matched_routing_arms import audit_matched_exp279_arm_triplet

    propagation, branch, hybrid = _triplet()
    baseline = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=4,
        variables=5,
        constraints=3,
    )
    required = baseline["declared_max_accounted_flops_per_episode"]
    with pytest.raises(ValueError, match="compute budget"):
        audit_matched_exp279_arm_triplet(
            propagation,
            branch,
            hybrid,
            timesteps=4,
            variables=5,
            constraints=3,
            max_accounted_flops_per_episode=required - 1,
        )
