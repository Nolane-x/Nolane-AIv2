from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _pair():
    from nolane_ai.experiments.matched_cbrf_arms import build_matched_exp277_arm_pair

    return build_matched_exp277_arm_pair(
        d_model=8,
        hidden_size=12,
        target_parameters=10_000,
    )


def test_matched_exp277_arms_have_exact_parameter_and_output_contract() -> None:
    from nolane_ai.experiments.matched_cbrf_arms import audit_matched_exp277_arm_pair

    arcs, oracle = _pair()
    surface = torch.randn(2, 4, 8)
    variables = torch.randn(2, 5, 8)
    incidence = torch.ones(2, 3, 5)

    arcs_out = arcs(surface, variables)
    oracle_out = oracle(surface, variables, incidence)
    assert arcs_out.decision_logits.shape == (2, 5, 2)
    assert oracle_out.decision_logits.shape == (2, 5, 2)
    assert arcs_out.verifier_confidence.shape == (2, 5)
    assert oracle_out.verifier_confidence.shape == (2, 5)

    audit = audit_matched_exp277_arm_pair(
        arcs,
        oracle,
        timesteps=4,
        variables=5,
        constraints=3,
    )
    assert audit["schema"] == "NLM-EXP-277-MATCHED-ARMS-DEV-V1"
    assert audit["evidence_level"] == "EV-E2"
    assert audit["decision"] == "UNVERIFIED"
    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["arcs_branch"]["total_parameters"] == 10_000
    assert audit["oracle_cbrf"]["total_parameters"] == 10_000
    assert audit["arcs_branch"]["functional_parameters"] == audit["oracle_cbrf"]["functional_parameters"]
    assert audit["compute_ledger"]["arcs_branch"]["hardware_profiler_flops_claimed"] is False
    assert audit["compute_ledger"]["oracle_cbrf"]["hardware_profiler_flops_claimed"] is False
    assert audit["compute_budget_closed"] is True


def test_arcs_branch_cannot_receive_oracle_incidence() -> None:
    arcs, _ = _pair()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)
    incidence = torch.ones(1, 2, 4)
    with pytest.raises(TypeError):
        arcs(surface, variables, incidence=incidence)


def test_oracle_cbrf_requires_valid_rank_three_incidence() -> None:
    _, oracle = _pair()
    surface = torch.randn(1, 3, 8)
    variables = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match="incidence"):
        oracle(surface, variables, torch.ones(1, 4))
    with pytest.raises(ValueError, match="incidence"):
        oracle(surface, variables, torch.ones(1, 2, 5))


def test_exp277_compute_budget_fails_closed() -> None:
    from nolane_ai.experiments.matched_cbrf_arms import audit_matched_exp277_arm_pair

    arcs, oracle = _pair()
    baseline = audit_matched_exp277_arm_pair(
        arcs,
        oracle,
        timesteps=4,
        variables=5,
        constraints=3,
    )
    required = baseline["declared_max_accounted_flops_per_episode"]
    with pytest.raises(ValueError, match="compute budget"):
        audit_matched_exp277_arm_pair(
            arcs,
            oracle,
            timesteps=4,
            variables=5,
            constraints=3,
            max_accounted_flops_per_episode=required - 1,
        )
