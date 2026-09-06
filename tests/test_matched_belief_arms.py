import pytest

torch = pytest.importorskip("torch")


def test_matched_belief_arm_regions_have_identical_parameter_and_affine_madd_budgets():
    from nolane_ai.experiments.matched_belief_arms import build_matched_belief_arm_pair, audit_matched_belief_arm_pair

    recurrent, explicit = build_matched_belief_arm_pair(d_model=32, hidden_size=24, target_parameters=20_000)
    audit = audit_matched_belief_arm_pair(recurrent, explicit)
    assert audit["recurrent_hidden"]["total_parameters"] == 20_000
    assert audit["explicit_belief"]["total_parameters"] == 20_000
    assert audit["recurrent_hidden"]["functional_parameters"] == audit["explicit_belief"]["functional_parameters"]
    assert audit["recurrent_hidden"]["affine_madds_per_observation"] == audit["explicit_belief"]["affine_madds_per_observation"]
    assert audit["parameter_match"] is True
    assert audit["affine_madd_match"] is True
    assert audit["full_accounted_flop_match"] is True
    assert audit["primitive_operation_match"] is True
    assert audit["relative_accounted_flop_difference"] == pytest.approx(0.0)
    assert audit["remaining_blockers"] == [
        "matched arms are experiment-local components and are not yet integrated into confirmatory checkpoint/evaluator lineage"
    ]


def test_matched_belief_arms_use_identical_observation_history_and_emit_decision_logits():
    from nolane_ai.experiments.matched_belief_arms import build_matched_belief_arm_pair

    recurrent, explicit = build_matched_belief_arm_pair(d_model=16, hidden_size=12, target_parameters=10_000)
    observations = torch.randn(3, 5, 4, 16)
    recurrent_logits = recurrent(observations)
    explicit_logits = explicit(observations)
    assert recurrent_logits.shape == explicit_logits.shape == (3, 4, 2)
    probs = explicit_logits.softmax(dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, 4), atol=1e-6)


def test_all_functional_parameters_receive_gradient_in_both_matched_arms():
    from nolane_ai.experiments.matched_belief_arms import build_matched_belief_arm_pair

    recurrent, explicit = build_matched_belief_arm_pair(d_model=16, hidden_size=12, target_parameters=10_000)
    observations = torch.randn(2, 3, 4, 16)
    targets = torch.randint(0, 2, (2, 4))
    for arm in (recurrent, explicit):
        arm.zero_grad(set_to_none=True)
        logits = arm(observations)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        loss.backward()
        for name, parameter in arm.named_parameters():
            if "capacity_reserve" in name:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None, name


def test_explicit_belief_state_is_exposed_while_recurrent_state_remains_hidden():
    from nolane_ai.experiments.matched_belief_arms import build_matched_belief_arm_pair

    recurrent, explicit = build_matched_belief_arm_pair(d_model=16, hidden_size=12, target_parameters=10_000)
    observations = torch.randn(2, 3, 4, 16)
    recurrent_result = recurrent.run_with_state(observations)
    explicit_result = explicit.run_with_state(observations)
    assert recurrent_result.state_semantics == "opaque_recurrent_hidden"
    assert explicit_result.state_semantics == "explicit_binary_belief_logits"
    assert recurrent_result.decision_logits.shape == explicit_result.decision_logits.shape
    assert explicit_result.state.shape[-1] == 2
    assert recurrent_result.state.shape[-1] == 12
