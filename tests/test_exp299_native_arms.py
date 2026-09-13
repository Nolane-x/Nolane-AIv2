from __future__ import annotations

import inspect

import torch

from nolane_ai.experiments.exp299_structural_encoding import PAIR_FEATURE_WIDTH
from nolane_ai.experiments.matched_native_fidelity_arms import (
    BinarySupervisionControlArm,
    CourtTeacherNativeArm,
    audit_matched_native_fidelity_arms,
    build_matched_native_fidelity_arms,
)


def _build() -> tuple[BinarySupervisionControlArm, CourtTeacherNativeArm]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(299)
        return build_matched_native_fidelity_arms(
            d_model=64,
            hidden_size=64,
            target_parameters=500_000,
        )


def test_matched_arms_have_exact_geometry_and_initialization() -> None:
    control, teacher = _build()
    audit = audit_matched_native_fidelity_arms(control, teacher)
    assert audit["schema"] == "NLM-EXP-299-MATCHED-NATIVE-ARMS-DEV-V1"
    assert audit["parameter_match"] is True
    assert audit["functional_parameter_match"] is True
    assert audit["optimizer_visible_parameter_match"] is True
    assert audit["initialization_match"] is True
    assert audit["neural_accounted_flops_match"] is True
    assert audit["forward_path_match"] is True
    assert audit["arms"]["binary_supervision_control"]["total_parameters"] == 500_000
    assert audit["arms"]["court_teacher_then_native"]["total_parameters"] == 500_000


def test_only_frozen_auxiliary_loss_weight_differs_between_arms() -> None:
    control, teacher = _build()
    assert control.auxiliary_loss_weight == 0.0
    assert teacher.auxiliary_loss_weight == 0.25
    control_state = control.state_dict()
    teacher_state = teacher.state_dict()
    assert control_state.keys() == teacher_state.keys()
    for name in control_state:
        assert torch.equal(control_state[name], teacher_state[name]), name


def test_forward_path_consumes_structural_pair_and_emits_matched_heads() -> None:
    control, teacher = _build()
    pair = torch.linspace(-1.0, 1.0, PAIR_FEATURE_WIDTH).view(1, -1)
    with torch.no_grad():
        left = control.forward_pair(pair)
        right = teacher.forward_pair(pair)
    assert left.authority_logit.shape == (1,)
    assert left.fidelity_logit.shape == (1,)
    assert left.teacher_prediction.shape == (1, 7)
    assert torch.equal(left.authority_logit, right.authority_logit)
    assert torch.equal(left.fidelity_logit, right.fidelity_logit)
    assert torch.equal(left.teacher_prediction, right.teacher_prediction)


def test_native_decide_is_neural_only_and_thresholded() -> None:
    control, teacher = _build()
    pair = [0.0] * PAIR_FEATURE_WIDTH
    for arm in (control, teacher):
        decision = arm.decide(pair, executable=False, threshold=0.5)
        assert decision.authority_granted is False
        assert 0.0 <= decision.authority_score <= 1.0
        assert 0.0 <= decision.fidelity_score <= 1.0
        assert len(decision.teacher_prediction) == 7
        assert decision.native_inference is True
        assert decision.teacher_scaffold_consumed is False


def test_native_interfaces_cannot_receive_court_or_evaluator_fields() -> None:
    forbidden = {
        "receipt",
        "court_receipt",
        "witness",
        "truth",
        "is_faithful",
        "stratum",
        "court_decision",
        "teacher_target",
    }
    for cls in (BinarySupervisionControlArm, CourtTeacherNativeArm):
        parameters = set(inspect.signature(cls.decide).parameters)
        assert not (parameters & forbidden)
        forward_parameters = set(inspect.signature(cls.forward_pair).parameters)
        assert not (forward_parameters & forbidden)


def test_wrong_pair_width_fails_closed() -> None:
    control, _ = _build()
    try:
        control.forward_pair(torch.zeros(1, PAIR_FEATURE_WIDTH - 1))
    except ValueError as exc:
        assert "512" in str(exc)
    else:
        raise AssertionError("wrong EXP-299 pair width must fail closed")
