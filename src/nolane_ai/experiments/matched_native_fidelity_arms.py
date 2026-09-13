from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.experiments.exp299_structural_encoding import PAIR_FEATURE_WIDTH
from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.training.optimizer import functional_trainable_named_parameters
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes

ArmId = Literal["binary_supervision_control", "court_teacher_then_native"]
TEACHER_TARGET_WIDTH = 7


@dataclass(frozen=True, slots=True)
class NativeFidelityForward:
    authority_logit: torch.Tensor
    fidelity_logit: torch.Tensor
    teacher_prediction: torch.Tensor


@dataclass(frozen=True, slots=True)
class NativeFidelityDecision:
    arm_id: ArmId
    authority_granted: bool
    authority_score: float
    fidelity_score: float
    teacher_prediction: tuple[float, ...]
    neural_accounted_flops: int
    native_inference: bool
    teacher_scaffold_consumed: bool


class _MatchedNativeFidelityArmBase(nn.Module):
    def __init__(
        self,
        *,
        arm_id: ArmId,
        auxiliary_loss_weight: float,
        d_model: int,
        hidden_size: int,
        target_parameters: int,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-299 dimensions and target_parameters must be positive")
        if auxiliary_loss_weight < 0:
            raise ValueError("EXP-299 auxiliary_loss_weight must be non-negative")
        self.arm_id = arm_id
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)

        self.structural_projection = nn.Linear(
            PAIR_FEATURE_WIDTH,
            d_model,
            device=device,
        )
        self.structural_norm = nn.LayerNorm(d_model, device=device)
        self.comparison_in = nn.Linear(d_model, hidden_size, device=device)
        self.comparison_out = nn.Linear(hidden_size, hidden_size, device=device)
        self.authority_head = nn.Linear(hidden_size, 1, device=device)
        self.fidelity_head = nn.Linear(hidden_size, 1, device=device)
        self.teacher_head = nn.Linear(
            hidden_size,
            TEACHER_TARGET_WIDTH,
            device=device,
        )
        self.residual_gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(
            self,
            target_parameters,
            device=device,
            frozen=False,
        )

    def _coerce_pair_features(
        self, pair_features: torch.Tensor | Sequence[float]
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        if isinstance(pair_features, torch.Tensor):
            tensor = pair_features.to(device=parameter.device, dtype=parameter.dtype)
        else:
            tensor = torch.tensor(
                tuple(float(value) for value in pair_features),
                device=parameter.device,
                dtype=parameter.dtype,
            )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != PAIR_FEATURE_WIDTH:
            raise ValueError(
                f"EXP-299 pair features must have width {PAIR_FEATURE_WIDTH}"
            )
        return tensor

    def forward_pair(
        self, pair_features: torch.Tensor | Sequence[float]
    ) -> NativeFidelityForward:
        pair = self._coerce_pair_features(pair_features)
        projected = F.silu(self.structural_projection(pair))
        normalized = self.structural_norm(projected)
        hidden = F.silu(self.comparison_in(normalized))
        residual = F.silu(self.comparison_out(hidden))
        hidden = hidden + torch.sigmoid(self.residual_gate) * residual
        authority_logit = self.authority_head(hidden).squeeze(-1)
        fidelity_logit = self.fidelity_head(hidden).squeeze(-1)
        teacher_prediction = torch.sigmoid(self.teacher_head(hidden))
        return NativeFidelityForward(
            authority_logit=authority_logit,
            fidelity_logit=fidelity_logit,
            teacher_prediction=teacher_prediction,
        )

    def _neural_accounted_flops(self) -> int:
        d = self.d_model
        h = self.hidden_size

        def linear(inputs: int, outputs: int) -> int:
            return 2 * inputs * outputs + outputs

        projection = linear(PAIR_FEATURE_WIDTH, d)
        norm_and_activation = 6 * d
        comparison = linear(d, h) + linear(h, h) + 8 * h
        heads = 2 * linear(h, 1) + linear(h, TEACHER_TARGET_WIDTH)
        return int(projection + norm_and_activation + comparison + heads)

    def decide(
        self,
        pair_features: torch.Tensor | Sequence[float],
        *,
        executable: bool,
        threshold: float = 0.5,
    ) -> NativeFidelityDecision:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("EXP-299 authority threshold must be in [0,1]")
        with torch.no_grad():
            output = self.forward_pair(pair_features)
            if output.authority_logit.numel() != 1:
                raise ValueError("EXP-299 native decide accepts exactly one pair")
            authority_score = float(
                torch.sigmoid(output.authority_logit).reshape(-1)[0].cpu().item()
            )
            fidelity_score = float(
                torch.sigmoid(output.fidelity_logit).reshape(-1)[0].cpu().item()
            )
            teacher_prediction = tuple(
                float(value)
                for value in output.teacher_prediction.reshape(-1).cpu().tolist()
            )
        return NativeFidelityDecision(
            arm_id=self.arm_id,
            authority_granted=bool(executable and authority_score >= threshold),
            authority_score=authority_score,
            fidelity_score=fidelity_score,
            teacher_prediction=teacher_prediction,
            neural_accounted_flops=self._neural_accounted_flops(),
            native_inference=True,
            teacher_scaffold_consumed=False,
        )


class BinarySupervisionControlArm(_MatchedNativeFidelityArmBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            arm_id="binary_supervision_control",
            auxiliary_loss_weight=0.0,
            **kwargs,
        )


class CourtTeacherNativeArm(_MatchedNativeFidelityArmBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            arm_id="court_teacher_then_native",
            auxiliary_loss_weight=0.25,
            **kwargs,
        )


def build_matched_native_fidelity_arms(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> tuple[BinarySupervisionControlArm, CourtTeacherNativeArm]:
    control = BinarySupervisionControlArm(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        device=device,
    )
    teacher = CourtTeacherNativeArm(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        device=device,
    )
    teacher.load_state_dict(control.state_dict())
    return control, teacher


def _parameter_audit(arm: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in arm.parameters())
    reserve = sum(
        parameter.numel()
        for name, parameter in arm.named_parameters()
        if name.endswith("capacity_reserve") or ".capacity_reserve" in name
    )
    functional = total - reserve
    optimizer_visible = sum(
        parameter.numel()
        for _, parameter in functional_trainable_named_parameters(arm)
    )
    return {
        "total_parameters": int(total),
        "functional_parameters": int(functional),
        "optimizer_visible_parameters": int(optimizer_visible),
        "reserved_parameters": int(reserve),
    }


def native_arm_state_digest(arm: nn.Module) -> str:
    digest = sha256()
    digest.update(tensor_byteorder().encode("ascii"))
    for name, tensor in arm.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(tensor_raw_bytes(value))
    return digest.hexdigest()


def _state_layout(arm: nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(tensor.shape)) for name, tensor in arm.state_dict().items())


def audit_matched_native_fidelity_arms(
    control: BinarySupervisionControlArm,
    teacher: CourtTeacherNativeArm,
) -> dict[str, Any]:
    control_audit = _parameter_audit(control)
    teacher_audit = _parameter_audit(teacher)
    return {
        "schema": "NLM-EXP-299-MATCHED-NATIVE-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "parameter_match": control_audit["total_parameters"]
        == teacher_audit["total_parameters"],
        "functional_parameter_match": control_audit["functional_parameters"]
        == teacher_audit["functional_parameters"],
        "optimizer_visible_parameter_match": control_audit[
            "optimizer_visible_parameters"
        ]
        == teacher_audit["optimizer_visible_parameters"],
        "initialization_match": native_arm_state_digest(control)
        == native_arm_state_digest(teacher),
        "neural_accounted_flops_match": control._neural_accounted_flops()
        == teacher._neural_accounted_flops(),
        "forward_path_match": _state_layout(control) == _state_layout(teacher),
        "primary_supervision_match_required": True,
        "only_causal_fit_intervention": "teacher_auxiliary_loss_weight",
        "auxiliary_loss_weights": {
            "binary_supervision_control": control.auxiliary_loss_weight,
            "court_teacher_then_native": teacher.auxiliary_loss_weight,
        },
        "arms": {
            "binary_supervision_control": control_audit,
            "court_teacher_then_native": teacher_audit,
        },
        "heldout_teacher_scaffold_consumed": False,
        "evaluator_fields_in_native_api": False,
    }
