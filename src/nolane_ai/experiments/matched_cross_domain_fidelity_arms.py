from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from nolane_ai.experiments.exp298_behavioral_fidelity import BehavioralFidelityReceipt
from nolane_ai.model.regions import finalize_region_budget
from nolane_ai.model.stage_a_regions import FidelityCourtRegion
from nolane_ai.training.optimizer import functional_trainable_named_parameters

ArmId = Literal["compile_only_control", "cross_domain_fidelity_fabric"]


@dataclass(frozen=True, slots=True)
class CrossDomainFidelityArmDecision:
    arm_id: ArmId
    authority_granted: bool
    fidelity_score: float
    authority_score: float
    verifier_score: float
    receipt_semantics: str
    neural_accounted_flops: int
    arm_observable_digest: str


class _MatchedCrossDomainFidelityArmBase(nn.Module):
    def __init__(
        self,
        *,
        arm_id: ArmId,
        d_model: int,
        hidden_size: int,
        target_parameters: int,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, hidden_size, target_parameters) <= 0:
            raise ValueError("EXP-298 dimensions and target_parameters must be positive")
        self.arm_id = arm_id
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.target_parameters = int(target_parameters)
        self.source_encoder = nn.Linear(d_model, hidden_size, device=device)
        self.candidate_encoder = nn.Linear(d_model, hidden_size, device=device)
        self.comparison_core = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, device=device)
        court_budget = max(512, target_parameters // 3)
        self.fidelity_region = FidelityCourtRegion(
            hidden_size,
            court_budget,
            device=device,
            frozen=False,
        )
        self.receipt_adapter = nn.Linear(8, hidden_size, device=device)
        self.authority_head = nn.Linear(hidden_size, 1, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        self.mix_gate = nn.Parameter(torch.zeros((), device=device))
        finalize_region_budget(self, target_parameters, device=device, frozen=False)

    def _digest_vector(
        self,
        digest_hex: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        try:
            raw = bytes.fromhex(digest_hex)
        except ValueError as exc:
            raise ValueError("canonical digest must be hexadecimal") from exc
        if len(raw) != 32:
            raise ValueError("canonical digest must be SHA256-sized")
        values: list[float] = []
        index = 0
        while len(values) < self.d_model:
            values.append(float(raw[index % len(raw)]) / 255.0)
            index += 1
        return torch.tensor(values, device=device, dtype=dtype)

    @staticmethod
    def _null_receipt_vector(
        *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(8, device=device, dtype=dtype)

    @staticmethod
    def _observed_receipt_vector(
        receipt: BehavioralFidelityReceipt,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        witness = receipt.witness or {}
        direction = witness.get("direction")
        operation_denom = max(receipt.semantic_domain_operations, 1)
        return torch.tensor(
            [
                float(receipt.decision == "court_accept"),
                float(receipt.decision == "court_reject"),
                float(receipt.decision == "court_inconclusive"),
                float(direction == "source_to_candidate"),
                float(direction == "candidate_to_source"),
                float(receipt.complete_probe_space),
                min(float(receipt.probes_evaluated) / 4096.0, 1.0),
                min(float(receipt.semantic_domain_operations) / float(operation_denom), 1.0),
            ],
            device=device,
            dtype=dtype,
        )

    def _neural_accounted_flops(self) -> int:
        d = self.d_model
        h = self.hidden_size

        def linear(inputs: int, outputs: int) -> int:
            return 2 * inputs * outputs + outputs

        source_candidate = 2 * linear(d, h)
        recurrent = 6 * (2 * h * h + h * h) + 12 * h
        receipt = linear(8, h)
        region_pair = linear(4 * h, h) + linear(h, 1) + 10 * h
        heads = 2 * linear(h, 1)
        fusion = 3 * h + 1
        return int(source_candidate + recurrent + receipt + region_pair + heads + fusion)

    def decide(
        self,
        *,
        source_digest: str,
        candidate_digest: str,
        executable: bool,
        court_receipt: BehavioralFidelityReceipt | None = None,
    ) -> CrossDomainFidelityArmDecision:
        if self.arm_id == "compile_only_control" and court_receipt is not None:
            raise ValueError("control arm must not consume a behavioral fidelity receipt")
        if self.arm_id == "cross_domain_fidelity_fabric" and court_receipt is None:
            raise ValueError("fidelity arm requires an observed behavioral fidelity receipt")

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        source_vector = self._digest_vector(source_digest, device=device, dtype=dtype)
        candidate_vector = self._digest_vector(candidate_digest, device=device, dtype=dtype)
        source_state = F.silu(self.source_encoder(source_vector))
        candidate_state = F.silu(self.candidate_encoder(candidate_vector))
        comparison_input = torch.cat((source_state, candidate_state), dim=-1).view(1, 1, -1)
        recurrent, _ = self.comparison_core(comparison_input)
        core_state = recurrent[0, -1]

        if court_receipt is None:
            receipt_vector = self._null_receipt_vector(device=device, dtype=dtype)
            receipt_semantics = "canonical_null_cross_domain_receipt"
        else:
            receipt_vector = self._observed_receipt_vector(
                court_receipt,
                device=device,
                dtype=dtype,
            )
            receipt_semantics = "observed_public_behavioral_fidelity_receipt"

        receipt_state = F.silu(self.receipt_adapter(receipt_vector))
        fused = core_state + torch.sigmoid(self.mix_gate) * receipt_state
        fidelity_score = self.fidelity_region.fidelity_score(source_state, candidate_state)
        authority_score = torch.sigmoid(self.authority_head(fused)).squeeze(-1)
        verifier_score = torch.sigmoid(self.verifier_head(fused)).squeeze(-1)

        if self.arm_id == "compile_only_control":
            authority_granted = bool(executable)
        else:
            assert court_receipt is not None
            authority_granted = bool(executable and court_receipt.decision == "court_accept")

        observable_digest = sha256(
            (
                source_digest
                + candidate_digest
                + receipt_semantics
                + str(bool(executable))
            ).encode("utf-8")
        ).hexdigest()
        return CrossDomainFidelityArmDecision(
            arm_id=self.arm_id,
            authority_granted=authority_granted,
            fidelity_score=float(fidelity_score.detach().cpu().item()),
            authority_score=float(authority_score.detach().cpu().item()),
            verifier_score=float(verifier_score.detach().cpu().item()),
            receipt_semantics=receipt_semantics,
            neural_accounted_flops=self._neural_accounted_flops(),
            arm_observable_digest=observable_digest,
        )


class CompileOnlyControlArm(_MatchedCrossDomainFidelityArmBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(arm_id="compile_only_control", **kwargs)


class CrossDomainFidelityFabricArm(_MatchedCrossDomainFidelityArmBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(arm_id="cross_domain_fidelity_fabric", **kwargs)


def build_matched_cross_domain_fidelity_arms(
    *,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    device: str | torch.device | None = None,
) -> tuple[CompileOnlyControlArm, CrossDomainFidelityFabricArm]:
    control = CompileOnlyControlArm(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        device=device,
    )
    fidelity = CrossDomainFidelityFabricArm(
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        device=device,
    )
    fidelity.load_state_dict(control.state_dict())
    return control, fidelity


def _parameter_audit(arm: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in arm.parameters())
    reserve = sum(
        parameter.numel()
        for name, parameter in arm.named_parameters()
        if "capacity_reserve" in name
    )
    functional = total - reserve
    optimizer_visible = sum(
        parameter.numel()
        for _, parameter in functional_trainable_named_parameters(arm)
    )
    return {
        "total_parameters": int(total),
        "functional_parameters": int(functional),
        "active_functional_parameters": int(functional),
        "optimizer_visible_parameters": int(optimizer_visible),
        "reserved_parameters": int(reserve),
    }


def _state_digest(arm: nn.Module) -> str:
    digest = sha256()
    for name, tensor in arm.state_dict().items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(bytes(contiguous.untyped_storage()))
    return digest.hexdigest()


def audit_matched_cross_domain_fidelity_arms(
    control: CompileOnlyControlArm,
    fidelity: CrossDomainFidelityFabricArm,
) -> dict[str, Any]:
    control_audit = _parameter_audit(control)
    fidelity_audit = _parameter_audit(fidelity)
    control_flops = control._neural_accounted_flops()
    fidelity_flops = fidelity._neural_accounted_flops()
    return {
        "schema": "NLM-EXP-298-MATCHED-CROSS-DOMAIN-FIDELITY-ARMS-DEV-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "parameter_match": control_audit["total_parameters"] == fidelity_audit["total_parameters"],
        "functional_parameter_match": control_audit["functional_parameters"] == fidelity_audit["functional_parameters"],
        "active_functional_parameter_match": control_audit["active_functional_parameters"] == fidelity_audit["active_functional_parameters"],
        "optimizer_visible_parameter_match": control_audit["optimizer_visible_parameters"] == fidelity_audit["optimizer_visible_parameters"],
        "initialization_match": _state_digest(control) == _state_digest(fidelity),
        "neural_accounted_flops_match": control_flops == fidelity_flops,
        "receipt_access_is_only_causal_intervention": True,
        "arms": {
            "compile_only_control": control_audit,
            "cross_domain_fidelity_fabric": fidelity_audit,
        },
        "compute_ledger": {
            "compile_only_control": {
                "neural_accounted_flops": control_flops,
                "semantic_verification_operations": 0,
                "hardware_profiler_flops_claimed": False,
            },
            "cross_domain_fidelity_fabric": {
                "neural_accounted_flops": fidelity_flops,
                "semantic_verification_operations": "charged_per_candidate_from_receipt",
                "hardware_profiler_flops_claimed": False,
            },
        },
        "label_information_consumed": False,
        "trap_family_consumed": False,
    }
