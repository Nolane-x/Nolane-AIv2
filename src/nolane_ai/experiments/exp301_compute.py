from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from nolane_ai.protocol.v017 import (
    EXP301_CHALLENGE_LOOPS,
    EXP301_MAX_RELATIVE_FLOP_MISMATCH,
)


# V2 closes a pre-freeze accounting gap: scientific prediction receipts charge
# every autoregressive decode forward rather than only a single prompt pass.
COMPUTE_LEDGER_VERSION = "exp301-flops-v2"
ARM_IDS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
VOCAB_SIZE = 4_608
D_MODEL = 448
D_FF = 1_152
SHARED_LAYERS = 3


@dataclass(frozen=True, slots=True)
class ArmComputeReceipt:
    ledger_version: str
    arm_id: str
    effort_multiplier: int
    sequence_length: int
    repeated_core_flops: int
    conditioning_flops: int
    restart_code_flops: int
    aggregation_flops: int
    finalization_flops: int
    output_projection_flops: int
    total_flops: int


@dataclass(frozen=True, slots=True)
class ComputeMatchReceipt:
    ledger_version: str
    effort_multiplier: int
    sequence_length: int
    flops_by_arm: dict[str, int]
    relative_mismatch: float
    status: str


@dataclass(frozen=True, slots=True)
class AutoregressiveComputeReceipt:
    ledger_version: str
    arm_id: str
    effort_multiplier: int
    prompt_token_count: int
    generated_token_count: int
    decode_sequence_lengths: tuple[int, ...]
    total_flops: int


@dataclass(frozen=True, slots=True)
class AutoregressiveComputeMatchReceipt:
    ledger_version: str
    effort_multiplier: int
    prompt_token_count: int
    generated_token_count: int
    flops_by_arm: dict[str, int]
    relative_mismatch: float
    status: str


def _validate_geometry_inputs(*, effort_multiplier: int, sequence_length: int) -> None:
    if effort_multiplier not in EXP301_CHALLENGE_LOOPS:
        raise ValueError(
            f"effort_multiplier must be one of {EXP301_CHALLENGE_LOOPS}, got {effort_multiplier}"
        )
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")


def _transformer_layer_flops(sequence_length: int) -> int:
    """Frozen analytical FLOP convention for one 448/1152 Transformer layer.

    Multiplication plus accumulation counts as two FLOPs. Causal attention uses
    the exact lower-triangular pair count. Norm/activation/residual elementwise
    work is deliberately explicit so no arm gets free repeated computation.
    """

    t = sequence_length
    d = D_MODEL
    d_ff = D_FF
    causal_pairs = t * (t + 1) // 2

    qkv_projection = 6 * t * d * d
    output_projection = 2 * t * d * d
    attention_score_and_value = 4 * causal_pairs * d
    two_rmsnorms = 10 * t * d
    swiglu_projections = 6 * t * d * d_ff
    swiglu_elementwise = 6 * t * d_ff
    residual_additions = 2 * t * d

    return (
        qkv_projection
        + output_projection
        + attention_score_and_value
        + two_rmsnorms
        + swiglu_projections
        + swiglu_elementwise
        + residual_additions
    )


def _capacity_exchange_flops(
    sequence_length: int,
    *,
    bottleneck: int,
    tail_parameters: int = 192,
) -> int:
    t = sequence_length
    d = D_MODEL

    two_linear_projections = 4 * t * d * bottleneck
    silu = 4 * t * bottleneck
    residual_add = t * d
    tail = 0
    if tail_parameters:
        tail = 2 * t * tail_parameters + 2 * t * d
    return two_linear_projections + silu + residual_add + tail


def _one_repeated_core_flops(sequence_length: int, *, arm_id: str) -> int:
    bottleneck = 973 if arm_id == "C_NRS_CORE" else 981
    return (
        SHARED_LAYERS * _transformer_layer_flops(sequence_length)
        + _capacity_exchange_flops(sequence_length, bottleneck=bottleneck)
    )


def account_arm_flops(
    arm_id: str,
    *,
    effort_multiplier: int,
    sequence_length: int,
) -> ArmComputeReceipt:
    if arm_id not in ARM_IDS:
        raise ValueError(f"arm must be one of {ARM_IDS}, got {arm_id!r}")
    _validate_geometry_inputs(
        effort_multiplier=effort_multiplier,
        sequence_length=sequence_length,
    )

    repeated_core_flops = effort_multiplier * _one_repeated_core_flops(
        sequence_length,
        arm_id=arm_id,
    )
    conditioning_flops = 0
    restart_code_flops = 0
    aggregation_flops = 0

    if arm_id == "C_NRS_CORE":
        conditioning_flops = effort_multiplier * sequence_length * D_MODEL
    elif arm_id == "A_FIXED":
        restart_code_flops = effort_multiplier * sequence_length * D_MODEL
        if effort_multiplier > 1:
            aggregation_flops = effort_multiplier * sequence_length * D_MODEL

    finalization_flops = 5 * sequence_length * D_MODEL
    output_projection_flops = 2 * sequence_length * D_MODEL * VOCAB_SIZE
    total_flops = (
        repeated_core_flops
        + conditioning_flops
        + restart_code_flops
        + aggregation_flops
        + finalization_flops
        + output_projection_flops
    )

    return ArmComputeReceipt(
        ledger_version=COMPUTE_LEDGER_VERSION,
        arm_id=arm_id,
        effort_multiplier=effort_multiplier,
        sequence_length=sequence_length,
        repeated_core_flops=repeated_core_flops,
        conditioning_flops=conditioning_flops,
        restart_code_flops=restart_code_flops,
        aggregation_flops=aggregation_flops,
        finalization_flops=finalization_flops,
        output_projection_flops=output_projection_flops,
        total_flops=total_flops,
    )


def account_autoregressive_flops(
    arm_id: str,
    *,
    effort_multiplier: int,
    prompt_token_count: int,
    generated_token_count: int,
) -> AutoregressiveComputeReceipt:
    if prompt_token_count <= 0:
        raise ValueError("prompt_token_count must be positive")
    if generated_token_count <= 0:
        raise ValueError("generated_token_count must be positive")
    sequence_lengths = tuple(prompt_token_count + offset for offset in range(generated_token_count))
    total = sum(
        account_arm_flops(
            arm_id,
            effort_multiplier=effort_multiplier,
            sequence_length=sequence_length,
        ).total_flops
        for sequence_length in sequence_lengths
    )
    return AutoregressiveComputeReceipt(
        ledger_version=COMPUTE_LEDGER_VERSION,
        arm_id=arm_id,
        effort_multiplier=effort_multiplier,
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        decode_sequence_lengths=sequence_lengths,
        total_flops=total,
    )


def validate_compute_match(
    flops_by_arm: Mapping[str, int],
    *,
    tolerance: float = EXP301_MAX_RELATIVE_FLOP_MISMATCH,
) -> tuple[str, float]:
    if set(flops_by_arm) != set(ARM_IDS):
        raise ValueError(f"compute match must contain exactly {ARM_IDS}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    values = [int(flops_by_arm[arm]) for arm in ARM_IDS]
    if any(value <= 0 for value in values):
        raise ValueError("all accounted FLOP totals must be positive")

    minimum = min(values)
    maximum = max(values)
    relative_mismatch = (maximum - minimum) / minimum
    status = (
        "VALID_COMPUTE_MATCH"
        if relative_mismatch <= tolerance
        else "INVALID_COMPUTE_MATCH"
    )
    return status, relative_mismatch


def match_common_compute(
    *,
    effort_multiplier: int,
    sequence_length: int,
) -> ComputeMatchReceipt:
    receipts = {
        arm: account_arm_flops(
            arm,
            effort_multiplier=effort_multiplier,
            sequence_length=sequence_length,
        )
        for arm in ARM_IDS
    }
    flops_by_arm = {arm: receipts[arm].total_flops for arm in ARM_IDS}
    status, mismatch = validate_compute_match(flops_by_arm)
    return ComputeMatchReceipt(
        ledger_version=COMPUTE_LEDGER_VERSION,
        effort_multiplier=effort_multiplier,
        sequence_length=sequence_length,
        flops_by_arm=flops_by_arm,
        relative_mismatch=mismatch,
        status=status,
    )


def match_autoregressive_compute(
    *,
    effort_multiplier: int,
    prompt_token_count: int,
    generated_token_count: int,
) -> AutoregressiveComputeMatchReceipt:
    receipts = {
        arm: account_autoregressive_flops(
            arm,
            effort_multiplier=effort_multiplier,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
        )
        for arm in ARM_IDS
    }
    flops_by_arm = {arm: receipt.total_flops for arm, receipt in receipts.items()}
    status, mismatch = validate_compute_match(flops_by_arm)
    return AutoregressiveComputeMatchReceipt(
        ledger_version=COMPUTE_LEDGER_VERSION,
        effort_multiplier=effort_multiplier,
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        flops_by_arm=flops_by_arm,
        relative_mismatch=mismatch,
        status=status,
    )
