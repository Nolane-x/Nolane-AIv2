from __future__ import annotations

EXPERIMENT_ID = "EXP-286"
PRIMARY_ENDPOINT = "accounted_reasoning_flops_to_verified_solution"
PRIMARY_DIRECTION = "lower"
MESI_RELATIVE_REDUCTION = 0.15
POWER_TARGET = 0.90
MIN_N = 32
MAX_N = 128
MULTIPLICITY_FAMILY = "CONFLICT_VALUE"
ANALYSIS_METHOD = (
    "paired log-cost ratio and bootstrap CI; failures included as "
    "censored/scientific outcomes per frozen rule"
)
PROTECTED_SOLUTION_FLOOR = (
    "oracle_conflict_core >= chronological_failure - 0.005"
)


def frozen_exp286_gate_a_contract() -> dict[str, object]:
    """Return the frozen Stage-A authority boundary for EXP-286 Gate A.

    This is preparation metadata only. It deliberately carries no challenge
    materialization, confirmatory observations, or evidence promotion.
    """

    return {
        "experiment_id": EXPERIMENT_ID,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": PRIMARY_DIRECTION,
        "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
        "power_target": POWER_TARGET,
        "min_n": MIN_N,
        "max_n": MAX_N,
        "paired": True,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "analysis_method": ANALYSIS_METHOD,
        "protected_solution_floor": PROTECTED_SOLUTION_FLOOR,
        "pilot_reuse_as_confirmatory": False,
        "challenge_seed_materialized": False,
        "confirmatory_data_consumed": False,
    }
