from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist, stdev
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256

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
SCHEMA = "NLM-EXP-286-CONFIRMATORY-PREP-V1"
PLANNING_ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10_000
SAMPLE_SIZE_METHOD = "paired-log-cost-normal-approximation-from-development-sd-v1"


def _prep_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def frozen_exp286_gate_a_contract() -> dict[str, object]:
    """Return the frozen Stage-A authority boundary for EXP-286 Gate A."""

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


def _validate_frozen_experiment(experiment: dict[str, Any]) -> None:
    if experiment.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("confirmatory prep requires EXP-286")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != PRIMARY_ENDPOINT or primary.get("direction") != PRIMARY_DIRECTION:
        raise ValueError("EXP-286 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "relative_reduction" or float(mesi.get("value", -1.0)) != MESI_RELATIVE_REDUCTION:
        raise ValueError("EXP-286 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if float(sample.get("power_target", -1.0)) != POWER_TARGET:
        raise ValueError("EXP-286 power target drift")
    if int(sample.get("min_n", -1)) != MIN_N or int(sample.get("max_n", -1)) != MAX_N:
        raise ValueError("EXP-286 sample-size bounds drift")
    if sample.get("paired") is not True:
        raise ValueError("EXP-286 paired-analysis contract drift")
    if experiment.get("analysis_method") != ANALYSIS_METHOD:
        raise ValueError("EXP-286 analysis method drift")
    if experiment.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        raise ValueError("EXP-286 multiplicity family drift")
    protected = {
        item.get("metric"): item.get("floor")
        for item in experiment.get("protected_endpoints") or []
    }
    if protected.get("verified_solution_rate") != PROTECTED_SOLUTION_FLOOR:
        raise ValueError("EXP-286 protected solution-rate floor drift")
    challenge = experiment.get("challenge_generator") or {}
    if challenge.get("lane") != "POST_FREEZE_CHALLENGE":
        raise ValueError("EXP-286 challenge lane drift")


def _development_rows(
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    from nolane_ai.experiments.exp286_paired_runner import validate_exp286_paired_development

    errors = validate_exp286_paired_development(execution_artifact)
    if errors:
        raise ValueError("invalid EXP-286 development artifact: " + "; ".join(errors))
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-286 confirmatory prep requires EV-E2 / UNVERIFIED development evidence")
    for field in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_seed_materialized",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if execution_artifact.get(field) is not False:
            raise ValueError(f"EXP-286 development boundary requires {field}=false")

    evaluation = execution_artifact.get("evaluation") or {}
    rows = list(evaluation.get("per_replicate") or [])
    if len(rows) < MIN_N:
        raise ValueError("EXP-286 confirmatory prep requires at least 32 paired pilot replicates")
    if int(evaluation.get("replicates", 0) or 0) != len(rows):
        raise ValueError("EXP-286 pilot replicate count mismatch")
    replicate_ids = [row.get("replicate") for row in rows]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in replicate_ids):
        raise ValueError("EXP-286 pilot replicate IDs must be non-negative integers")
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("EXP-286 pilot replicate IDs must be unique")

    training = execution_artifact.get("training") or {}
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    if training.get("rng_stream") != "augmentation" or train_start != 0 or train_count <= 0:
        raise ValueError("EXP-286 DEVELOPMENT training lineage is incomplete")
    training_ids = list(range(train_start, train_start + train_count))
    if set(training_ids).intersection(replicate_ids):
        raise ValueError("EXP-286 training and pilot replicate lineages overlap")

    for row in rows:
        if row.get("initial_world_pairing_closed") is not True or not row.get("paired_batch_digest"):
            raise ValueError("EXP-286 paired pilot world lineage is incomplete")
        for arm in ("chronological_failure", "oracle_conflict_core"):
            metrics = row.get(arm) or {}
            cost = metrics.get(PRIMARY_ENDPOINT)
            solution = metrics.get("verified_solution_rate")
            if not isinstance(cost, (int, float)) or not math.isfinite(float(cost)) or float(cost) <= 0.0:
                raise ValueError("EXP-286 pilot cost must be finite and positive")
            if not isinstance(solution, (int, float)) or not math.isfinite(float(solution)):
                raise ValueError("EXP-286 pilot solution rate must be finite")

    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-286 confirmatory prep requires neural arm registry")
    if arm_registry.get("evidence_level") != "EV-E2" or arm_registry.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-286 arm registry must remain EV-E2 / UNVERIFIED")
    protocol_digest = execution_artifact.get("protocol_digest")
    if not protocol_digest or arm_registry.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-286 registry/execution protocol digest mismatch")
    registry_exp = (arm_registry.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    if registry_exp.get("development_match_status") != "PAIRED_CONFLICT_HEADROOM_DEV_READY":
        raise ValueError("EXP-286 registry development match is not ready")
    evidence = registry_exp.get("paired_execution_evidence") or {}
    if evidence.get("artifact_digest") != execution_artifact.get("artifact_digest"):
        raise ValueError("EXP-286 registry execution artifact digest mismatch")
    if evidence.get("code_digest") not in (None, execution_artifact.get("code_digest")):
        raise ValueError("EXP-286 registry execution code digest mismatch")

    return rows, training_ids, [int(item) for item in replicate_ids]


def _paired_log_cost_summary(rows: list[dict[str, Any]]) -> tuple[float, float, float, list[float]]:
    chronological = [float(row["chronological_failure"][PRIMARY_ENDPOINT]) for row in rows]
    oracle = [float(row["oracle_conflict_core"][PRIMARY_ENDPOINT]) for row in rows]
    log_ratios = [math.log(o / c) for c, o in zip(chronological, oracle, strict=True)]
    if any(not math.isfinite(value) for value in log_ratios):
        raise ValueError("EXP-286 paired log-cost pilot contains non-finite values")
    paired_sd = stdev(log_ratios) if len(log_ratios) > 1 else 0.0
    return (
        sum(chronological) / len(chronological),
        sum(oracle) / len(oracle),
        float(paired_sd),
        log_ratios,
    )


def _required_n(*, paired_sd: float) -> int:
    if not math.isfinite(paired_sd) or paired_sd < 0.0:
        raise ValueError("paired_sd must be finite and non-negative")
    if paired_sd == 0.0:
        return 1
    target_log_reduction = abs(math.log(1.0 - MESI_RELATIVE_REDUCTION))
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - PLANNING_ALPHA)
    z_power = normal.inv_cdf(POWER_TARGET)
    return max(1, int(math.ceil(((z_alpha + z_power) * paired_sd / target_log_reduction) ** 2)))


def validate_exp286_confirmatory_prep(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-286 confirmatory prep schema")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-286 confirmatory prep experiment id drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-286 confirmatory prep cannot promote scientific evidence")
    for field in ("confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(field) is not False:
            errors.append(f"EXP-286 confirmatory prep requires {field}=false")

    frozen = payload.get("frozen_analysis") or {}
    expected_frozen = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": PRIMARY_DIRECTION,
        "effect_type": "paired_log_cost_ratio_relative_reduction",
        "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
        "planning_alpha": PLANNING_ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "analysis_method": ANALYSIS_METHOD,
        "protected_solution_floor": PROTECTED_SOLUTION_FLOOR,
    }
    if frozen != expected_frozen:
        errors.append("EXP-286 frozen analysis contract drift")

    pilot = payload.get("pilot_summary") or {}
    if pilot.get("source_lane") != "DEVELOPMENT_PILOT_ONLY" or int(pilot.get("n", 0) or 0) < MIN_N:
        errors.append("EXP-286 pilot summary is not eligible for Gate A planning")
    paired_sd = pilot.get("paired_log_cost_sd")
    if not isinstance(paired_sd, (int, float)) or not math.isfinite(float(paired_sd)) or float(paired_sd) < 0.0:
        errors.append("EXP-286 paired log-cost SD is invalid")

    freeze = payload.get("sample_size_freeze") or {}
    if freeze.get("method") != SAMPLE_SIZE_METHOD:
        errors.append("EXP-286 sample-size method drift")
    if float(freeze.get("power_target", -1.0)) != POWER_TARGET:
        errors.append("EXP-286 power target drift")
    if int(freeze.get("min_n", -1)) != MIN_N or int(freeze.get("max_n", -1)) != MAX_N:
        errors.append("EXP-286 sample-size bounds drift")
    if freeze.get("paired") is not True or freeze.get("pilot_reuse_as_confirmatory") is not False:
        errors.append("EXP-286 paired/pilot-reuse sample-size boundary drift")
    required = freeze.get("unclamped_required_n")
    if not isinstance(required, int) or isinstance(required, bool) or required < 1:
        errors.append("EXP-286 unclamped required n is invalid")
        required = None

    lineage = payload.get("confirmatory_lineage") or {}
    if lineage.get("lane") != "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED":
        errors.append("EXP-286 confirmatory lane drift")
    if lineage.get("pilot_reuse_forbidden") is not True:
        errors.append("EXP-286 confirmatory lineage must forbid pilot reuse")
    if lineage.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-286 challenge seed cannot materialize during Gate A prep")

    status = payload.get("status")
    ready = payload.get("confirmatory_ready")
    confirmatory_n = freeze.get("confirmatory_n")
    reserved = list(lineage.get("reserved_replicate_ids") or [])
    if required is not None and required > MAX_N:
        if status != "NOT_READY_VARIANCE_EXCEEDS_MAX_N" or ready is not False or confirmatory_n is not None or reserved:
            errors.append("EXP-286 variance-exceeds-max status must remain non-ready and unreserved")
    elif required is not None:
        expected_n = max(MIN_N, required)
        if status != "CONFIRMATORY_GATE_A_PREPARED" or ready is not True or confirmatory_n != expected_n:
            errors.append("EXP-286 prepared Gate A sample-size state drift")
        if len(reserved) != expected_n or len(set(reserved)) != len(reserved):
            errors.append("EXP-286 reserved confirmatory replicate IDs are incomplete")

    if payload.get("prep_digest") not in (None, "") and payload.get("prep_digest") != _prep_digest(payload):
        errors.append("EXP-286 confirmatory prep digest mismatch")
    return errors


def build_exp286_confirmatory_prep(
    *,
    experiment: dict[str, Any],
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    if not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")
    _validate_frozen_experiment(experiment)
    rows, training_ids, replicate_ids = _development_rows(execution_artifact, arm_registry)
    mean_chronological, mean_oracle, paired_sd, log_ratios = _paired_log_cost_summary(rows)
    required = _required_n(paired_sd=paired_sd)
    confirmatory_n = max(MIN_N, required) if required <= MAX_N else None
    ready = confirmatory_n is not None
    status = "CONFIRMATORY_GATE_A_PREPARED" if ready else "NOT_READY_VARIANCE_EXCEEDS_MAX_N"

    reserved: list[int] = []
    if ready:
        lineage_max = max(training_ids + replicate_ids)
        reserved = list(range(lineage_max + 1, lineage_max + 1 + int(confirmatory_n)))

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": status,
        "confirmatory_ready": bool(ready),
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "protocol_digest": execution_artifact.get("protocol_digest"),
        "development_execution_digest": execution_artifact.get("artifact_digest"),
        "arm_registry_digest": arm_registry.get("registry_digest"),
        "analysis_code_digest": analysis_code_digest,
        "frozen_analysis": {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "primary_direction": PRIMARY_DIRECTION,
            "effect_type": "paired_log_cost_ratio_relative_reduction",
            "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
            "planning_alpha": PLANNING_ALPHA,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "multiplicity_family": MULTIPLICITY_FAMILY,
            "analysis_method": ANALYSIS_METHOD,
            "protected_solution_floor": PROTECTED_SOLUTION_FLOOR,
        },
        "pilot_summary": {
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
            "n": len(rows),
            "training_replicate_ids": training_ids,
            "replicate_ids": replicate_ids,
            "mean_chronological_cost": mean_chronological,
            "mean_oracle_cost": mean_oracle,
            "paired_mean_log_cost_ratio": sum(log_ratios) / len(log_ratios),
            "paired_log_cost_sd": paired_sd,
        },
        "sample_size_freeze": {
            "method": SAMPLE_SIZE_METHOD,
            "power_target": POWER_TARGET,
            "planning_alpha": PLANNING_ALPHA,
            "min_n": MIN_N,
            "max_n": MAX_N,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "unclamped_required_n": required,
            "confirmatory_n": confirmatory_n,
            "planning_constants": {
                "target_log_reduction": abs(math.log(1.0 - MESI_RELATIVE_REDUCTION)),
                "z_alpha": NormalDist().inv_cdf(1.0 - PLANNING_ALPHA),
                "z_power": NormalDist().inv_cdf(POWER_TARGET),
            },
        },
        "confirmatory_lineage": {
            "lane": "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "seed_materialization_status": "NOT_EXECUTED",
            "reserved_replicate_ids": reserved,
        },
        "prep_digest": "",
    }
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp286_confirmatory_prep(payload)
    if errors:
        raise RuntimeError("invalid EXP-286 confirmatory prep: " + "; ".join(errors))
    return payload
