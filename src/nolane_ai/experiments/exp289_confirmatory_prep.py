from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist, mean, stdev
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


EXPERIMENT_ID = "EXP-289"
PRIMARY_ENDPOINT = "repeat_dead_end_rate"
PRIMARY_DIRECTION = "lower"
MESI_RELATIVE_REDUCTION = 0.25
POWER_TARGET = 0.90
MIN_N = 32
MAX_N = 128
FAMILYWISE_ALPHA = 0.05
MULTIPLICITY_FAMILY = "LOCAL_NOGOOD"
ANALYSIS_METHOD = "paired relative RDER reduction with bootstrap CI and explicit over-prune safety bound"
PROTECTED_OVERPRUNE_CEILING = 0.005
PROTECTED_SOLUTION_FLOOR = "local_nogood >= no_nogood - 0.01"
ZERO_OPPORTUNITY_POLICY = "retain_raw_episode_exclude_from_rder_denominator"
SCHEMA = "NLM-EXP-289-CONFIRMATORY-PREP-V1"
SAMPLE_SIZE_METHOD = "paired-relative-rder-normal-approximation-from-development-sd-v1"


def _prep_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def frozen_exp289_gate_a_contract() -> dict[str, object]:
    """Return only authority frozen by Stage-A plus explicit pre-beacon safeguards."""

    return {
        "experiment_id": EXPERIMENT_ID,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": PRIMARY_DIRECTION,
        "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
        "power_target": POWER_TARGET,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "min_n": MIN_N,
        "max_n": MAX_N,
        "paired": True,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "analysis_method": ANALYSIS_METHOD,
        "zero_opportunity_policy": ZERO_OPPORTUNITY_POLICY,
        "epsilon_denominator_rescue": False,
        "protected_overprune_ceiling": PROTECTED_OVERPRUNE_CEILING,
        "protected_solution_floor": PROTECTED_SOLUTION_FLOOR,
        "pilot_reuse_as_confirmatory": False,
        "challenge_seed_materialized": False,
        "confirmatory_data_consumed": False,
    }


def _validate_frozen_experiment(
    experiment: dict[str, Any],
    *,
    familywise_alpha: float,
) -> None:
    if experiment.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("confirmatory prep requires EXP-289")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != PRIMARY_ENDPOINT or primary.get("direction") != PRIMARY_DIRECTION:
        raise ValueError("EXP-289 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "relative_reduction" or float(mesi.get("value", -1.0)) != MESI_RELATIVE_REDUCTION:
        raise ValueError("EXP-289 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if float(sample.get("power_target", -1.0)) != POWER_TARGET:
        raise ValueError("EXP-289 power target drift")
    if int(sample.get("min_n", -1)) != MIN_N or int(sample.get("max_n", -1)) != MAX_N:
        raise ValueError("EXP-289 sample-size bounds drift")
    if sample.get("paired") is not True:
        raise ValueError("EXP-289 paired-analysis contract drift")
    if float(familywise_alpha) != FAMILYWISE_ALPHA:
        raise ValueError("EXP-289 familywise alpha drift")
    if experiment.get("analysis_method") != ANALYSIS_METHOD:
        raise ValueError("EXP-289 analysis method drift")
    if experiment.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        raise ValueError("EXP-289 multiplicity family drift")
    protected = {
        item.get("metric"): item.get("floor")
        for item in experiment.get("protected_endpoints") or []
    }
    if float(protected.get("valid_state_overprune_rate", -1.0)) != PROTECTED_OVERPRUNE_CEILING:
        raise ValueError("EXP-289 protected over-prune ceiling drift")
    if protected.get("verified_solution_rate") != PROTECTED_SOLUTION_FLOOR:
        raise ValueError("EXP-289 protected solution-rate floor drift")
    resource = experiment.get("resource_match") or {}
    if resource.get("memory_cost") != "nogood storage/retrieval charged to accounted cost":
        raise ValueError("EXP-289 memory-cost accounting drift")
    challenge = experiment.get("challenge_generator") or {}
    if challenge.get("lane") != "POST_FREEZE_CHALLENGE":
        raise ValueError("EXP-289 challenge lane drift")


def _development_rows(
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    from nolane_ai.experiments.exp289_paired_runner import validate_exp289_paired_development

    errors = validate_exp289_paired_development(execution_artifact)
    if errors:
        raise ValueError("invalid EXP-289 development artifact: " + "; ".join(errors))
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-289 confirmatory prep requires EV-E2 / UNVERIFIED development evidence")
    for field in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_seed_materialized",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if execution_artifact.get(field) is not False:
            raise ValueError(f"EXP-289 development boundary requires {field}=false")

    evaluation = execution_artifact.get("evaluation") or {}
    rows = list(evaluation.get("per_replicate") or [])
    if len(rows) < MIN_N:
        raise ValueError("EXP-289 confirmatory prep requires at least 32 paired pilot replicates")
    if int(evaluation.get("replicates", 0) or 0) != len(rows):
        raise ValueError("EXP-289 pilot replicate count mismatch")
    replicate_ids = [row.get("replicate") for row in rows]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in replicate_ids):
        raise ValueError("EXP-289 pilot replicate IDs must be non-negative integers")
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("EXP-289 pilot replicate IDs must be unique")

    training = execution_artifact.get("training") or {}
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    if training.get("rng_stream") != "augmentation" or train_start != 0 or train_count <= 0:
        raise ValueError("EXP-289 DEVELOPMENT training lineage is incomplete")
    training_ids = list(range(train_start, train_start + train_count))
    if set(training_ids).intersection(replicate_ids):
        raise ValueError("EXP-289 training and pilot replicate lineages overlap")

    for row in rows:
        if row.get("initial_world_pairing_closed") is not True:
            raise ValueError("EXP-289 paired pilot world lineage is incomplete")
        if row.get("opportunity_manifest_pairing_closed") is not True:
            raise ValueError("EXP-289 opportunity manifest pairing is not closed")
        if row.get("evaluator_metadata_delivered_to_arm") is not False:
            raise ValueError("EXP-289 evaluator metadata leaked into arm execution")
        for arm in ("no_nogood", "local_nogood"):
            metrics = row.get(arm) or {}
            if metrics.get("predeclared_repeat_opportunities") is None:
                raise ValueError("EXP-289 predeclared RDER opportunity denominator is missing")
            rate = metrics.get(PRIMARY_ENDPOINT)
            if rate is None or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
                raise ValueError("EXP-289 pilot RDER must be finite and defined")
            solution = metrics.get("verified_solution_rate")
            if not isinstance(solution, (int, float)) or not math.isfinite(float(solution)):
                raise ValueError("EXP-289 pilot solution rate must be finite")
        if row["no_nogood"]["predeclared_repeat_opportunities"] != row["local_nogood"]["predeclared_repeat_opportunities"]:
            raise ValueError("EXP-289 paired RDER denominator drift")

    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-289 confirmatory prep requires neural arm registry")
    if arm_registry.get("evidence_level") != "EV-E2" or arm_registry.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-289 arm registry must remain EV-E2 / UNVERIFIED")
    protocol_digest = execution_artifact.get("protocol_digest")
    if not protocol_digest or arm_registry.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-289 registry/execution protocol digest mismatch")
    registry_exp = (arm_registry.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    if registry_exp.get("development_match_status") != "PAIRED_LOCAL_NOGOOD_DEV_READY":
        raise ValueError("EXP-289 registry development match is not ready")
    evidence = registry_exp.get("paired_execution_evidence") or {}
    if evidence.get("artifact_digest") != execution_artifact.get("artifact_digest"):
        raise ValueError("EXP-289 registry execution artifact digest mismatch")
    if evidence.get("code_digest") not in (None, execution_artifact.get("code_digest")):
        raise ValueError("EXP-289 registry execution code digest mismatch")

    return rows, training_ids, [int(item) for item in replicate_ids]


def _paired_relative_rder_effects(rows: list[dict[str, Any]]) -> list[float]:
    effects: list[float] = []
    for row in rows:
        baseline = (row.get("no_nogood") or {}).get(PRIMARY_ENDPOINT)
        candidate = (row.get("local_nogood") or {}).get(PRIMARY_ENDPOINT)
        if baseline is None or candidate is None:
            raise ValueError("EXP-289 pilot RDER must be finite and defined")
        try:
            baseline_value = float(baseline)
            candidate_value = float(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError("EXP-289 pilot RDER must be finite and defined") from exc
        if not math.isfinite(baseline_value) or not math.isfinite(candidate_value):
            raise ValueError("EXP-289 pilot RDER must be finite and defined")
        if baseline_value <= 0.0:
            raise ValueError(
                "EXP-289 baseline RDER denominator must be strictly positive; no epsilon rescue"
            )
        effects.append((baseline_value - candidate_value) / baseline_value)
    return effects


def _required_n(*, paired_sd: float, familywise_alpha: float) -> int:
    if not math.isfinite(paired_sd) or paired_sd < 0.0:
        raise ValueError("paired_sd must be finite and non-negative")
    if paired_sd == 0.0:
        return 1
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - familywise_alpha)
    z_power = normal.inv_cdf(POWER_TARGET)
    return max(
        1,
        int(math.ceil(((z_alpha + z_power) * paired_sd / MESI_RELATIVE_REDUCTION) ** 2)),
    )


def validate_exp289_confirmatory_prep(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-289 confirmatory prep schema")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-289 confirmatory prep experiment id drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-289 confirmatory prep cannot promote scientific evidence")
    for field in ("confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(field) is not False:
            errors.append(f"EXP-289 confirmatory prep requires {field}=false")

    expected_frozen = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": PRIMARY_DIRECTION,
        "effect_type": "paired_relative_rder_reduction",
        "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "analysis_method": ANALYSIS_METHOD,
        "zero_opportunity_policy": ZERO_OPPORTUNITY_POLICY,
        "epsilon_denominator_rescue": False,
        "protected_overprune_ceiling": PROTECTED_OVERPRUNE_CEILING,
        "protected_solution_floor": PROTECTED_SOLUTION_FLOOR,
    }
    if payload.get("frozen_analysis") != expected_frozen:
        errors.append("EXP-289 frozen analysis contract drift")

    pilot = payload.get("pilot_summary") or {}
    if pilot.get("source_lane") != "DEVELOPMENT_PILOT_ONLY":
        errors.append("EXP-289 pilot source lane drift")
    try:
        pilot_n = int(pilot.get("n", 0) or 0)
        eligible_n = int(pilot.get("eligible_paired_rder_replicates", 0) or 0)
    except (TypeError, ValueError):
        pilot_n = eligible_n = 0
    if pilot_n < MIN_N or eligible_n != pilot_n:
        errors.append("EXP-289 pilot summary is not eligible for Gate A planning")
    paired_sd = pilot.get("paired_relative_rder_reduction_sd")
    if not isinstance(paired_sd, (int, float)) or not math.isfinite(float(paired_sd)) or float(paired_sd) < 0.0:
        errors.append("EXP-289 paired relative RDER SD is invalid")

    freeze = payload.get("sample_size_freeze") or {}
    if freeze.get("method") != SAMPLE_SIZE_METHOD:
        errors.append("EXP-289 sample-size method drift")
    if float(freeze.get("power_target", -1.0)) != POWER_TARGET:
        errors.append("EXP-289 power target drift")
    if float(freeze.get("familywise_alpha", -1.0)) != FAMILYWISE_ALPHA:
        errors.append("EXP-289 familywise alpha drift")
    if int(freeze.get("min_n", -1)) != MIN_N or int(freeze.get("max_n", -1)) != MAX_N:
        errors.append("EXP-289 sample-size bounds drift")
    if freeze.get("paired") is not True or freeze.get("pilot_reuse_as_confirmatory") is not False:
        errors.append("EXP-289 paired/pilot-reuse sample-size boundary drift")
    required = freeze.get("unclamped_required_n")
    if not isinstance(required, int) or isinstance(required, bool) or required < 1:
        errors.append("EXP-289 unclamped required n is invalid")
        required = None

    lineage = payload.get("confirmatory_lineage") or {}
    if lineage.get("lane") != "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED":
        errors.append("EXP-289 confirmatory lane drift")
    if lineage.get("pilot_reuse_forbidden") is not True:
        errors.append("EXP-289 confirmatory lineage must forbid pilot reuse")
    if lineage.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-289 challenge seed cannot materialize during Gate A prep")

    status = payload.get("status")
    ready = payload.get("confirmatory_ready")
    confirmatory_n = freeze.get("confirmatory_n")
    reserved = list(lineage.get("reserved_replicate_ids") or [])
    if required is not None and required > MAX_N:
        if status != "NOT_READY_VARIANCE_EXCEEDS_MAX_N" or ready is not False or confirmatory_n is not None or reserved:
            errors.append("EXP-289 variance-exceeds-max status must remain non-ready and unreserved")
    elif required is not None:
        expected_n = max(MIN_N, required)
        if status != "CONFIRMATORY_GATE_A_PREPARED" or ready is not True or confirmatory_n != expected_n:
            errors.append("EXP-289 prepared Gate A sample-size state drift")
        if len(reserved) != expected_n or len(set(reserved)) != len(reserved):
            errors.append("EXP-289 reserved confirmatory replicate IDs are incomplete")

    if payload.get("prep_digest") not in (None, "") and payload.get("prep_digest") != _prep_digest(payload):
        errors.append("EXP-289 confirmatory prep digest mismatch")
    return errors


def build_exp289_confirmatory_prep(
    *,
    experiment: dict[str, Any],
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    analysis_code_digest: str,
    familywise_alpha: float,
) -> dict[str, Any]:
    if not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")
    _validate_frozen_experiment(experiment, familywise_alpha=familywise_alpha)
    rows, training_ids, replicate_ids = _development_rows(execution_artifact, arm_registry)
    effects = _paired_relative_rder_effects(rows)
    paired_sd = stdev(effects) if len(effects) > 1 else 0.0
    required = _required_n(paired_sd=paired_sd, familywise_alpha=familywise_alpha)
    confirmatory_n = max(MIN_N, required) if required <= MAX_N else None
    ready = confirmatory_n is not None
    status = "CONFIRMATORY_GATE_A_PREPARED" if ready else "NOT_READY_VARIANCE_EXCEEDS_MAX_N"

    reserved: list[int] = []
    if ready:
        lineage_max = max(training_ids + replicate_ids)
        reserved = list(range(lineage_max + 1, lineage_max + 1 + int(confirmatory_n)))

    no_rates = [float(row["no_nogood"][PRIMARY_ENDPOINT]) for row in rows]
    local_rates = [float(row["local_nogood"][PRIMARY_ENDPOINT]) for row in rows]
    local_overprune = [float(row["local_nogood"]["valid_state_overprune_rate"]) for row in rows]
    no_solution = [float(row["no_nogood"]["verified_solution_rate"]) for row in rows]
    local_solution = [float(row["local_nogood"]["verified_solution_rate"]) for row in rows]

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
            "effect_type": "paired_relative_rder_reduction",
            "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
            "familywise_alpha": float(familywise_alpha),
            "multiplicity_family": MULTIPLICITY_FAMILY,
            "analysis_method": ANALYSIS_METHOD,
            "zero_opportunity_policy": ZERO_OPPORTUNITY_POLICY,
            "epsilon_denominator_rescue": False,
            "protected_overprune_ceiling": PROTECTED_OVERPRUNE_CEILING,
            "protected_solution_floor": PROTECTED_SOLUTION_FLOOR,
        },
        "pilot_summary": {
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
            "n": len(rows),
            "eligible_paired_rder_replicates": len(effects),
            "training_replicate_ids": training_ids,
            "replicate_ids": replicate_ids,
            "mean_no_nogood_repeat_dead_end_rate": mean(no_rates),
            "mean_local_repeat_dead_end_rate": mean(local_rates),
            "paired_mean_relative_rder_reduction": mean(effects),
            "paired_relative_rder_reduction_sd": float(paired_sd),
            "mean_local_valid_state_overprune_rate": mean(local_overprune),
            "mean_no_nogood_verified_solution_rate": mean(no_solution),
            "mean_local_verified_solution_rate": mean(local_solution),
        },
        "sample_size_freeze": {
            "method": SAMPLE_SIZE_METHOD,
            "power_target": POWER_TARGET,
            "familywise_alpha": float(familywise_alpha),
            "min_n": MIN_N,
            "max_n": MAX_N,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "unclamped_required_n": required,
            "confirmatory_n": confirmatory_n,
            "planning_constants": {
                "target_relative_reduction": MESI_RELATIVE_REDUCTION,
                "z_alpha": NormalDist().inv_cdf(1.0 - familywise_alpha),
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
    errors = validate_exp289_confirmatory_prep(payload)
    if errors:
        raise RuntimeError("invalid EXP-289 confirmatory prep: " + "; ".join(errors))
    return payload
