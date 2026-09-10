from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp279_paired_runner import (
    MULTIPLICITY_FAMILY,
    PRIMARY_METRIC,
    PROTECTED_FLOOR,
    validate_exp279_paired_development,
)
from .exp279_routing_worlds import STRATA

SCHEMA = "NLM-EXP-279-CONFIRMATORY-PREP-V1"
EXPERIMENT_ID = "EXP-279"
HYPOTHESIS_ID = "H-CBRF-03"
MESI = 0.08
POWER = 0.90
MIN_N = 32
MAX_N = 128
FAMILYWISE_ALPHA = 0.05
PLANNING_ALPHA = 0.025
HOLM_THRESHOLDS = (0.025, 0.05)
BOOTSTRAP_SAMPLES = 10_000
ANALYSIS_METHOD = (
    "blocked paired contrasts; Holm-adjusted familywise comparisons against best simpler arm"
)
EFFECT_TYPE = "equal_weight_blocked_ratio_of_means_relative_gain"
SAMPLE_SIZE_METHOD = "max-two-contrast-blocked-within-stratum-normal-approximation-v1"
PRIMARY_CONTRASTS = (
    "hybrid_vs_propagation_only",
    "hybrid_vs_branch_only",
)
BEST_SIMPLE_SELECTION_RULE = (
    "higher equal-weight blocked mean primary utility; exact tie -> propagation_only"
)
DENOMINATOR_POLICY = (
    "no_epsilon; every simpler-arm blocked mean denominator must be finite and positive"
)


def _prep_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def _close(left: Any, right: float, *, tol: float = 1e-12) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=tol)
    )


def _required_n(*, blocked_sd: float) -> int:
    if not math.isfinite(blocked_sd) or blocked_sd < 0.0:
        raise ValueError("blocked_sd must be finite and non-negative")
    if blocked_sd == 0.0:
        return 1
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - PLANNING_ALPHA)
    z_power = normal.inv_cdf(POWER)
    raw = ((z_alpha + z_power) * blocked_sd / MESI) ** 2
    return max(1, int(math.ceil(raw)))


def _validate_frozen_experiment(experiment: dict[str, Any]) -> None:
    if experiment.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("confirmatory prep requires frozen EXP-279")
    if experiment.get("hypothesis_id") != HYPOTHESIS_ID:
        raise ValueError("EXP-279 hypothesis drift")
    arm_ids = [item.get("id") for item in experiment.get("arms") or []]
    if arm_ids != ["propagation_only", "branch_only", "hybrid"]:
        raise ValueError("EXP-279 arm ordering drift")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != PRIMARY_METRIC or primary.get("direction") != "higher":
        raise ValueError("EXP-279 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "relative_gain" or not _close(mesi.get("value"), MESI):
        raise ValueError("EXP-279 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if (
        not _close(sample.get("power_target"), POWER)
        or int(sample.get("min_n", -1)) != MIN_N
        or int(sample.get("max_n", -1)) != MAX_N
        or sample.get("paired") is not True
        or sample.get("freeze_rule") != "pilot blocked variance by structure-fit stratum"
    ):
        raise ValueError("EXP-279 sample-size contract drift")
    if experiment.get("analysis_method") != ANALYSIS_METHOD:
        raise ValueError("EXP-279 analysis method drift")
    if experiment.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        raise ValueError("EXP-279 multiplicity-family drift")
    protected = {
        item.get("metric"): item.get("floor")
        for item in (experiment.get("protected_endpoints") or [])
    }
    if protected.get("verified_solution_rate") != PROTECTED_FLOOR:
        raise ValueError("EXP-279 protected solution-rate floor drift")
    resource = experiment.get("resource_match") or {}
    if resource != {
        "parameter_budget": "reclaimed parameters assigned to simpler rivals",
        "inference_budget": "equal max accounted FLOPs",
        "structure_fit": "predeclared strata",
    }:
        raise ValueError("EXP-279 resource-match contract drift")
    challenge = experiment.get("challenge_generator") or {}
    if challenge.get("lane") != "POST_FREEZE_CHALLENGE":
        raise ValueError("EXP-279 challenge lane drift")
    if challenge.get("seed_rule") != "future beacon after freeze":
        raise ValueError("EXP-279 challenge seed rule drift")
    decision = experiment.get("decision_rule") or {}
    if decision.get("promote_if") != (
        "hybrid exceeds best simpler arm by >=8% with noninferior solution rate"
    ):
        raise ValueError("EXP-279 promotion rule drift")
    if decision.get("kill_if") != (
        "branch-only or propagation-only is practically equivalent or superior"
    ):
        raise ValueError("EXP-279 kill rule drift")


def _validate_registry_binding(
    arm_registry: dict[str, Any],
    execution_artifact: dict[str, Any],
) -> None:
    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-279 confirmatory prep requires neural arm registry")
    if arm_registry.get("evidence_level") != "EV-E2" or arm_registry.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-279 arm registry must remain EV-E2 / UNVERIFIED")
    if arm_registry.get("protocol_digest") != execution_artifact.get("protocol_digest"):
        raise ValueError("EXP-279 registry/execution protocol digest mismatch")
    if not isinstance(arm_registry.get("registry_digest"), str) or not arm_registry.get("registry_digest"):
        raise ValueError("EXP-279 arm registry digest missing")
    item = (arm_registry.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    if item.get("development_match_status") != "PAIRED_ROUTING_DEV_READY":
        raise ValueError("EXP-279 registry development match is not ready")
    if item.get("match_court") != "BLOCKED":
        raise ValueError("EXP-279 DEVELOPMENT registry must keep confirmatory court blocked")
    evidence = item.get("paired_execution_evidence") or {}
    if evidence.get("artifact_digest") != execution_artifact.get("artifact_digest"):
        raise ValueError("EXP-279 registry execution artifact digest mismatch")
    if evidence.get("code_digest") != execution_artifact.get("code_digest"):
        raise ValueError("EXP-279 registry execution code digest mismatch")


def _development_rows(
    execution_artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    execution_errors = validate_exp279_paired_development(execution_artifact)
    if execution_errors:
        raise ValueError("invalid EXP-279 development artifact: " + "; ".join(execution_errors))
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-279 confirmatory prep requires EV-E2 / UNVERIFIED development evidence")
    for key in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if execution_artifact.get(key) is not False:
            raise ValueError(f"EXP-279 development artifact must keep {key}=false")
    if execution_artifact.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        raise ValueError("invalid EXP-279 development artifact: multiplicity family drift")
    if execution_artifact.get("primary_endpoint") != {
        "metric": PRIMARY_METRIC,
        "direction": "higher",
        "mesi_relative_gain": MESI,
    }:
        raise ValueError("invalid EXP-279 development artifact: primary endpoint drift")
    if (execution_artifact.get("protected_endpoints") or {}).get(
        "verified_solution_rate_floor"
    ) != PROTECTED_FLOOR:
        raise ValueError("invalid EXP-279 development artifact: protected floor drift")
    if execution_artifact.get("information_receipt") != {
        "artifact": "constraint_variable_incidence",
        "ground_truth": True,
        "delivered_to": ["propagation_only", "hybrid"],
        "withheld_from": ["branch_only"],
        "branch_only_received_incidence": False,
    }:
        raise ValueError("invalid EXP-279 development artifact: information separation drift")

    training = execution_artifact.get("training") or {}
    if training.get("rng_stream") != "augmentation":
        raise ValueError("EXP-279 DEVELOPMENT training stream drift")
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    if train_start < 0 or train_count <= 0:
        raise ValueError("EXP-279 DEVELOPMENT training lineage invalid")
    training_ids = list(range(train_start, train_start + train_count))

    evaluation = execution_artifact.get("evaluation") or {}
    if evaluation.get("rng_stream") != "evaluation":
        raise ValueError("EXP-279 DEVELOPMENT pilot must use evaluation RNG")
    rows = list(evaluation.get("per_replicate") or [])
    if len(rows) < MIN_N:
        raise ValueError("EXP-279 confirmatory prep requires at least 32 DEVELOPMENT pilot replicates")
    if int(evaluation.get("replicates", 0) or 0) != len(rows):
        raise ValueError("EXP-279 pilot replicate count mismatch")
    replicate_ids = [row.get("replicate") for row in rows]
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in replicate_ids
    ):
        raise ValueError("EXP-279 pilot replicate IDs must be non-negative integers")
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("EXP-279 pilot replicate IDs must be unique")
    if set(training_ids).intersection(replicate_ids):
        raise ValueError("EXP-279 training and pilot replicate lineages overlap")
    observed_strata = {str(row.get("stratum")) for row in rows}
    if observed_strata != set(STRATA):
        raise ValueError("EXP-279 pilot must cover every predeclared structure-fit stratum")

    for row in rows:
        if row.get("world_pairing_closed") is not True or not row.get("paired_batch_digest"):
            raise ValueError("EXP-279 paired pilot world lineage is incomplete")
        if row.get("stratum") not in STRATA:
            raise ValueError("EXP-279 pilot stratum drift")
        for arm in ("propagation_only", "branch_only", "hybrid"):
            metrics = row.get(arm) or {}
            utility = metrics.get(PRIMARY_METRIC)
            solution = metrics.get("verified_solution_rate")
            flops = metrics.get("accounted_flops_per_episode")
            if (
                not isinstance(utility, (int, float))
                or isinstance(utility, bool)
                or not math.isfinite(float(utility))
                or float(utility) < 0.0
            ):
                raise ValueError("EXP-279 pilot primary utility invalid")
            if (
                not isinstance(solution, (int, float))
                or isinstance(solution, bool)
                or not math.isfinite(float(solution))
                or not 0.0 <= float(solution) <= 1.0
            ):
                raise ValueError("EXP-279 pilot verified solution rate invalid")
            if (
                not isinstance(flops, (int, float))
                or isinstance(flops, bool)
                or not math.isfinite(float(flops))
                or float(flops) <= 0.0
            ):
                raise ValueError("EXP-279 pilot accounted FLOPs invalid")
    return rows, training_ids, [int(item) for item in replicate_ids]


def _blocked_relative_effect_summary(
    rows: list[dict[str, Any]],
    *,
    simple_arm: str,
) -> dict[str, Any]:
    if simple_arm not in ("propagation_only", "branch_only"):
        raise ValueError("EXP-279 blocked contrast requires a simpler arm")
    by_stratum: dict[str, list[dict[str, Any]]] = {
        stratum: [row for row in rows if row.get("stratum") == stratum]
        for stratum in STRATA
    }
    if any(not selected for selected in by_stratum.values()):
        raise ValueError("EXP-279 blocked contrast requires every predeclared stratum")

    simple_stratum_means: dict[str, float] = {}
    hybrid_stratum_means: dict[str, float] = {}
    for stratum, selected in by_stratum.items():
        simple_values = [float(row[simple_arm][PRIMARY_METRIC]) for row in selected]
        hybrid_values = [float(row["hybrid"][PRIMARY_METRIC]) for row in selected]
        simple_stratum_means[stratum] = sum(simple_values) / len(simple_values)
        hybrid_stratum_means[stratum] = sum(hybrid_values) / len(hybrid_values)

    simple_blocked_mean = sum(simple_stratum_means.values()) / len(STRATA)
    hybrid_blocked_mean = sum(hybrid_stratum_means.values()) / len(STRATA)
    denominator_valid = (
        math.isfinite(simple_blocked_mean)
        and math.isfinite(hybrid_blocked_mean)
        and simple_blocked_mean > 0.0
    )

    effects_by_stratum: dict[str, list[float]] = {}
    residuals: list[float] = []
    all_effects: list[float] = []
    stratum_relative_gains: dict[str, float | None] = {}
    if denominator_valid:
        for stratum, selected in by_stratum.items():
            effects = [
                (
                    float(row["hybrid"][PRIMARY_METRIC])
                    - float(row[simple_arm][PRIMARY_METRIC])
                )
                / simple_blocked_mean
                for row in selected
            ]
            effects_by_stratum[stratum] = effects
            all_effects.extend(effects)
            mean_effect = sum(effects) / len(effects)
            residuals.extend(value - mean_effect for value in effects)
            denominator = simple_stratum_means[stratum]
            stratum_relative_gains[stratum] = (
                hybrid_stratum_means[stratum] / denominator - 1.0
                if denominator > 0.0 and math.isfinite(denominator)
                else None
            )
        degrees_of_freedom = len(all_effects) - len(STRATA)
        if degrees_of_freedom <= 0:
            blocked_sd = 0.0
        else:
            blocked_sd = math.sqrt(
                sum(value * value for value in residuals) / degrees_of_freedom
            )
        observed_gain: float | None = hybrid_blocked_mean / simple_blocked_mean - 1.0
    else:
        for stratum in STRATA:
            effects_by_stratum[stratum] = []
            stratum_relative_gains[stratum] = None
        blocked_sd = None
        observed_gain = None

    return {
        "simple_arm": simple_arm,
        "effect_type": EFFECT_TYPE,
        "stratum_counts": {
            stratum: len(by_stratum[stratum])
            for stratum in STRATA
        },
        "simple_stratum_mean_utility": simple_stratum_means,
        "hybrid_stratum_mean_utility": hybrid_stratum_means,
        "stratum_relative_gains": stratum_relative_gains,
        "simple_blocked_mean_utility": simple_blocked_mean,
        "hybrid_blocked_mean_utility": hybrid_blocked_mean,
        "observed_blocked_relative_gain": observed_gain,
        "denominator_valid": denominator_valid,
        "scientific_denominator_policy": DENOMINATOR_POLICY,
        "blocked_within_stratum_sd": blocked_sd,
        "blocked_effect_digest": canonical_sha256(
            {
                "simple_arm": simple_arm,
                "strata": list(STRATA),
                "effects_by_stratum": effects_by_stratum,
            }
        ),
    }


def _frozen_analysis() -> dict[str, Any]:
    return {
        "primary_endpoint": PRIMARY_METRIC,
        "primary_direction": "higher",
        "effect_type": EFFECT_TYPE,
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "mesi_relative_gain": MESI,
        "power_target": POWER,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "planning_alpha": PLANNING_ALPHA,
        "holm_step_down_thresholds": list(HOLM_THRESHOLDS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "analysis_method": ANALYSIS_METHOD,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "structure_fit_strata": list(STRATA),
        "block_weighting": "equal_weight_across_predeclared_structure_fit_strata",
        "best_simple_selection_rule": BEST_SIMPLE_SELECTION_RULE,
        "protected_solution_floor": PROTECTED_FLOOR,
        "scientific_denominator_policy": DENOMINATOR_POLICY,
        "scientific_best_simple_policy": (
            "compute and Holm-control both hybrid-vs-simple contrasts before applying the frozen best-simple rule"
        ),
    }


def build_exp279_confirmatory_prep(
    *,
    experiment: dict[str, Any],
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    _validate_frozen_experiment(experiment)
    if not isinstance(analysis_code_digest, str) or not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")
    rows, training_ids, replicate_ids = _development_rows(execution_artifact)
    _validate_registry_binding(arm_registry, execution_artifact)

    summaries = {
        "hybrid_vs_propagation_only": _blocked_relative_effect_summary(
            rows, simple_arm="propagation_only"
        ),
        "hybrid_vs_branch_only": _blocked_relative_effect_summary(
            rows, simple_arm="branch_only"
        ),
    }
    denominators_valid = all(
        summary["denominator_valid"] is True
        for summary in summaries.values()
    )
    contrast_required_n: dict[str, int | None]
    required: int | None
    confirmatory_n: int | None
    reserved: list[int]
    blockers: list[str]

    if not denominators_valid:
        status = "NOT_READY_SIMPLE_BLOCKED_MEAN_NONPOSITIVE"
        contrast_required_n = {name: None for name in PRIMARY_CONTRASTS}
        required = None
        confirmatory_n = None
        reserved = []
        blockers = [
            "at least one simpler arm has nonpositive or nonfinite equal-weight blocked mean primary utility; no epsilon rescue is allowed"
        ]
    else:
        contrast_required_n = {
            name: _required_n(blocked_sd=float(summary["blocked_within_stratum_sd"]))
            for name, summary in summaries.items()
        }
        required = max(int(value) for value in contrast_required_n.values())
        if required > MAX_N:
            status = "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
            confirmatory_n = None
            reserved = []
            blockers = [
                f"worst blocked pilot contrast requires n above frozen maximum {MAX_N}; challenge remains unmaterialized"
            ]
        else:
            status = "CONFIRMATORY_GATE_A_PREPARED"
            confirmatory_n = max(MIN_N, required)
            start = max(training_ids + replicate_ids) + 1
            reserved = list(range(start, start + confirmatory_n))
            blockers = [
                "future public beacon has not been consumed",
                "post-freeze EXP-279 confirmatory challenge execution remains unrun",
            ]

    normal = NormalDist()
    pair_audit = (execution_artifact.get("resource_match") or {}).get("pair_audit") or {}
    route_config = execution_artifact.get("route_config") or {}
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": status,
        "confirmatory_ready": status == "CONFIRMATORY_GATE_A_PREPARED",
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "frozen_analysis": _frozen_analysis(),
        "pilot_summary": {
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
            "n": len(rows),
            "training_replicate_ids": training_ids,
            "replicate_ids": replicate_ids,
            "stratum_counts": {
                stratum: sum(row.get("stratum") == stratum for row in rows)
                for stratum in STRATA
            },
            "contrasts": summaries,
            "pilot_best_simple_not_carried_into_confirmatory": True,
            "development_descriptive_best_simple_arm": (
                (execution_artifact.get("evaluation") or {}).get("aggregate") or {}
            ).get("best_simple_arm"),
        },
        "sample_size_freeze": {
            "method": SAMPLE_SIZE_METHOD,
            "power_target": POWER,
            "min_n": MIN_N,
            "max_n": MAX_N,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "contrast_required_n": contrast_required_n,
            "unclamped_required_n": required,
            "confirmatory_n": confirmatory_n,
            "planning_constants": {
                "mesi_relative_gain": MESI,
                "power_target": POWER,
                "familywise_alpha": FAMILYWISE_ALPHA,
                "planning_alpha": PLANNING_ALPHA,
                "holm_family_size": 2,
                "z_alpha": normal.inv_cdf(1.0 - PLANNING_ALPHA),
                "z_power": normal.inv_cdf(POWER),
            },
        },
        "confirmatory_lineage": {
            "lane": "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "seed_materialization_status": "NOT_EXECUTED",
            "reserved_replicate_ids": reserved,
            "stratum_schedule": list(STRATA),
            "stratum_schedule_rule": (
                "cycle PROPAGATION_FIT, BRANCH_FIT, MIXED_RESIDUAL in reserved replicate order"
            ),
        },
        "lineage": {
            "protocol_digest": execution_artifact.get("protocol_digest"),
            "development_execution_digest": execution_artifact.get("artifact_digest"),
            "development_code_digest": execution_artifact.get("code_digest"),
            "arm_registry_digest": arm_registry.get("registry_digest"),
            "pair_audit_digest": canonical_sha256(pair_audit),
            "analysis_code_digest": analysis_code_digest,
            "route_threshold": route_config.get("threshold"),
        },
        "development_execution_artifact": deepcopy(execution_artifact),
        "remaining_blockers": blockers,
        "prep_digest": "",
    }
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp279_confirmatory_prep(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 confirmatory prep: " + "; ".join(errors))
    return payload


def validate_exp279_confirmatory_prep(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 confirmatory prep identity")
    if payload.get("hypothesis_id") != HYPOTHESIS_ID:
        errors.append("EXP-279 confirmatory prep hypothesis drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 confirmatory prep cannot promote evidence")
    for key in ("confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(key) is not False:
            errors.append(f"EXP-279 confirmatory prep forbidden flag enabled: {key}")
    status = payload.get("status")
    if payload.get("confirmatory_ready") is not (status == "CONFIRMATORY_GATE_A_PREPARED"):
        errors.append("EXP-279 confirmatory-ready/status mismatch")
    frozen = payload.get("frozen_analysis") or {}
    expected_frozen = _frozen_analysis()
    for key, expected in expected_frozen.items():
        observed = frozen.get(key)
        if isinstance(expected, float):
            if not _close(observed, expected):
                errors.append(f"EXP-279 frozen analysis {key} drift")
        elif observed != expected:
            errors.append(f"EXP-279 frozen analysis {key} drift")

    execution = payload.get("development_execution_artifact")
    if not isinstance(execution, dict):
        errors.append("EXP-279 development execution evidence missing")
        return errors
    try:
        rows, training_ids, replicate_ids = _development_rows(execution)
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    pilot = payload.get("pilot_summary") or {}
    if pilot.get("source_lane") != "DEVELOPMENT_PILOT_ONLY":
        errors.append("EXP-279 pilot source lane drift")
    if pilot.get("n") != len(rows):
        errors.append("EXP-279 pilot n mismatch")
    if pilot.get("training_replicate_ids") != training_ids or pilot.get("replicate_ids") != replicate_ids:
        errors.append("EXP-279 pilot replicate lineage mismatch")
    expected_counts = {
        stratum: sum(row.get("stratum") == stratum for row in rows)
        for stratum in STRATA
    }
    if pilot.get("stratum_counts") != expected_counts:
        errors.append("EXP-279 pilot stratum-count drift")
    if pilot.get("pilot_best_simple_not_carried_into_confirmatory") is not True:
        errors.append("EXP-279 prep must forbid carrying pilot best-simple selection into confirmatory")

    expected_summaries = {
        "hybrid_vs_propagation_only": _blocked_relative_effect_summary(
            rows, simple_arm="propagation_only"
        ),
        "hybrid_vs_branch_only": _blocked_relative_effect_summary(
            rows, simple_arm="branch_only"
        ),
    }
    if pilot.get("contrasts") != expected_summaries:
        errors.append("EXP-279 blocked pilot contrast summary mismatch")

    denominators_valid = all(
        summary["denominator_valid"] is True
        for summary in expected_summaries.values()
    )
    freeze = payload.get("sample_size_freeze") or {}
    if freeze.get("method") != SAMPLE_SIZE_METHOD:
        errors.append("EXP-279 sample-size method drift")
    if not _close(freeze.get("power_target"), POWER):
        errors.append("EXP-279 sample-size power drift")
    if int(freeze.get("min_n", -1)) != MIN_N or int(freeze.get("max_n", -1)) != MAX_N:
        errors.append("EXP-279 sample-size bounds drift")
    if freeze.get("paired") is not True or freeze.get("pilot_reuse_as_confirmatory") is not False:
        errors.append("EXP-279 sample-size pairing/reuse contract drift")
    constants = freeze.get("planning_constants") or {}
    normal = NormalDist()
    if not _close(constants.get("mesi_relative_gain"), MESI):
        errors.append("EXP-279 planning MESI drift")
    if not _close(constants.get("planning_alpha"), PLANNING_ALPHA):
        errors.append("EXP-279 planning alpha drift")
    if not _close(constants.get("familywise_alpha"), FAMILYWISE_ALPHA):
        errors.append("EXP-279 planning familywise alpha drift")
    if int(constants.get("holm_family_size", -1)) != 2:
        errors.append("EXP-279 Holm family-size drift")
    if not _close(constants.get("z_alpha"), normal.inv_cdf(1.0 - PLANNING_ALPHA)):
        errors.append("EXP-279 sample-size z_alpha drift")
    if not _close(constants.get("z_power"), normal.inv_cdf(POWER)):
        errors.append("EXP-279 sample-size z_power drift")

    expected_required: dict[str, int | None]
    expected_max: int | None
    if denominators_valid:
        expected_required = {
            name: _required_n(blocked_sd=float(summary["blocked_within_stratum_sd"]))
            for name, summary in expected_summaries.items()
        }
        expected_max = max(int(value) for value in expected_required.values())
    else:
        expected_required = {name: None for name in PRIMARY_CONTRASTS}
        expected_max = None
    if freeze.get("contrast_required_n") != expected_required:
        errors.append("EXP-279 per-contrast required n mismatch")
    if freeze.get("unclamped_required_n") != expected_max:
        errors.append("EXP-279 worst-contrast required n mismatch")

    confirmatory_n = freeze.get("confirmatory_n")
    lineage = payload.get("confirmatory_lineage") or {}
    reserved = list(lineage.get("reserved_replicate_ids") or [])
    if lineage.get("lane") != "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED":
        errors.append("EXP-279 confirmatory lane drift")
    if lineage.get("pilot_reuse_forbidden") is not True:
        errors.append("EXP-279 confirmatory lineage must forbid pilot reuse")
    if lineage.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-279 prep cannot materialize confirmatory seeds")
    if lineage.get("stratum_schedule") != list(STRATA):
        errors.append("EXP-279 confirmatory stratum schedule drift")
    if set(reserved).intersection(training_ids) or set(reserved).intersection(replicate_ids):
        errors.append("EXP-279 reserved confirmatory IDs overlap DEVELOPMENT lineage")

    if not denominators_valid:
        if status != "NOT_READY_SIMPLE_BLOCKED_MEAN_NONPOSITIVE":
            errors.append("EXP-279 invalid denominator must block Gate A")
        if confirmatory_n is not None or reserved:
            errors.append("EXP-279 invalid denominator cannot reserve confirmatory data")
    elif expected_max is not None and expected_max > MAX_N:
        if status != "NOT_READY_VARIANCE_EXCEEDS_MAX_N":
            errors.append("EXP-279 variance above max must block Gate A")
        if confirmatory_n is not None or reserved:
            errors.append("EXP-279 variance-above-max prep cannot reserve confirmatory data")
    else:
        if status != "CONFIRMATORY_GATE_A_PREPARED":
            errors.append("EXP-279 valid blocked pilot must prepare Gate A")
        expected_n = max(MIN_N, int(expected_max or 1))
        if confirmatory_n != expected_n:
            errors.append("EXP-279 confirmatory n mismatch")
        if len(reserved) != expected_n:
            errors.append("EXP-279 reserved replicate count mismatch")
        expected_start = max(training_ids + replicate_ids) + 1
        if reserved != list(range(expected_start, expected_start + expected_n)):
            errors.append("EXP-279 reserved replicate IDs are not contiguous after DEVELOPMENT lineage")

    lineage_binding = payload.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": execution.get("protocol_digest"),
        "development_execution_digest": execution.get("artifact_digest"),
        "development_code_digest": execution.get("code_digest"),
        "pair_audit_digest": canonical_sha256(
            (execution.get("resource_match") or {}).get("pair_audit") or {}
        ),
        "route_threshold": (execution.get("route_config") or {}).get("threshold"),
    }
    for key, expected in expected_lineage.items():
        if lineage_binding.get(key) != expected:
            errors.append(f"EXP-279 confirmatory prep lineage {key} mismatch")
    if not lineage_binding.get("arm_registry_digest") or not lineage_binding.get("analysis_code_digest"):
        errors.append("EXP-279 registry/analysis lineage missing")

    digest = payload.get("prep_digest")
    if not isinstance(digest, str) or digest != _prep_digest(payload):
        errors.append("EXP-279 confirmatory prep digest mismatch")
    return errors
