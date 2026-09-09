from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist, stdev
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-277-CONFIRMATORY-PREP-V1"
EXPECTED_EXPERIMENT_ID = "EXP-277"
EXPECTED_PRIMARY_ENDPOINT = "verified_utility_per_accounted_flop"
EXPECTED_MESI = 0.10
EXPECTED_POWER = 0.90
EXPECTED_MIN_N = 32
EXPECTED_MAX_N = 128
DEFAULT_FAMILYWISE_ALPHA = 0.05
ENDPOINT_ALPHA = 0.025
BOOTSTRAP_SAMPLES = 10_000
SAMPLE_SIZE_METHOD = "paired-normal-approximation-from-normalized-utility-sd-v1"
EXPECTED_ANALYSIS_METHOD = (
    "paired effect with bootstrap confidence interval; report mean, median, sign consistency and divergence rate"
)
EXPECTED_MULTIPLICITY_FAMILY = "CBRF_HEADROOM"
EXPECTED_SOLUTION_FLOOR = "oracle_cbrf >= arcs_branch - 0.005"
EXPECTED_ENERGY_FLOOR = "report-only at Stage-A; cannot be hidden from cost ledger"


def _prep_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def _required_n(*, paired_sd: float, mesi: float, power: float, alpha: float) -> int:
    if not math.isfinite(paired_sd) or paired_sd < 0.0:
        raise ValueError("paired_sd must be finite and non-negative")
    if paired_sd == 0.0:
        return 1
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(power)
    raw_n = ((z_alpha + z_power) * paired_sd / mesi) ** 2
    return max(1, int(math.ceil(raw_n)))


def _validate_frozen_experiment(experiment: dict[str, Any], *, familywise_alpha: float) -> None:
    if experiment.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("confirmatory prep requires EXP-277")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != EXPECTED_PRIMARY_ENDPOINT or primary.get("direction") != "higher":
        raise ValueError("EXP-277 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "relative_gain" or float(mesi.get("value", -1.0)) != EXPECTED_MESI:
        raise ValueError("EXP-277 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if float(sample.get("power_target", -1.0)) != EXPECTED_POWER:
        raise ValueError("EXP-277 power target drift")
    if int(sample.get("min_n", -1)) != EXPECTED_MIN_N or int(sample.get("max_n", -1)) != EXPECTED_MAX_N:
        raise ValueError("EXP-277 sample-size bounds drift")
    if sample.get("paired") is not True:
        raise ValueError("EXP-277 paired-analysis contract drift")
    if experiment.get("analysis_method") != EXPECTED_ANALYSIS_METHOD:
        raise ValueError("EXP-277 analysis method drift")
    if experiment.get("multiplicity_family") != EXPECTED_MULTIPLICITY_FAMILY:
        raise ValueError("EXP-277 multiplicity family drift")
    protected = {item.get("metric"): item.get("floor") for item in experiment.get("protected_endpoints") or []}
    if protected.get("verified_solution_rate") != EXPECTED_SOLUTION_FLOOR:
        raise ValueError("EXP-277 protected solution-rate floor drift")
    if protected.get("wall_energy_per_episode") != EXPECTED_ENERGY_FLOOR:
        raise ValueError("EXP-277 wall-energy reporting contract drift")
    challenge = experiment.get("challenge_generator") or {}
    if challenge.get("lane") != "POST_FREEZE_CHALLENGE":
        raise ValueError("EXP-277 challenge lane drift")
    if float(familywise_alpha) != DEFAULT_FAMILYWISE_ALPHA:
        raise ValueError("EXP-277 alpha drift")


def _development_rows(
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    if execution_artifact.get("schema") != "NLM-EXP-277-PAIRED-DEV-EVAL-V1":
        raise ValueError("EXP-277 paired development artifact schema mismatch")
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("confirmatory prep requires EV-E2 / UNVERIFIED development evidence")
    if execution_artifact.get("confirmatory_ready") is not False:
        raise ValueError("EXP-277 development artifact cannot already be confirmatory-ready")
    if execution_artifact.get("confirmatory_data_consumed") is not False:
        raise ValueError("EXP-277 development artifact cannot consume confirmatory data")
    if execution_artifact.get("challenge_materialized") is not False:
        raise ValueError("EXP-277 development artifact cannot materialize challenge data")
    if execution_artifact.get("decision_rule_executed") is not False:
        raise ValueError("EXP-277 development artifact cannot execute the scientific decision rule")

    resource = execution_artifact.get("resource_match") or {}
    for field in ("parameter_match", "functional_parameter_match", "same_world_lineage", "compute_budget_closed"):
        if resource.get(field) is not True:
            raise ValueError(f"EXP-277 development resource court is open: {field}")
    oracle_receipt = execution_artifact.get("oracle_information_receipt") or {}
    if oracle_receipt != {
        "artifact": "oracle_incidence",
        "ground_truth": True,
        "delivered_to": ["oracle_cbrf"],
        "withheld_from": ["arcs_branch"],
        "arcs_received_oracle_incidence": False,
    }:
        raise ValueError("EXP-277 oracle-information separation drift")

    training = execution_artifact.get("training") or {}
    if training.get("rng_stream") != "augmentation":
        raise ValueError("EXP-277 DEVELOPMENT training must use augmentation RNG")
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = list(training.get("paired_batch_digests") or [])
    if train_start != 0 or train_count <= 0 or len(train_digests) != train_count:
        raise ValueError("EXP-277 DEVELOPMENT training lineage is incomplete")
    training_ids = list(range(train_start, train_start + train_count))

    evaluation = execution_artifact.get("evaluation") or {}
    if evaluation.get("rng_stream") != "evaluation":
        raise ValueError("EXP-277 pilot must use evaluation RNG")
    rows = list(evaluation.get("per_replicate") or [])
    if len(rows) < EXPECTED_MIN_N:
        raise ValueError("confirmatory prep requires at least 32 paired pilot replicates")
    if int(evaluation.get("replicates", 0) or 0) != len(rows):
        raise ValueError("EXP-277 paired pilot replicate count mismatch")
    replicate_ids = [row.get("replicate") for row in rows]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in replicate_ids):
        raise ValueError("EXP-277 pilot replicate IDs must be non-negative integers")
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("EXP-277 pilot replicate IDs must be unique")
    if set(training_ids).intersection(replicate_ids):
        raise ValueError("EXP-277 training and evaluation replicate lineages overlap")

    for row in rows:
        if row.get("world_pairing_closed") is not True or not row.get("paired_batch_digest"):
            raise ValueError("EXP-277 paired pilot world lineage is incomplete")
        for arm in ("arcs_branch", "oracle_cbrf"):
            metrics = row.get(arm) or {}
            utility = metrics.get(EXPECTED_PRIMARY_ENDPOINT)
            solution = metrics.get("verified_solution_rate")
            flops = metrics.get("accounted_flops_per_episode")
            if not isinstance(utility, (int, float)) or not math.isfinite(float(utility)):
                raise ValueError("EXP-277 pilot utility contains non-finite values")
            if not isinstance(solution, (int, float)) or not math.isfinite(float(solution)):
                raise ValueError("EXP-277 pilot solution-rate contains non-finite values")
            if not isinstance(flops, int) or isinstance(flops, bool) or flops <= 0:
                raise ValueError("EXP-277 pilot accounted FLOPs are invalid")

    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-277 confirmatory prep requires neural arm registry")
    if arm_registry.get("evidence_level") != "EV-E2" or arm_registry.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-277 arm registry must remain EV-E2 / UNVERIFIED")
    protocol_digest = execution_artifact.get("protocol_digest")
    if not protocol_digest or arm_registry.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-277 registry/execution protocol digest mismatch")
    registry_exp = (arm_registry.get("experiments") or {}).get("EXP-277") or {}
    if registry_exp.get("development_match_status") != "PAIRED_STRUCTURE_DENSE_DEV_READY":
        raise ValueError("EXP-277 registry development match is not ready")
    registry_execution = registry_exp.get("paired_execution_evidence") or {}
    if registry_execution.get("artifact_digest") != execution_artifact.get("artifact_digest"):
        raise ValueError("EXP-277 registry execution artifact digest mismatch")
    if registry_execution.get("code_digest") not in (None, execution_artifact.get("code_digest")):
        raise ValueError("EXP-277 registry execution code digest mismatch")

    return rows, training_ids, [int(item) for item in replicate_ids]


def _normalized_effects(rows: list[dict[str, Any]]) -> tuple[float, float, list[float]]:
    baseline = [float(row["arcs_branch"][EXPECTED_PRIMARY_ENDPOINT]) for row in rows]
    oracle = [float(row["oracle_cbrf"][EXPECTED_PRIMARY_ENDPOINT]) for row in rows]
    baseline_mean = sum(baseline) / len(baseline)
    oracle_mean = sum(oracle) / len(oracle)
    if not math.isfinite(baseline_mean) or not math.isfinite(oracle_mean):
        raise ValueError("EXP-277 pilot mean utility is non-finite")
    if baseline_mean <= 0.0:
        return baseline_mean, oracle_mean, []
    effects = [(o - a) / baseline_mean for a, o in zip(baseline, oracle, strict=True)]
    if any(not math.isfinite(value) for value in effects):
        raise ValueError("EXP-277 normalized pilot effect is non-finite")
    return baseline_mean, oracle_mean, effects


def validate_exp277_confirmatory_prep(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-277 confirmatory prep schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-277 confirmatory prep cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("confirmatory prep cannot promote H-CBRF-01")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("confirmatory prep cannot consume confirmatory data")
    if payload.get("challenge_materialized") is not False:
        errors.append("confirmatory prep cannot materialize challenge data")
    if payload.get("decision_rule_executed") is not False:
        errors.append("confirmatory prep cannot execute the scientific decision rule")

    frozen = payload.get("frozen_analysis") or {}
    if frozen.get("primary_endpoint") != EXPECTED_PRIMARY_ENDPOINT or frozen.get("primary_direction") != "higher":
        errors.append("EXP-277 primary endpoint drift")
    if frozen.get("effect_type") != "ratio_of_means_relative_gain":
        errors.append("EXP-277 primary effect contract drift")
    if float(frozen.get("mesi_relative_gain", -1.0)) != EXPECTED_MESI:
        errors.append("EXP-277 MESI drift")
    if float(frozen.get("familywise_alpha", -1.0)) != DEFAULT_FAMILYWISE_ALPHA:
        errors.append("EXP-277 familywise alpha drift")
    if float(frozen.get("endpoint_alpha", -1.0)) != ENDPOINT_ALPHA:
        errors.append("EXP-277 endpoint alpha drift")
    if int(frozen.get("bootstrap_samples", -1)) != BOOTSTRAP_SAMPLES:
        errors.append("EXP-277 bootstrap-count drift")
    if frozen.get("analysis_method") != EXPECTED_ANALYSIS_METHOD:
        errors.append("EXP-277 analysis method drift")
    if frozen.get("multiplicity_family") != EXPECTED_MULTIPLICITY_FAMILY:
        errors.append("EXP-277 multiplicity family drift")
    if frozen.get("protected_solution_floor") != EXPECTED_SOLUTION_FLOOR:
        errors.append("EXP-277 protected solution-rate floor drift")
    if frozen.get("scientific_denominator_policy") != "no_epsilon; nonpositive_or_nonfinite_baseline_is_not_promotable":
        errors.append("EXP-277 scientific denominator policy drift")

    pilot = payload.get("pilot_summary") or {}
    pilot_n = int(pilot.get("n", 0) or 0)
    if pilot_n < EXPECTED_MIN_N:
        errors.append("confirmatory prep requires at least 32 paired pilot replicates")
    if pilot.get("source_lane") != "DEVELOPMENT_PILOT_ONLY":
        errors.append("EXP-277 pilot source lane drift")
    training_ids = list(pilot.get("training_replicate_ids") or [])
    replicate_ids = list(pilot.get("replicate_ids") or [])
    if set(training_ids).intersection(replicate_ids):
        errors.append("EXP-277 training and pilot replicate lineages overlap")

    freeze = payload.get("sample_size_freeze") or {}
    if int(freeze.get("min_n", -1)) != EXPECTED_MIN_N or int(freeze.get("max_n", -1)) != EXPECTED_MAX_N:
        errors.append("EXP-277 frozen sample-size bounds drift")
    if freeze.get("paired") is not True:
        errors.append("EXP-277 paired sample-size contract drift")
    if freeze.get("pilot_reuse_as_confirmatory") is not False:
        errors.append("confirmatory prep must forbid pilot reuse as confirmatory")
    if freeze.get("method") != SAMPLE_SIZE_METHOD:
        errors.append("EXP-277 sample-size method drift")
    constants = freeze.get("planning_constants") or {}
    expected_z_alpha = NormalDist().inv_cdf(1.0 - ENDPOINT_ALPHA)
    expected_z_power = NormalDist().inv_cdf(EXPECTED_POWER)
    if not math.isclose(float(constants.get("z_alpha", math.nan)), expected_z_alpha, rel_tol=0.0, abs_tol=1e-12):
        errors.append("EXP-277 sample-size z_alpha drift")
    if not math.isclose(float(constants.get("z_power", math.nan)), expected_z_power, rel_tol=0.0, abs_tol=1e-12):
        errors.append("EXP-277 sample-size z_power drift")

    confirmatory = payload.get("confirmatory_lineage") or {}
    if confirmatory.get("pilot_reuse_forbidden") is not True:
        errors.append("EXP-277 confirmatory lineage must forbid pilot reuse")
    if confirmatory.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("confirmatory seed materialization must remain NOT_EXECUTED during prep")
    if confirmatory.get("lane") != "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED":
        errors.append("EXP-277 confirmatory lane drift")
    reserved = list(confirmatory.get("reserved_replicate_ids") or [])
    if set(reserved).intersection(training_ids) or set(reserved).intersection(replicate_ids):
        errors.append("EXP-277 reserved confirmatory IDs overlap DEVELOPMENT lineage")

    status = payload.get("status")
    baseline_mean = pilot.get("baseline_mean_utility")
    paired_sd = pilot.get("paired_sd")
    required = freeze.get("unclamped_required_n")
    confirmatory_n = freeze.get("confirmatory_n")

    if status == "NOT_READY_PRIMARY_BASELINE_NONPOSITIVE":
        if not isinstance(baseline_mean, (int, float)) or not math.isfinite(float(baseline_mean)) or float(baseline_mean) > 0.0:
            errors.append("baseline-nonpositive status requires finite baseline mean <=0")
        if paired_sd is not None or required is not None or confirmatory_n is not None or reserved:
            errors.append("baseline-nonpositive prep cannot freeze confirmatory sample size")
    else:
        if not isinstance(baseline_mean, (int, float)) or not math.isfinite(float(baseline_mean)) or float(baseline_mean) <= 0.0:
            errors.append("prepared EXP-277 prep requires positive finite baseline mean")
        if not isinstance(paired_sd, (int, float)) or not math.isfinite(float(paired_sd)) or float(paired_sd) < 0.0:
            errors.append("invalid EXP-277 paired pilot SD")
        else:
            expected_required = _required_n(
                paired_sd=float(paired_sd),
                mesi=EXPECTED_MESI,
                power=EXPECTED_POWER,
                alpha=ENDPOINT_ALPHA,
            )
            if required != expected_required:
                errors.append("unclamped required n does not match frozen power formula")

        if status == "CONFIRMATORY_GATE_A_PREPARED":
            if not isinstance(confirmatory_n, int) or not (EXPECTED_MIN_N <= confirmatory_n <= EXPECTED_MAX_N):
                errors.append("prepared confirmatory n must be inside frozen bounds")
            elif isinstance(required, int) and confirmatory_n != max(EXPECTED_MIN_N, required):
                errors.append("confirmatory n does not match frozen sample-size rule")
            if isinstance(confirmatory_n, int) and len(reserved) != confirmatory_n:
                errors.append("reserved confirmatory replicate count does not match confirmatory n")
            if reserved:
                all_dev = training_ids + replicate_ids
                expected_start = max(all_dev) + 1 if all_dev else 0
                if reserved != list(range(expected_start, expected_start + len(reserved))):
                    errors.append("EXP-277 reserved confirmatory IDs are not contiguous after DEVELOPMENT lineage")
        elif status == "NOT_READY_VARIANCE_EXCEEDS_MAX_N":
            if confirmatory_n is not None or reserved:
                errors.append("variance-exceeds-max prep cannot reserve confirmatory data")
            if not isinstance(required, int) or required <= EXPECTED_MAX_N:
                errors.append("variance-exceeds-max status requires n above frozen maximum")
        else:
            errors.append("invalid EXP-277 confirmatory prep status")

    lineage = payload.get("lineage") or {}
    for key in ("protocol_digest", "execution_artifact_digest", "arm_registry_digest", "analysis_code_digest", "development_code_digest", "arcs_final_state_digest", "oracle_final_state_digest"):
        if not lineage.get(key):
            errors.append(f"missing EXP-277 confirmatory prep lineage {key}")

    digest = payload.get("prep_digest")
    if not isinstance(digest, str) or digest != _prep_digest(payload):
        errors.append("EXP-277 confirmatory prep digest mismatch")
    return errors


def build_exp277_confirmatory_prep(
    *,
    experiment: dict[str, Any],
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    analysis_code_digest: str,
    familywise_alpha: float = DEFAULT_FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    _validate_frozen_experiment(experiment, familywise_alpha=familywise_alpha)
    if not isinstance(analysis_code_digest, str) or not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")

    rows, training_ids, replicate_ids = _development_rows(execution_artifact, arm_registry)
    baseline_mean, oracle_mean, effects = _normalized_effects(rows)

    if baseline_mean <= 0.0:
        status = "NOT_READY_PRIMARY_BASELINE_NONPOSITIVE"
        paired_sd: float | None = None
        required: int | None = None
        confirmatory_n: int | None = None
        reserved_ids: list[int] = []
        blockers = [
            "DEVELOPMENT baseline mean verified utility per accounted FLOP is nonpositive; relative confirmatory endpoint cannot be preregistered safely"
        ]
    else:
        paired_sd = stdev(effects) if len(effects) > 1 else 0.0
        required = _required_n(
            paired_sd=paired_sd,
            mesi=EXPECTED_MESI,
            power=EXPECTED_POWER,
            alpha=ENDPOINT_ALPHA,
        )
        if required > EXPECTED_MAX_N:
            status = "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
            confirmatory_n = None
            reserved_ids = []
            blockers = [
                f"pilot variance implies required n above frozen maximum {EXPECTED_MAX_N}; confirmatory challenge remains unmaterialized"
            ]
        else:
            status = "CONFIRMATORY_GATE_A_PREPARED"
            confirmatory_n = max(EXPECTED_MIN_N, required)
            start = max(training_ids + replicate_ids) + 1
            reserved_ids = list(range(start, start + confirmatory_n))
            blockers = [
                "future public beacon has not been consumed",
                "post-freeze confirmatory challenge execution remains unrun",
            ]

    effect_digest = canonical_sha256({"normalized_paired_effects": effects})
    normal = NormalDist()
    final_state = execution_artifact.get("final_state") or {}
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": status,
        "confirmatory_ready": status == "CONFIRMATORY_GATE_A_PREPARED",
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "frozen_analysis": {
            "primary_endpoint": EXPECTED_PRIMARY_ENDPOINT,
            "primary_direction": "higher",
            "effect_type": "ratio_of_means_relative_gain",
            "mesi_relative_gain": EXPECTED_MESI,
            "power_target": EXPECTED_POWER,
            "familywise_alpha": DEFAULT_FAMILYWISE_ALPHA,
            "endpoint_alpha": ENDPOINT_ALPHA,
            "primary_inference_tail": "one_sided_lower_bound",
            "kill_inference_tail": "one_sided_upper_bound",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "analysis_method": EXPECTED_ANALYSIS_METHOD,
            "multiplicity_family": EXPECTED_MULTIPLICITY_FAMILY,
            "protected_solution_floor": EXPECTED_SOLUTION_FLOOR,
            "wall_energy_contract": EXPECTED_ENERGY_FLOOR,
            "scientific_denominator_policy": "no_epsilon; nonpositive_or_nonfinite_baseline_is_not_promotable",
        },
        "pilot_summary": {
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
            "n": len(rows),
            "training_replicate_ids": training_ids,
            "replicate_ids": replicate_ids,
            "baseline_mean_utility": baseline_mean,
            "oracle_mean_utility": oracle_mean,
            "normalized_mean_effect": (sum(effects) / len(effects)) if effects else None,
            "paired_sd": paired_sd,
            "paired_effect_digest": effect_digest,
        },
        "sample_size_freeze": {
            "method": SAMPLE_SIZE_METHOD,
            "min_n": EXPECTED_MIN_N,
            "max_n": EXPECTED_MAX_N,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "unclamped_required_n": required,
            "confirmatory_n": confirmatory_n,
            "planning_constants": {
                "power_target": EXPECTED_POWER,
                "mesi_relative_gain": EXPECTED_MESI,
                "familywise_alpha": DEFAULT_FAMILYWISE_ALPHA,
                "endpoint_alpha": ENDPOINT_ALPHA,
                "alpha_tail": "one_sided_lower_bound",
                "z_alpha": normal.inv_cdf(1.0 - ENDPOINT_ALPHA),
                "z_power": normal.inv_cdf(EXPECTED_POWER),
            },
        },
        "confirmatory_lineage": {
            "lane": "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "seed_materialization_status": "NOT_EXECUTED",
            "reserved_replicate_ids": reserved_ids,
        },
        "lineage": {
            "protocol_digest": execution_artifact.get("protocol_digest"),
            "execution_artifact_digest": execution_artifact.get("artifact_digest"),
            "arm_registry_digest": arm_registry.get("registry_digest"),
            "analysis_code_digest": analysis_code_digest,
            "development_code_digest": execution_artifact.get("code_digest"),
            "arcs_final_state_digest": final_state.get("arcs_branch_digest"),
            "oracle_final_state_digest": final_state.get("oracle_cbrf_digest"),
        },
        "remaining_blockers": blockers,
        "prep_digest": "",
    }
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp277_confirmatory_prep(payload)
    if errors:
        raise RuntimeError("invalid EXP-277 confirmatory prep: " + "; ".join(errors))
    return payload
