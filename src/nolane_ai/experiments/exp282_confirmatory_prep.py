from __future__ import annotations

import math
from statistics import NormalDist, stdev
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-282-CONFIRMATORY-OPEN-PREP-V1"
EXPECTED_EXPERIMENT_ID = "EXP-282"
EXPECTED_PRIMARY_ENDPOINT = "grounded_decision_accuracy"
EXPECTED_MESI = 0.03
EXPECTED_POWER = 0.90
EXPECTED_MIN_N = 32
EXPECTED_MAX_N = 128
EXPECTED_PAIRED = True
DEFAULT_FAMILYWISE_ALPHA = 0.05
SAMPLE_SIZE_METHOD = "paired-normal-approximation-from-pilot-sd-v1"


def _prep_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def _validate_frozen_experiment(experiment: dict[str, Any], *, familywise_alpha: float) -> None:
    if experiment.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("confirmatory prep requires EXP-282")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != EXPECTED_PRIMARY_ENDPOINT or primary.get("direction") != "higher":
        raise ValueError("EXP-282 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "absolute_gain" or float(mesi.get("value", -1.0)) != EXPECTED_MESI:
        raise ValueError("EXP-282 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if float(sample.get("power_target", -1.0)) != EXPECTED_POWER:
        raise ValueError("EXP-282 power target drift")
    if int(sample.get("min_n", -1)) != EXPECTED_MIN_N or int(sample.get("max_n", -1)) != EXPECTED_MAX_N:
        raise ValueError("EXP-282 sample-size bounds drift")
    if bool(sample.get("paired")) is not EXPECTED_PAIRED:
        raise ValueError("EXP-282 paired-analysis contract drift")
    if float(familywise_alpha) != DEFAULT_FAMILYWISE_ALPHA:
        raise ValueError("EXP-282 alpha drift")


def _pilot_effects(execution_artifact: dict[str, Any]) -> list[float]:
    if execution_artifact.get("schema") != "NLM-EXP-282-PAIRED-DEV-EVAL-V1":
        raise ValueError("EXP-282 paired development artifact schema mismatch")
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("confirmatory prep requires EV-E2 / UNVERIFIED development evidence")
    if execution_artifact.get("confirmatory_ready") is not False:
        raise ValueError("development artifact cannot already claim confirmatory readiness")
    resource = execution_artifact.get("resource_match") or {}
    if resource.get("parameter_match") is not True or resource.get("observation_history_match") is not True:
        raise ValueError("EXP-282 paired pilot parameter/observation match is not closed")
    if resource.get("accounted_flop_match") is not True:
        raise ValueError("EXP-282 paired pilot accounted-FLOP match is not closed")
    if float(resource.get("relative_accounted_flop_difference", 1.0)) > 0.05:
        raise ValueError("EXP-282 paired pilot exceeds frozen compute guard")
    training = execution_artifact.get("training") or {}
    if training.get("rng_stream") != "augmentation":
        raise ValueError("EXP-282 pilot training must use augmentation RNG")
    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    train_digests = training.get("batch_digests") or []
    if train_start != 0 or train_count <= 0 or len(train_digests) != train_count:
        raise ValueError("EXP-282 pilot training replicate lineage is incomplete")
    training_ids = set(range(train_start, train_start + train_count))

    evaluation = execution_artifact.get("evaluation") or {}
    if evaluation.get("rng_stream") != "evaluation":
        raise ValueError("EXP-282 pilot must use evaluation RNG for held-out pilot data")
    raw = evaluation.get("per_replicate") or []
    if len(raw) < EXPECTED_MIN_N:
        raise ValueError("confirmatory prep requires at least 32 paired pilot replicates")
    declared_n = int(evaluation.get("replicates", 0) or 0)
    if declared_n != len(raw):
        raise ValueError("EXP-282 paired pilot replicate count mismatch")
    replicate_ids = [row.get("replicate") for row in raw]
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("EXP-282 paired pilot replicate IDs must be unique")
    if training_ids.intersection(replicate_ids):
        raise ValueError("EXP-282 training and evaluation replicate lineages overlap")
    effects: list[float] = []
    for row in raw:
        value = row.get("explicit_minus_recurrent_accuracy")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("EXP-282 paired pilot contains invalid accuracy contrast")
        recurrent_metrics = row.get("recurrent_hidden") or {}
        explicit_metrics = row.get("explicit_belief") or {}
        expected_accuracy = float(explicit_metrics.get("grounded_decision_accuracy", math.nan)) - float(
            recurrent_metrics.get("grounded_decision_accuracy", math.nan)
        )
        if not math.isfinite(expected_accuracy) or not math.isclose(float(value), expected_accuracy, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("EXP-282 paired pilot accuracy contrast mismatch")
        brier_value = row.get("explicit_minus_recurrent_brier")
        expected_brier = float(explicit_metrics.get("brier_score", math.nan)) - float(
            recurrent_metrics.get("brier_score", math.nan)
        )
        if not isinstance(brier_value, (int, float)) or not math.isfinite(float(brier_value)):
            raise ValueError("EXP-282 paired pilot contains invalid Brier contrast")
        if not math.isfinite(expected_brier) or not math.isclose(float(brier_value), expected_brier, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("EXP-282 paired pilot Brier contrast mismatch")
        effects.append(float(value))
    return effects


def _required_n(*, paired_sd: float, mesi: float, power: float, alpha: float) -> int:
    if paired_sd < 0.0 or not math.isfinite(paired_sd):
        raise ValueError("paired_sd must be finite and non-negative")
    if paired_sd == 0.0:
        return 1
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(power)
    raw_n = ((z_alpha + z_power) * paired_sd / mesi) ** 2
    return max(1, int(math.ceil(raw_n)))


def validate_exp282_confirmatory_prep(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-282 confirmatory prep schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("confirmatory prep cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("confirmatory prep cannot promote a neural claim")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("confirmatory prep cannot consume confirmatory data")

    confirmatory_lineage = payload.get("confirmatory_lineage") or {}
    if confirmatory_lineage.get("pilot_reuse_forbidden") is not True:
        errors.append("confirmatory lineage must forbid pilot reuse")

    frozen = payload.get("frozen_analysis") or {}
    if frozen.get("primary_endpoint") != EXPECTED_PRIMARY_ENDPOINT:
        errors.append("EXP-282 primary endpoint drift")
    if frozen.get("primary_direction") != "higher":
        errors.append("EXP-282 primary direction drift")
    if frozen.get("inference_tail") != "one_sided_lower_bound":
        errors.append("EXP-282 inference-tail drift")
    if float(frozen.get("mesi_absolute_gain", -1.0)) != EXPECTED_MESI:
        errors.append("EXP-282 MESI drift")
    if float(frozen.get("power_target", -1.0)) != EXPECTED_POWER:
        errors.append("EXP-282 power target drift")
    if float(frozen.get("familywise_alpha", -1.0)) != DEFAULT_FAMILYWISE_ALPHA:
        errors.append("EXP-282 alpha drift")
    if frozen.get("analysis_method") != "paired accuracy difference with bootstrap CI plus calibration guard":
        errors.append("EXP-282 analysis method drift")
    if frozen.get("multiplicity_family") != "BELIEF_STATE":
        errors.append("EXP-282 multiplicity family drift")
    if frozen.get("brier_guard") != "explicit_belief <= recurrent_hidden + 0.02":
        errors.append("EXP-282 Brier guard drift")
    if frozen.get("compute_guard") != "difference <= 0.05 relative unless included in primary cost normalization":
        errors.append("EXP-282 compute guard drift")

    lineage = payload.get("lineage") or {}
    for key in ("protocol_digest", "execution_artifact_digest", "arm_registry_digest", "analysis_code_digest"):
        if not lineage.get(key):
            errors.append(f"missing confirmatory prep lineage {key}")

    pilot = payload.get("pilot_summary") or {}
    pilot_n = int(pilot.get("n", 0) or 0)
    if pilot_n < EXPECTED_MIN_N:
        errors.append("confirmatory prep requires at least 32 paired pilot replicates")
    paired_sd = pilot.get("paired_sd")
    if not isinstance(paired_sd, (int, float)) or not math.isfinite(float(paired_sd)) or float(paired_sd) < 0.0:
        errors.append("invalid paired pilot SD")
    if pilot.get("source_lane") != "DEVELOPMENT_PILOT_ONLY":
        errors.append("EXP-282 pilot source lane drift")

    freeze = payload.get("sample_size_freeze") or {}
    if int(freeze.get("min_n", -1)) != EXPECTED_MIN_N or int(freeze.get("max_n", -1)) != EXPECTED_MAX_N:
        errors.append("EXP-282 frozen sample-size bounds drift")
    if freeze.get("paired") is not True:
        errors.append("EXP-282 paired sample-size contract drift")
    if freeze.get("pilot_reuse_as_confirmatory") is not False:
        errors.append("confirmatory prep must forbid pilot reuse as confirmatory")
    if freeze.get("method") != SAMPLE_SIZE_METHOD:
        errors.append("EXP-282 sample-size method drift")
    constants = freeze.get("planning_constants") or {}
    if constants.get("alpha_tail") != "one_sided_lower_bound":
        errors.append("EXP-282 sample-size alpha-tail drift")
    expected_z_alpha = NormalDist().inv_cdf(1.0 - DEFAULT_FAMILYWISE_ALPHA)
    expected_z_power = NormalDist().inv_cdf(EXPECTED_POWER)
    if not math.isclose(float(constants.get("z_alpha", math.nan)), expected_z_alpha, rel_tol=0.0, abs_tol=1e-12):
        errors.append("EXP-282 sample-size z_alpha drift")
    if not math.isclose(float(constants.get("z_power", math.nan)), expected_z_power, rel_tol=0.0, abs_tol=1e-12):
        errors.append("EXP-282 sample-size z_power drift")
    required = freeze.get("unclamped_required_n")
    if not isinstance(required, int) or required <= 0:
        errors.append("invalid unclamped required n")
    elif isinstance(paired_sd, (int, float)) and math.isfinite(float(paired_sd)) and float(paired_sd) >= 0.0:
        expected_required = _required_n(
            paired_sd=float(paired_sd),
            mesi=EXPECTED_MESI,
            power=EXPECTED_POWER,
            alpha=DEFAULT_FAMILYWISE_ALPHA,
        )
        if required != expected_required:
            errors.append("unclamped required n does not match frozen power formula")
    status = payload.get("status")
    confirmatory_n = freeze.get("confirmatory_n")
    reserved_ids = confirmatory_lineage.get("reserved_replicate_ids") or []
    pilot_ids = (payload.get("pilot_summary") or {}).get("replicate_ids") or []
    training_ids = (payload.get("pilot_summary") or {}).get("training_replicate_ids") or []
    if training_ids != list(range(len(training_ids))):
        errors.append("EXP-282 training replicate lineage drift")
    if set(training_ids).intersection(pilot_ids):
        errors.append("EXP-282 development training/evaluation replicate lineages overlap")
    if confirmatory_lineage.get("lane") != "CONFIRMATORY_OPEN_RESERVED_UNCONSUMED":
        errors.append("EXP-282 confirmatory lineage lane drift")
    if confirmatory_lineage.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("confirmatory seed materialization must remain NOT_EXECUTED during prep")
    if set(reserved_ids).intersection(pilot_ids):
        errors.append("confirmatory replicate IDs overlap development pilot IDs")
    if set(reserved_ids).intersection(training_ids):
        errors.append("confirmatory replicate IDs overlap development training IDs")
    if status == "CONFIRMATORY_OPEN_PREPARED":
        if not isinstance(confirmatory_n, int) or not (EXPECTED_MIN_N <= confirmatory_n <= EXPECTED_MAX_N):
            errors.append("prepared confirmatory n must be inside frozen bounds")
        if isinstance(required, int) and confirmatory_n != max(EXPECTED_MIN_N, required):
            errors.append("confirmatory n does not match frozen sample-size rule")
        if len(reserved_ids) != confirmatory_n:
            errors.append("reserved confirmatory replicate count does not match confirmatory n")
    elif status == "NOT_READY_VARIANCE_EXCEEDS_MAX_N":
        if confirmatory_n is not None:
            errors.append("confirmatory n must remain null when required n exceeds frozen maximum")
        if isinstance(required, int) and required <= EXPECTED_MAX_N:
            errors.append("variance-exceeds-max status requires n above frozen maximum")
        if reserved_ids:
            errors.append("not-ready prep cannot reserve confirmatory replicate IDs")
    else:
        errors.append("invalid EXP-282 confirmatory prep status")

    if payload.get("prep_digest") not in (None, "") and payload.get("prep_digest") != _prep_digest(payload):
        errors.append("confirmatory prep digest mismatch")
    return errors


def build_exp282_confirmatory_prep(
    *,
    experiment: dict[str, Any],
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    analysis_code_digest: str,
    familywise_alpha: float = DEFAULT_FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    _validate_frozen_experiment(experiment, familywise_alpha=familywise_alpha)
    if not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")
    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-282 confirmatory prep requires neural arm registry")
    if arm_registry.get("evidence_level") != "EV-E2" or arm_registry.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-282 arm registry must remain EV-E2 / UNVERIFIED")
    protocol_digest = execution_artifact.get("protocol_digest")
    if not protocol_digest or arm_registry.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-282 registry/execution protocol digest mismatch")
    execution_digest = execution_artifact.get("artifact_digest")
    registry_execution = ((arm_registry.get("experiments") or {}).get("EXP-282") or {}).get("paired_execution_evidence") or {}
    if not execution_digest or registry_execution.get("artifact_digest") != execution_digest:
        raise ValueError("EXP-282 registry execution artifact digest mismatch")
    if ((arm_registry.get("experiments") or {}).get("EXP-282") or {}).get("development_match_status") != "PAIRED_PARTIAL_OBSERVABILITY_DEV_READY":
        raise ValueError("EXP-282 Match Court has not reached paired development readiness")

    effects = _pilot_effects(execution_artifact)
    paired_sd = stdev(effects) if len(effects) >= 2 else 0.0
    mean_effect = sum(effects) / len(effects)
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - familywise_alpha)
    z_power = normal.inv_cdf(EXPECTED_POWER)
    required_n = _required_n(paired_sd=paired_sd, mesi=EXPECTED_MESI, power=EXPECTED_POWER, alpha=familywise_alpha)
    if required_n > EXPECTED_MAX_N:
        status = "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
        confirmatory_n: int | None = None
        blockers = [
            f"pilot variance implies required n above frozen maximum {EXPECTED_MAX_N}",
            "confirmatory-open execution has not consumed any reserved confirmatory data",
            "post-freeze challenge beacon and independent replication remain open",
        ]
    else:
        status = "CONFIRMATORY_OPEN_PREPARED"
        confirmatory_n = max(EXPECTED_MIN_N, required_n)
        blockers = [
            "confirmatory-open execution has not consumed any reserved confirmatory data",
            "post-freeze challenge beacon and independent replication remain open",
        ]

    pilot_ids = [int(row["replicate"]) for row in execution_artifact["evaluation"]["per_replicate"]]
    train_start = int(execution_artifact["training"]["start_replicate"])
    train_count = int(execution_artifact["training"]["replicates"])
    training_ids = list(range(train_start, train_start + train_count))
    if confirmatory_n is None:
        reserved_ids: list[int] = []
    else:
        reserve_start = max([*pilot_ids, *training_ids]) + 1
        reserved_ids = list(range(reserve_start, reserve_start + confirmatory_n))

    protected = experiment.get("protected_endpoints") or []
    brier_guard = next((item.get("floor") for item in protected if item.get("metric") == "brier_score"), None)
    compute_guard = next((item.get("floor") for item in protected if item.get("metric") == "accounted_flops"), None)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": status,
        "scope": "exp282-confirmatory-open-preparation-only",
        "confirmatory_data_consumed": False,
        "frozen_analysis": {
            "primary_endpoint": EXPECTED_PRIMARY_ENDPOINT,
            "primary_direction": "higher",
            "mesi_absolute_gain": EXPECTED_MESI,
            "power_target": EXPECTED_POWER,
            "familywise_alpha": float(familywise_alpha),
            "inference_tail": "one_sided_lower_bound",
            "analysis_method": str(experiment.get("analysis_method")),
            "multiplicity_family": str(experiment.get("multiplicity_family")),
            "brier_guard": brier_guard,
            "compute_guard": compute_guard,
        },
        "pilot_summary": {
            "n": len(effects),
            "mean_accuracy_gain": mean_effect,
            "paired_sd": paired_sd,
            "paired_effect_digest": canonical_sha256({"replicate_ids": pilot_ids, "effects": effects}),
            "replicate_start": min(pilot_ids),
            "replicate_end": max(pilot_ids),
            "replicate_ids": pilot_ids,
            "training_replicate_ids": training_ids,
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
        },
        "sample_size_freeze": {
            "method": SAMPLE_SIZE_METHOD,
            "unclamped_required_n": required_n,
            "confirmatory_n": confirmatory_n,
            "min_n": EXPECTED_MIN_N,
            "max_n": EXPECTED_MAX_N,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "planning_constants": {
                "alpha_tail": "one_sided_lower_bound",
                "z_alpha": z_alpha,
                "z_power": z_power,
                "formula": "ceil(((z_alpha + z_power) * paired_sd / mesi)^2), then apply frozen min_n without max_n truncation",
            },
        },
        "confirmatory_lineage": {
            "lane": "CONFIRMATORY_OPEN_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "reserved_replicate_ids": reserved_ids,
            "seed_materialization_status": "NOT_EXECUTED",
        },
        "lineage": {
            "protocol_digest": protocol_digest,
            "execution_artifact_digest": execution_digest,
            "arm_registry_digest": arm_registry.get("registry_digest"),
            "analysis_code_digest": analysis_code_digest,
        },
        "remaining_blockers": blockers,
        "prep_digest": "",
    }
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp282_confirmatory_prep(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 confirmatory prep: " + "; ".join(errors))
    return payload
