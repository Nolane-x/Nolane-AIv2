from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist, stdev
from typing import Any

from nolane_ai.experiments.exp297_paired_runner import validate_exp297_execution
from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-297-CONFIRMATORY-GATE-A-PREP-V1"
EXPERIMENT_ID = "EXP-297"
PRIMARY_ENDPOINT = "semantic_fidelity_balanced_accuracy"
MESI = 0.10
POWER = 0.90
MIN_N = 32
MAX_N = 128
FAMILYWISE_ALPHA = 0.05
ENDPOINT_COUNT = 3
ALPHA_PER_ENDPOINT = FAMILYWISE_ALPHA / ENDPOINT_COUNT
BOOTSTRAP_SAMPLES = 10_000
WRONG_AUTHORITY_LIMIT = 0.05
FAITHFUL_REJECTION_LIMIT = 0.10
ANALYSIS_METHOD = (
    "paired balanced-accuracy difference with Wilson/bootstrap safety intervals "
    "for wrong-authority rate"
)
MULTIPLICITY_FAMILY = "ENCODING_FIDELITY"
SAMPLE_SIZE_METHOD = "paired-normal-approximation-from-development-ba-sd-v1"


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


def _validate_experiment(experiment: dict[str, Any]) -> None:
    if experiment.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("confirmatory prep requires frozen EXP-297")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != PRIMARY_ENDPOINT or primary.get("direction") != "higher":
        raise ValueError("EXP-297 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "absolute_gain" or not _close(mesi.get("value"), MESI):
        raise ValueError("EXP-297 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if (
        not _close(sample.get("power_target"), POWER)
        or int(sample.get("min_n", -1)) != MIN_N
        or int(sample.get("max_n", -1)) != MAX_N
        or sample.get("paired") is not True
    ):
        raise ValueError("EXP-297 sample-size contract drift")
    if experiment.get("analysis_method") != ANALYSIS_METHOD:
        raise ValueError("EXP-297 analysis method drift")
    if experiment.get("multiplicity_family") != MULTIPLICITY_FAMILY:
        raise ValueError("EXP-297 multiplicity family drift")
    protected = {
        item.get("metric"): item.get("floor")
        for item in (experiment.get("protected_endpoints") or [])
    }
    if protected.get("wrong_formalization_authority_rate") != "<=0.05":
        raise ValueError("EXP-297 wrong-authority safety floor drift")
    if protected.get("faithful_formalization_rejection_rate") != "<=0.10":
        raise ValueError("EXP-297 faithful-rejection safety floor drift")


def _validate_registry_binding(
    arm_registry: dict[str, Any], execution_artifact: dict[str, Any]
) -> None:
    registry_digest = arm_registry.get("registry_digest")
    if not isinstance(registry_digest, str) or not registry_digest:
        raise ValueError("EXP-297 arm registry digest missing")
    item = (arm_registry.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    if item.get("evidence_level") != "EV-E2" or item.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-297 arm registry must remain EV-E2 / UNVERIFIED")
    if item.get("match_court") != "BLOCKED":
        raise ValueError("EXP-297 arm registry match court must remain BLOCKED")
    execution_digest = execution_artifact.get("artifact_digest")
    registry_execution = item.get("paired_execution_evidence") or {}
    if not execution_digest or registry_execution.get("execution_digest") != execution_digest:
        raise ValueError("EXP-297 arm registry/development execution binding mismatch")


def _balanced_accuracy(rows: list[dict[str, Any]], arm_id: str) -> float:
    faithful = [row for row in rows if row.get("is_faithful") is True]
    wrong = [row for row in rows if row.get("is_faithful") is False]
    if not faithful or not wrong:
        raise ValueError("EXP-297 pilot replicate must contain both semantic classes")
    tp = sum(bool((row.get("arms") or {}).get(arm_id, {}).get("authority_granted")) for row in faithful)
    tn = sum(not bool((row.get("arms") or {}).get(arm_id, {}).get("authority_granted")) for row in wrong)
    return 0.5 * (tp / len(faithful) + tn / len(wrong))


def _pilot_effects(execution_artifact: dict[str, Any]) -> tuple[list[int], list[float]]:
    execution_errors = validate_exp297_execution(execution_artifact)
    if execution_errors:
        raise ValueError("invalid EXP-297 development artifact: " + "; ".join(execution_errors))
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("confirmatory prep requires EV-E2 / UNVERIFIED development evidence")
    if execution_artifact.get("confirmatory_ready") is not False:
        raise ValueError("development artifact cannot already claim confirmatory readiness")
    rows = (execution_artifact.get("evaluation") or {}).get("raw_candidates") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("EXP-297 development pilot raw rows missing")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        replicate = row.get("replicate")
        if not isinstance(replicate, int) or replicate < 0:
            raise ValueError("EXP-297 development replicate ID invalid")
        grouped.setdefault(replicate, []).append(row)
    replicate_ids = sorted(grouped)
    if len(replicate_ids) < MIN_N:
        raise ValueError("EXP-297 confirmatory prep requires at least 32 development replicates")
    config = execution_artifact.get("config") or {}
    if int(config.get("eval_replicates", -1)) != len(replicate_ids):
        raise ValueError("EXP-297 development pilot replicate count mismatch")

    effects: list[float] = []
    for replicate in replicate_ids:
        replicate_rows = grouped[replicate]
        if len(replicate_rows) != 16:
            raise ValueError("EXP-297 pilot replicate must contain exactly 16 candidates")
        indices = [row.get("candidate_index") for row in replicate_rows]
        if indices != list(range(16)):
            raise ValueError("EXP-297 pilot candidate order is not canonical")
        compile_ba = _balanced_accuracy(replicate_rows, "compile_only")
        fidelity_ba = _balanced_accuracy(replicate_rows, "fidelity_court")
        effects.append(fidelity_ba - compile_ba)
    return replicate_ids, effects


def _required_n(*, paired_sd: float) -> int:
    if not math.isfinite(paired_sd) or paired_sd < 0.0:
        raise ValueError("paired_sd must be finite and non-negative")
    if paired_sd == 0.0:
        return 1
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - ALPHA_PER_ENDPOINT)
    z_power = normal.inv_cdf(POWER)
    raw_n = ((z_alpha + z_power) * paired_sd / MESI) ** 2
    return max(1, int(math.ceil(raw_n)))


def _frozen_analysis() -> dict[str, Any]:
    return {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": "higher",
        "paired_contrast": "fidelity_court_minus_compile_only",
        "mesi_absolute_gain": MESI,
        "power_target": POWER,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "alpha_per_endpoint": ALPHA_PER_ENDPOINT,
        "endpoint_count": ENDPOINT_COUNT,
        "primary_inference": "one_sided_lower_bootstrap_bound",
        "wrong_authority_guard": "one_sided_wilson_upper_bound<=0.05",
        "faithful_rejection_guard": "one_sided_wilson_upper_bound<=0.10",
        "wrong_authority_limit": WRONG_AUTHORITY_LIMIT,
        "faithful_rejection_limit": FAITHFUL_REJECTION_LIMIT,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "analysis_method": ANALYSIS_METHOD,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "court_inconclusive_policy": "no_authority_and_counts_as_faithful_rejection",
    }


def build_exp297_confirmatory_prep(
    *,
    experiment: dict[str, Any],
    execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    _validate_experiment(experiment)
    if not isinstance(analysis_code_digest, str) or not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")
    _validate_registry_binding(arm_registry, execution_artifact)
    replicate_ids, effects = _pilot_effects(execution_artifact)
    paired_sd = stdev(effects) if len(effects) > 1 else 0.0
    required = _required_n(paired_sd=paired_sd)

    if required > MAX_N:
        status = "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
        confirmatory_n: int | None = None
        reserved: list[int] = []
    else:
        status = "CONFIRMATORY_GATE_A_PREPARED"
        confirmatory_n = max(MIN_N, required)
        start = max(replicate_ids) + 1
        reserved = list(range(start, start + confirmatory_n))

    normal = NormalDist()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": status,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "semantic_authority_promoted": False,
        "frozen_analysis": _frozen_analysis(),
        "pilot_summary": {
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
            "n": len(replicate_ids),
            "replicate_ids": replicate_ids,
            "paired_balanced_accuracy_effects": effects,
            "paired_sd": paired_sd,
            "execution_artifact_digest": execution_artifact["artifact_digest"],
        },
        "sample_size_freeze": {
            "method": SAMPLE_SIZE_METHOD,
            "min_n": MIN_N,
            "max_n": MAX_N,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "unclamped_required_n": required,
            "confirmatory_n": confirmatory_n,
            "planning_constants": {
                "mesi": MESI,
                "power": POWER,
                "familywise_alpha": FAMILYWISE_ALPHA,
                "alpha_per_endpoint": ALPHA_PER_ENDPOINT,
                "z_alpha": normal.inv_cdf(1.0 - ALPHA_PER_ENDPOINT),
                "z_power": normal.inv_cdf(POWER),
            },
        },
        "confirmatory_lineage": {
            "lane": "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "seed_materialization_status": "NOT_EXECUTED",
            "reserved_replicate_ids": reserved,
        },
        "lineage": {
            "protocol_digest": execution_artifact["protocol_digest"],
            "development_execution_digest": execution_artifact["artifact_digest"],
            "development_code_digest": execution_artifact["code_digest"],
            "arm_registry_digest": arm_registry["registry_digest"],
            "pair_audit_digest": canonical_sha256(
                (execution_artifact.get("resource_match") or {}).get("pair_audit") or {}
            ),
            "analysis_code_digest": analysis_code_digest,
        },
        "development_execution_artifact": deepcopy(execution_artifact),
        "prep_digest": "",
    }
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp297_confirmatory_prep(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 confirmatory prep: " + "; ".join(errors))
    return payload


def validate_exp297_confirmatory_prep(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-297 confirmatory prep identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 confirmatory prep cannot promote evidence")
    required_false = (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
        "semantic_authority_promoted",
    )
    for key in required_false:
        if payload.get(key) is not False:
            errors.append(f"EXP-297 confirmatory prep forbidden flag enabled: {key}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-297 confirmatory prep cannot materialize seeds")
    if payload.get("prep_digest") not in (None, "") and payload.get("prep_digest") != _prep_digest(payload):
        errors.append("EXP-297 confirmatory prep digest mismatch")

    frozen = payload.get("frozen_analysis") or {}
    expected_frozen = _frozen_analysis()
    for key, expected in expected_frozen.items():
        observed = frozen.get(key)
        if isinstance(expected, float):
            if not _close(observed, expected):
                errors.append(f"EXP-297 frozen analysis {key} drift")
        elif observed != expected:
            errors.append(f"EXP-297 frozen analysis {key} drift")

    execution = payload.get("development_execution_artifact")
    if not isinstance(execution, dict):
        errors.append("EXP-297 development execution evidence missing")
        return errors
    try:
        replicate_ids, effects = _pilot_effects(execution)
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    pilot = payload.get("pilot_summary") or {}
    if pilot.get("source_lane") != "DEVELOPMENT_PILOT_ONLY":
        errors.append("EXP-297 pilot source lane drift")
    if pilot.get("n") != len(replicate_ids) or pilot.get("replicate_ids") != replicate_ids:
        errors.append("EXP-297 pilot replicate lineage mismatch")
    if pilot.get("paired_balanced_accuracy_effects") != effects:
        errors.append("EXP-297 pilot paired BA effects mismatch")
    expected_sd = stdev(effects) if len(effects) > 1 else 0.0
    if not _close(pilot.get("paired_sd"), expected_sd):
        errors.append("EXP-297 pilot paired SD mismatch")
    if pilot.get("execution_artifact_digest") != execution.get("artifact_digest"):
        errors.append("EXP-297 pilot execution digest mismatch")

    lineage = payload.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": execution.get("protocol_digest"),
        "development_execution_digest": execution.get("artifact_digest"),
        "development_code_digest": execution.get("code_digest"),
        "pair_audit_digest": canonical_sha256(
            (execution.get("resource_match") or {}).get("pair_audit") or {}
        ),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            errors.append(f"EXP-297 confirmatory prep lineage {key} mismatch")
    if not lineage.get("arm_registry_digest") or not lineage.get("analysis_code_digest"):
        errors.append("EXP-297 confirmatory prep registry/analysis lineage missing")

    freeze = payload.get("sample_size_freeze") or {}
    if freeze.get("method") != SAMPLE_SIZE_METHOD:
        errors.append("EXP-297 sample-size method drift")
    if freeze.get("min_n") != MIN_N or freeze.get("max_n") != MAX_N or freeze.get("paired") is not True:
        errors.append("EXP-297 sample-size bounds/paired contract drift")
    if freeze.get("pilot_reuse_as_confirmatory") is not False:
        errors.append("EXP-297 pilot reuse must remain forbidden")
    expected_required = _required_n(paired_sd=expected_sd)
    if freeze.get("unclamped_required_n") != expected_required:
        errors.append("EXP-297 unclamped required n mismatch")
    constants = freeze.get("planning_constants") or {}
    normal = NormalDist()
    expected_constants = {
        "mesi": MESI,
        "power": POWER,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "alpha_per_endpoint": ALPHA_PER_ENDPOINT,
        "z_alpha": normal.inv_cdf(1.0 - ALPHA_PER_ENDPOINT),
        "z_power": normal.inv_cdf(POWER),
    }
    for key, expected in expected_constants.items():
        if not _close(constants.get(key), expected):
            errors.append(f"EXP-297 sample-size constant {key} drift")

    confirmatory = payload.get("confirmatory_lineage") or {}
    if confirmatory.get("lane") != "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED":
        errors.append("EXP-297 confirmatory lane drift")
    if confirmatory.get("pilot_reuse_forbidden") is not True:
        errors.append("EXP-297 confirmatory lineage must forbid pilot reuse")
    if confirmatory.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-297 prep cannot materialize confirmatory seeds")
    reserved = confirmatory.get("reserved_replicate_ids") or []
    status = payload.get("status")
    confirmatory_n = freeze.get("confirmatory_n")
    if expected_required > MAX_N:
        if status != "NOT_READY_VARIANCE_EXCEEDS_MAX_N" or confirmatory_n is not None or reserved:
            errors.append("EXP-297 high-variance not-ready freeze mismatch")
    else:
        expected_n = max(MIN_N, expected_required)
        expected_start = max(replicate_ids) + 1
        expected_reserved = list(range(expected_start, expected_start + expected_n))
        if status != "CONFIRMATORY_GATE_A_PREPARED":
            errors.append("EXP-297 prepared status mismatch")
        if confirmatory_n != expected_n:
            errors.append("EXP-297 confirmatory n mismatch")
        if reserved != expected_reserved:
            errors.append("EXP-297 reserved confirmatory IDs mismatch")
        if set(reserved).intersection(replicate_ids):
            errors.append("EXP-297 confirmatory IDs overlap development pilot")
    return errors
