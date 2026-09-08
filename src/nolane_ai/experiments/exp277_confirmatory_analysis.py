from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import random
from statistics import fmean, median
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp277_confirmatory_executor import validate_exp277_confirmatory_raw
from .exp277_confirmatory_prep import (
    BOOTSTRAP_SAMPLES,
    DEFAULT_FAMILYWISE_ALPHA,
    ENDPOINT_ALPHA,
    EXPECTED_ANALYSIS_METHOD,
    EXPECTED_ENERGY_FLOOR,
    EXPECTED_MESI,
    EXPECTED_MULTIPLICITY_FAMILY,
    EXPECTED_POWER,
    EXPECTED_PRIMARY_ENDPOINT,
    EXPECTED_SOLUTION_FLOOR,
)


SCHEMA = "NLM-EXP-277-CONFIRMATORY-ANALYSIS-V1"
EXPERIMENT_ID = "EXP-277"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_ANALYZED"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_ANALYZED"
VALID_SCIENTIFIC_DECISIONS = {
    "PROMOTE_TO_NEXT_STAGE",
    "HOLD_UNSTABLE",
    "KILL_SUBSYSTEM",
}
SOLUTION_FLOOR_DIFFERENCE = -0.005
PRIMARY_BOOTSTRAP_STREAM = "ratio-of-means-bootstrap-v1"
SOLUTION_BOOTSTRAP_STREAM = "solution-rate-bootstrap-v1"
DENOMINATOR_POLICY = "no_epsilon"
WALL_ENERGY_REPORT = {
    "stage_a_role": "report-only",
    "numeric_measurement": None,
    "fabricated": False,
    "decision_use": False,
    "note": "not measured by EXP-277 confirmatory executor; no numeric value fabricated",
}


def _analysis_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("analysis_digest", None)
    return canonical_sha256(clean)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _quantile_indices(count: int, alpha: float) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("EXP-277 bootstrap requires at least one valid resample")
    lower = max(0, math.ceil(float(alpha) * count) - 1)
    upper = min(count - 1, math.ceil((1.0 - float(alpha)) * count) - 1)
    return lower, upper


def bootstrap_exp277_ratio_of_means(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ENDPOINT_ALPHA,
) -> dict[str, Any]:
    if (
        not isinstance(baseline_values, list)
        or not isinstance(candidate_values, list)
        or not baseline_values
        or len(baseline_values) != len(candidate_values)
    ):
        raise ValueError("EXP-277 ratio-of-means bootstrap requires equal non-empty paired values")
    if any(not _finite(value) for value in baseline_values + candidate_values):
        raise ValueError("EXP-277 ratio-of-means bootstrap requires finite paired values")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-277 ratio-of-means bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-277 ratio-of-means bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-277 ratio-of-means bootstrap alpha must be in (0, 0.5)")

    baseline = [float(value) for value in baseline_values]
    candidate = [float(value) for value in candidate_values]
    observed_baseline = float(fmean(baseline))
    observed_candidate = float(fmean(candidate))
    observed_valid = math.isfinite(observed_baseline) and observed_baseline > 0.0
    observed_gain = (
        (observed_candidate / observed_baseline) - 1.0 if observed_valid else None
    )

    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(baseline)
    gains: list[float] = []
    invalid_denominators = 0
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        baseline_mean = float(fmean(baseline[index] for index in indices))
        candidate_mean = float(fmean(candidate[index] for index in indices))
        if (
            not math.isfinite(baseline_mean)
            or baseline_mean <= 0.0
            or not math.isfinite(candidate_mean)
        ):
            invalid_denominators += 1
            continue
        gain = candidate_mean / baseline_mean - 1.0
        if not math.isfinite(gain):
            invalid_denominators += 1
            continue
        gains.append(float(gain))

    gains.sort()
    if gains:
        lower_index, upper_index = _quantile_indices(len(gains), float(alpha))
        lower: float | None = float(gains[lower_index])
        upper: float | None = float(gains[upper_index])
        bootstrap_median: float | None = float(median(gains))
    else:
        lower = None
        upper = None
        bootstrap_median = None

    denominator_valid = bool(
        observed_valid and invalid_denominators == 0 and len(gains) == samples
    )
    return {
        "observed_baseline_mean": observed_baseline,
        "observed_candidate_mean": observed_candidate,
        "observed_relative_gain": observed_gain,
        "one_sided_lower": lower,
        "one_sided_upper": upper,
        "bootstrap_median": bootstrap_median,
        "samples": samples,
        "valid_resamples": len(gains),
        "invalid_denominator_resamples": invalid_denominators,
        "alpha": float(alpha),
        "seed_digest": seed_digest,
        "denominator_policy": DENOMINATOR_POLICY,
        "denominator_valid": denominator_valid,
    }


def _bootstrap_paired_difference(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ENDPOINT_ALPHA,
) -> dict[str, Any]:
    if (
        not baseline_values
        or len(baseline_values) != len(candidate_values)
        or any(not _finite(value) for value in baseline_values + candidate_values)
    ):
        raise ValueError("EXP-277 protected bootstrap requires equal finite paired values")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-277 protected bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-277 protected bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-277 protected bootstrap alpha must be in (0, 0.5)")

    baseline = [float(value) for value in baseline_values]
    candidate = [float(value) for value in candidate_values]
    paired = [c - b for b, c in zip(baseline, candidate, strict=True)]
    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(paired)
    means = [
        float(fmean(paired[rng.randrange(n)] for _ in range(n)))
        for _ in range(samples)
    ]
    means.sort()
    lower_index, upper_index = _quantile_indices(samples, float(alpha))
    observed = float(fmean(paired))
    return {
        "observed_baseline_mean": float(fmean(baseline)),
        "observed_candidate_mean": float(fmean(candidate)),
        "observed_difference": observed,
        "median_paired_difference": float(median(paired)),
        "one_sided_lower": float(means[lower_index]),
        "one_sided_upper": float(means[upper_index]),
        "positive_count": sum(value > 0.0 for value in paired),
        "zero_count": sum(value == 0.0 for value in paired),
        "negative_count": sum(value < 0.0 for value in paired),
        "samples": samples,
        "alpha": float(alpha),
        "seed_digest": seed_digest,
    }


def decide_exp277_confirmatory_outcome(
    *,
    primary_lower: float | None,
    primary_upper: float | None,
    solution_lower: float | None,
    solution_upper: float | None,
    denominator_valid: bool,
) -> str:
    if not isinstance(denominator_valid, bool):
        raise ValueError("EXP-277 denominator validity flag must be boolean")
    for label, value in (
        ("primary_lower", primary_lower),
        ("primary_upper", primary_upper),
        ("solution_lower", solution_lower),
        ("solution_upper", solution_upper),
    ):
        if value is not None and not _finite(value):
            raise ValueError(f"EXP-277 confirmatory decision {label} must be finite or None")

    # Frozen kill conditions are evaluated first. This preserves a genuinely
    # independent failure even when the primary denominator cannot support a
    # promotion claim.
    if primary_upper is not None and float(primary_upper) < EXPECTED_MESI:
        return "KILL_SUBSYSTEM"
    if solution_upper is not None and float(solution_upper) < SOLUTION_FLOOR_DIFFERENCE:
        return "KILL_SUBSYSTEM"

    # No epsilon repair: an observed or bootstrap denominator failure blocks
    # promotion and resolves to HOLD unless an independent kill was proven.
    if denominator_valid is not True:
        return "HOLD_UNSTABLE"

    if (
        primary_lower is not None
        and solution_lower is not None
        and float(primary_lower) >= EXPECTED_MESI
        and float(solution_lower) >= SOLUTION_FLOOR_DIFFERENCE
    ):
        return "PROMOTE_TO_NEXT_STAGE"
    return "HOLD_UNSTABLE"


def _frozen_contract() -> dict[str, Any]:
    return {
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
    }


def _sealed_analysis_identity(raw: dict[str, Any]) -> tuple[str, str]:
    seal = raw.get("seal") or {}
    authorization = seal.get("authorization_snapshot") or {}
    analysis_code_digest = authorization.get("analysis_code_digest")
    frozen_analysis_digest = authorization.get("frozen_analysis_digest")
    if not isinstance(analysis_code_digest, str) or len(analysis_code_digest) != 64:
        raise ValueError("EXP-277 pre-beacon frozen analysis code identity missing")
    if not isinstance(frozen_analysis_digest, str) or len(frozen_analysis_digest) != 64:
        raise ValueError("EXP-277 pre-beacon frozen analysis contract digest missing")
    expected_frozen_digest = canonical_sha256(_frozen_contract())
    if frozen_analysis_digest != expected_frozen_digest:
        raise ValueError("EXP-277 pre-beacon frozen analysis contract digest drift")
    return analysis_code_digest, frozen_analysis_digest


def _replicate_statistics(raw: dict[str, Any]) -> dict[str, Any]:
    reserved = list(raw.get("reserved_replicate_ids") or [])
    rows = raw.get("per_replicate")
    if not isinstance(rows, list) or [row.get("replicate") for row in rows] != reserved:
        raise ValueError("EXP-277 analysis raw replicate lineage mismatch")

    baseline_utility: list[float] = []
    candidate_utility: list[float] = []
    baseline_solution: list[float] = []
    candidate_solution: list[float] = []
    per_replicate: list[dict[str, Any]] = []
    divergence_count = 0
    positive = zero = negative = 0

    for row in rows:
        arcs = row.get("arcs_branch") or {}
        oracle = row.get("oracle_cbrf") or {}
        a_utility = arcs.get(EXPECTED_PRIMARY_ENDPOINT)
        o_utility = oracle.get(EXPECTED_PRIMARY_ENDPOINT)
        a_solution = arcs.get("verified_solution_rate")
        o_solution = oracle.get("verified_solution_rate")
        if any(not _finite(value) for value in (a_utility, o_utility, a_solution, o_solution)):
            raise ValueError("EXP-277 analysis raw metrics contain non-finite values")
        a_u = float(a_utility)
        o_u = float(o_utility)
        a_s = float(a_solution)
        o_s = float(o_solution)
        baseline_utility.append(a_u)
        candidate_utility.append(o_u)
        baseline_solution.append(a_s)
        candidate_solution.append(o_s)
        utility_difference = o_u - a_u
        solution_difference = o_s - a_s
        if utility_difference > 0.0:
            positive += 1
        elif utility_difference < 0.0:
            negative += 1
        else:
            zero += 1
        if a_u <= 0.0:
            divergence_count += 1
        relative_gain = (o_u / a_u - 1.0) if a_u > 0.0 else None
        per_replicate.append(
            {
                "replicate": row.get("replicate"),
                "arcs_utility": a_u,
                "oracle_utility": o_u,
                "utility_difference": utility_difference,
                "relative_utility_gain": relative_gain,
                "arcs_solution_rate": a_s,
                "oracle_solution_rate": o_s,
                "solution_rate_difference": solution_difference,
                "baseline_denominator_positive": a_u > 0.0,
            }
        )

    utility_differences = [item["utility_difference"] for item in per_replicate]
    return {
        "baseline_utility": baseline_utility,
        "candidate_utility": candidate_utility,
        "baseline_solution": baseline_solution,
        "candidate_solution": candidate_solution,
        "per_replicate": per_replicate,
        "diagnostics": {
            "paired_utility_difference_mean": float(fmean(utility_differences)),
            "paired_utility_difference_median": float(median(utility_differences)),
            "positive_count": positive,
            "zero_count": zero,
            "negative_count": negative,
            "positive_sign_consistency": positive / len(rows),
            "baseline_nonpositive_or_nonfinite_count": divergence_count,
            "baseline_denominator_divergence_rate": divergence_count / len(rows),
        },
    }


def _expected_statistics(raw: dict[str, Any]) -> dict[str, Any]:
    sealed = raw.get("seal", {}).get("authorization_snapshot") or {}
    protocol_digest = sealed.get("protocol_digest")
    if not isinstance(protocol_digest, str) or not protocol_digest:
        raise ValueError("EXP-277 frozen protocol lineage missing from analysis")
    _, frozen_analysis_digest = _sealed_analysis_identity(raw)
    metrics = _replicate_statistics(raw)

    primary_seed = (
        f"{protocol_digest}|{frozen_analysis_digest}|{EXPERIMENT_ID}|{PRIMARY_BOOTSTRAP_STREAM}"
    )
    primary = bootstrap_exp277_ratio_of_means(
        metrics["baseline_utility"],
        metrics["candidate_utility"],
        seed_material=primary_seed,
        samples=BOOTSTRAP_SAMPLES,
        alpha=ENDPOINT_ALPHA,
    )
    primary.update(
        {
            "metric": EXPECTED_PRIMARY_ENDPOINT,
            "effect_type": "ratio_of_means_relative_gain",
            "baseline_arm": "arcs_branch",
            "candidate_arm": "oracle_cbrf",
            "mesi_relative_gain": EXPECTED_MESI,
            "replicates": len(metrics["per_replicate"]),
        }
    )

    solution_seed = (
        f"{protocol_digest}|{frozen_analysis_digest}|{EXPERIMENT_ID}|{SOLUTION_BOOTSTRAP_STREAM}"
    )
    solution = _bootstrap_paired_difference(
        metrics["baseline_solution"],
        metrics["candidate_solution"],
        seed_material=solution_seed,
        samples=BOOTSTRAP_SAMPLES,
        alpha=ENDPOINT_ALPHA,
    )
    solution.update(
        {
            "metric": "verified_solution_rate",
            "effect_type": "paired_absolute_difference",
            "baseline_arm": "arcs_branch",
            "candidate_arm": "oracle_cbrf",
            "floor_difference": SOLUTION_FLOOR_DIFFERENCE,
            "lower_floor_pass": float(solution["one_sided_lower"]) >= SOLUTION_FLOOR_DIFFERENCE,
            "upper_definite_failure": float(solution["one_sided_upper"]) < SOLUTION_FLOOR_DIFFERENCE,
        }
    )

    decision = decide_exp277_confirmatory_outcome(
        primary_lower=primary["one_sided_lower"],
        primary_upper=primary["one_sided_upper"],
        solution_lower=solution["one_sided_lower"],
        solution_upper=solution["one_sided_upper"],
        denominator_valid=bool(primary["denominator_valid"]),
    )
    diagnostics = deepcopy(metrics["diagnostics"])
    diagnostics["invalid_denominator_resamples"] = primary["invalid_denominator_resamples"]
    diagnostics["bootstrap_denominator_divergence_rate"] = (
        primary["invalid_denominator_resamples"] / BOOTSTRAP_SAMPLES
    )
    return {
        "primary": primary,
        "solution": solution,
        "per_replicate": metrics["per_replicate"],
        "diagnostics": diagnostics,
        "would_be_decision": decision,
    }


def _lineage(raw: dict[str, Any], *, analysis_code_digest: str) -> dict[str, Any]:
    seal = raw.get("seal") or {}
    reconstruction = raw.get("reconstruction_authorization") or {}
    authorization = seal.get("authorization_snapshot") or {}
    return {
        "protocol_digest": authorization.get("protocol_digest"),
        "source_tree_digest": authorization.get("source_tree_digest"),
        "frozen_analysis_digest": authorization.get("frozen_analysis_digest"),
        "analysis_code_digest": analysis_code_digest,
        "seal_digest": seal.get("seal_digest"),
        "reconstruction_digest": reconstruction.get("reconstruction_digest"),
        "raw_artifact_digest": raw.get("artifact_digest"),
        "beacon_receipt_digest": (raw.get("lineage") or {}).get("beacon_receipt_digest"),
        "checkpoint_scientific_identity_digest": (raw.get("lineage") or {}).get(
            "checkpoint_scientific_identity_digest"
        ),
        "executor_code_digest": (raw.get("lineage") or {}).get("executor_code_digest"),
    }


def build_exp277_confirmatory_analysis(
    *,
    raw_artifact: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    raw_errors = validate_exp277_confirmatory_raw(
        raw_artifact,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        raise ValueError("invalid EXP-277 confirmatory raw artifact: " + "; ".join(raw_errors))
    if raw_artifact.get("status") == "INVALID_RUN" or raw_artifact.get("decision") == "INVALID_RUN":
        raise ValueError("invalid EXP-277 confirmatory raw artifact cannot be analyzed")

    frozen_code, _ = _sealed_analysis_identity(raw_artifact)
    if analysis_code_digest != frozen_code:
        raise ValueError("EXP-277 analysis code digest does not match pre-beacon freeze")

    stats = _expected_statistics(raw_artifact)
    test_only = raw_artifact.get("test_only") is True
    if test_only:
        evidence_level = "EV-E2"
        decision = "UNVERIFIED"
        status = TEST_ONLY_STATUS
        scientific_eligible = False
        confirmatory_consumed = False
        synthetic_consumed = True
        decision_rule_executed = False
        test_only_decision_executed = True
        test_only_would_be_decision = stats["would_be_decision"]
    else:
        evidence_level = "EV-E3"
        decision = stats["would_be_decision"]
        status = SCIENTIFIC_STATUS
        scientific_eligible = raw_artifact.get("scientific_evidence_eligible") is True
        confirmatory_consumed = True
        synthetic_consumed = False
        decision_rule_executed = True
        test_only_decision_executed = False
        test_only_would_be_decision = None

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": evidence_level,
        "decision": decision,
        "status": status,
        "test_only": test_only,
        "scientific_evidence_eligible": scientific_eligible,
        "confirmatory_data_consumed": confirmatory_consumed,
        "synthetic_challenge_data_consumed": synthetic_consumed,
        "challenge_materialized": True,
        "decision_rule_executed": decision_rule_executed,
        "test_only_decision_executed": test_only_decision_executed,
        "test_only_would_be_decision": test_only_would_be_decision,
        "confirmatory_n": raw_artifact.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(raw_artifact.get("reserved_replicate_ids") or []),
        "frozen_contract": _frozen_contract(),
        "primary_endpoint": stats["primary"],
        "protected_solution_rate": stats["solution"],
        "wall_energy_per_episode": deepcopy(WALL_ENERGY_REPORT),
        "per_replicate_metrics": stats["per_replicate"],
        "diagnostics": stats["diagnostics"],
        "lineage": _lineage(raw_artifact, analysis_code_digest=analysis_code_digest),
        "analysis_digest": "",
    }
    payload["analysis_digest"] = _analysis_digest(payload)
    errors = validate_exp277_confirmatory_analysis(
        payload,
        raw_artifact=raw_artifact,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if errors:
        raise RuntimeError("invalid EXP-277 confirmatory analysis: " + "; ".join(errors))
    return payload


def validate_exp277_confirmatory_analysis(
    payload: dict[str, Any],
    *,
    raw_artifact: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-277 confirmatory analysis must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-277 confirmatory analysis identity")
    if payload.get("analysis_digest") != _analysis_digest(payload):
        errors.append("EXP-277 confirmatory analysis digest mismatch")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-277 confirmatory analysis challenge-materialization boundary drift")

    raw_errors = validate_exp277_confirmatory_raw(
        raw_artifact,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        errors.append("invalid EXP-277 confirmatory raw artifact: " + "; ".join(raw_errors))
        return errors
    if raw_artifact.get("status") == "INVALID_RUN" or raw_artifact.get("decision") == "INVALID_RUN":
        errors.append("EXP-277 INVALID_RUN raw artifact cannot be analyzed")
        return errors

    raw_test_only = raw_artifact.get("test_only") is True
    if payload.get("test_only") is not raw_test_only:
        errors.append("EXP-277 analysis/raw TEST-ONLY classification mismatch")
    if raw_test_only:
        if payload.get("status") != TEST_ONLY_STATUS:
            errors.append("EXP-277 TEST-ONLY analysis status drift")
        if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
            errors.append("EXP-277 TEST-ONLY analysis cannot promote scientific evidence")
        if payload.get("scientific_evidence_eligible") is not False:
            errors.append("EXP-277 TEST-ONLY analysis cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-277 TEST-ONLY analysis cannot consume confirmatory data")
        if payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("EXP-277 TEST-ONLY analysis synthetic-consumption flag drift")
        if payload.get("decision_rule_executed") is not False:
            errors.append("EXP-277 TEST-ONLY analysis cannot execute scientific decision rule")
        if payload.get("test_only_decision_executed") is not True:
            errors.append("EXP-277 TEST-ONLY diagnostic decision flag drift")
    else:
        if payload.get("status") != SCIENTIFIC_STATUS:
            errors.append("EXP-277 scientific analysis status drift")
        if payload.get("evidence_level") != "EV-E3":
            errors.append("EXP-277 scientific confirmatory analysis must be EV-E3")
        if payload.get("decision") not in VALID_SCIENTIFIC_DECISIONS:
            errors.append("EXP-277 scientific confirmatory decision invalid")
        if payload.get("scientific_evidence_eligible") is not True:
            errors.append("EXP-277 scientific analysis eligibility missing")
        if payload.get("confirmatory_data_consumed") is not True:
            errors.append("EXP-277 scientific analysis confirmatory-consumption flag missing")
        if payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("EXP-277 scientific analysis cannot claim synthetic consumption")
        if payload.get("decision_rule_executed") is not True:
            errors.append("EXP-277 scientific analysis must record decision-rule execution")
        if payload.get("test_only_decision_executed") is not False:
            errors.append("EXP-277 scientific analysis TEST-ONLY flag drift")
        if payload.get("test_only_would_be_decision") is not None:
            errors.append("EXP-277 scientific analysis cannot expose TEST-ONLY decision")

    try:
        expected_code, _ = _sealed_analysis_identity(raw_artifact)
        expected_stats = _expected_statistics(raw_artifact)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"EXP-277 confirmatory analysis reconstruction failed: {exc}")
        return errors

    if payload.get("frozen_contract") != _frozen_contract():
        errors.append("EXP-277 frozen analysis contract drift")
    if payload.get("confirmatory_n") != raw_artifact.get("confirmatory_n"):
        errors.append("EXP-277 confirmatory analysis n mismatch")
    if payload.get("reserved_replicate_ids") != raw_artifact.get("reserved_replicate_ids"):
        errors.append("EXP-277 confirmatory analysis reserved replicate mismatch")
    if payload.get("primary_endpoint") != expected_stats["primary"]:
        errors.append("EXP-277 confirmatory primary ratio-of-means reconstruction mismatch")
    if payload.get("protected_solution_rate") != expected_stats["solution"]:
        errors.append("EXP-277 protected solution-rate reconstruction mismatch")
    if payload.get("wall_energy_per_episode") != WALL_ENERGY_REPORT:
        errors.append("EXP-277 wall-energy report-only boundary drift")
    if payload.get("per_replicate_metrics") != expected_stats["per_replicate"]:
        errors.append("EXP-277 confirmatory per-replicate reconstruction mismatch")
    if payload.get("diagnostics") != expected_stats["diagnostics"]:
        errors.append("EXP-277 confirmatory diagnostics reconstruction mismatch")

    expected_decision = expected_stats["would_be_decision"]
    if raw_test_only:
        if payload.get("test_only_would_be_decision") != expected_decision:
            errors.append("EXP-277 TEST-ONLY would-be decision mismatch")
    elif payload.get("decision") != expected_decision:
        errors.append("EXP-277 scientific decision does not match frozen rule")

    expected_lineage = _lineage(raw_artifact, analysis_code_digest=expected_code)
    if payload.get("lineage") != expected_lineage:
        errors.append("EXP-277 confirmatory analysis lineage mismatch")
    return errors
