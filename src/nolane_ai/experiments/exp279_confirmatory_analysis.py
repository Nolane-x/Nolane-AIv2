from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import random
from statistics import fmean, median
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp279_confirmatory_executor import validate_exp279_confirmatory_raw
from .exp279_confirmatory_prep import (
    BOOTSTRAP_SAMPLES,
    DENOMINATOR_POLICY,
    EFFECT_TYPE,
    FAMILYWISE_ALPHA,
    HOLM_THRESHOLDS,
    MESI,
    PLANNING_ALPHA,
    PRIMARY_CONTRASTS,
    _frozen_analysis,
)
from .exp279_paired_runner import (
    MULTIPLICITY_FAMILY,
    PRIMARY_METRIC,
    PROTECTED_FLOOR,
)
from .exp279_routing_worlds import STRATA


SCHEMA = "NLM-EXP-279-CONFIRMATORY-ANALYSIS-V1"
EXPERIMENT_ID = "EXP-279"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_ANALYZED"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_ANALYZED"
VALID_SCIENTIFIC_DECISIONS = {
    "PROMOTE_TO_NEXT_STAGE",
    "HOLD_UNSTABLE",
    "KILL_SUBSYSTEM",
}
SOLUTION_FLOOR_DIFFERENCE = -0.01
BLOCK_WEIGHTING = "equal_weight_across_predeclared_structure_fit_strata"
PRIMARY_BOOTSTRAP_STREAM = "blocked-mesi-bootstrap-v1"
SOLUTION_BOOTSTRAP_STREAM = "blocked-solution-bootstrap-v1"
WALL_ENERGY_REPORT = {
    "stage_a_role": "report-only",
    "numeric_measurement": None,
    "fabricated": False,
    "decision_use": False,
    "note": "not measured by EXP-279 confirmatory executor; no numeric value fabricated",
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
        raise ValueError("EXP-279 bootstrap requires at least one valid resample")
    lower = max(0, math.ceil(float(alpha) * count) - 1)
    upper = min(count - 1, math.ceil((1.0 - float(alpha)) * count) - 1)
    return lower, upper


def _validate_blocked_pair(
    simple_by_stratum: dict[str, list[float]],
    hybrid_by_stratum: dict[str, list[float]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    if not isinstance(simple_by_stratum, dict) or not isinstance(
        hybrid_by_stratum, dict
    ):
        raise ValueError("EXP-279 blocked bootstrap requires stratum mappings")
    if list(simple_by_stratum) != list(STRATA) or list(hybrid_by_stratum) != list(
        STRATA
    ):
        raise ValueError("EXP-279 blocked bootstrap stratum order drift")

    simple: dict[str, list[float]] = {}
    hybrid: dict[str, list[float]] = {}
    for stratum in STRATA:
        simple_values = simple_by_stratum.get(stratum)
        hybrid_values = hybrid_by_stratum.get(stratum)
        if (
            not isinstance(simple_values, list)
            or not isinstance(hybrid_values, list)
            or not simple_values
            or len(simple_values) != len(hybrid_values)
        ):
            raise ValueError(
                "EXP-279 blocked bootstrap requires equal non-empty paired values "
                f"within {stratum}"
            )
        if any(
            not _finite(value) for value in list(simple_values) + list(hybrid_values)
        ):
            raise ValueError("EXP-279 blocked bootstrap requires finite paired values")
        simple[stratum] = [float(value) for value in simple_values]
        hybrid[stratum] = [float(value) for value in hybrid_values]
    return simple, hybrid


def _blocked_means(
    simple: dict[str, list[float]],
    hybrid: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, float], float, float]:
    simple_means = {
        stratum: float(fmean(simple[stratum]))
        for stratum in STRATA
    }
    hybrid_means = {
        stratum: float(fmean(hybrid[stratum]))
        for stratum in STRATA
    }
    return (
        simple_means,
        hybrid_means,
        float(fmean(simple_means[stratum] for stratum in STRATA)),
        float(fmean(hybrid_means[stratum] for stratum in STRATA)),
    )


def bootstrap_exp279_blocked_contrast(
    simple_by_stratum: dict[str, list[float]],
    hybrid_by_stratum: dict[str, list[float]],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = PLANNING_ALPHA,
    mesi: float = MESI,
) -> dict[str, Any]:
    simple, hybrid = _validate_blocked_pair(simple_by_stratum, hybrid_by_stratum)
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-279 blocked bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-279 blocked bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-279 blocked bootstrap alpha must be in (0, 0.5)")
    if not _finite(mesi) or float(mesi) < 0.0:
        raise ValueError("EXP-279 blocked bootstrap MESI must be finite and non-negative")

    simple_means, hybrid_means, observed_simple, observed_hybrid = _blocked_means(
        simple,
        hybrid,
    )
    observed_valid = (
        math.isfinite(observed_simple)
        and observed_simple > 0.0
        and math.isfinite(observed_hybrid)
    )
    observed_gain = (
        observed_hybrid / observed_simple - 1.0 if observed_valid else None
    )
    observed_excess = (
        observed_hybrid - (1.0 + float(mesi)) * observed_simple
        if observed_valid
        else None
    )

    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    base_result: dict[str, Any] = {
        "strata": list(STRATA),
        "block_weighting": BLOCK_WEIGHTING,
        "simple_stratum_means": simple_means,
        "hybrid_stratum_means": hybrid_means,
        "observed_simple_blocked_mean": observed_simple,
        "observed_hybrid_blocked_mean": observed_hybrid,
        "observed_relative_gain": observed_gain,
        "observed_mesi_excess": observed_excess,
        "mesi_relative_gain": float(mesi),
        "null_hypothesis": f"blocked_relative_gain <= {float(mesi):.2f}",
        "one_sided_p_value": None,
        "one_sided_lower_relative_gain": None,
        "one_sided_upper_relative_gain": None,
        "bootstrap_median_relative_gain": None,
        "samples": samples,
        "valid_resamples": 0,
        "invalid_denominator_resamples": samples,
        "alpha": float(alpha),
        "seed_digest": seed_digest,
        "denominator_policy": DENOMINATOR_POLICY,
        "denominator_valid": False,
    }
    if not observed_valid:
        return base_result

    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    gains: list[float] = []
    excesses: list[float] = []
    invalid_denominators = 0
    for _ in range(samples):
        simple_stratum_means: list[float] = []
        hybrid_stratum_means: list[float] = []
        for stratum in STRATA:
            n = len(simple[stratum])
            indices = [rng.randrange(n) for _ in range(n)]
            simple_stratum_means.append(
                float(fmean(simple[stratum][index] for index in indices))
            )
            hybrid_stratum_means.append(
                float(fmean(hybrid[stratum][index] for index in indices))
            )
        simple_blocked = float(fmean(simple_stratum_means))
        hybrid_blocked = float(fmean(hybrid_stratum_means))
        if (
            not math.isfinite(simple_blocked)
            or simple_blocked <= 0.0
            or not math.isfinite(hybrid_blocked)
        ):
            invalid_denominators += 1
            continue
        gain = hybrid_blocked / simple_blocked - 1.0
        excess = hybrid_blocked - (1.0 + float(mesi)) * simple_blocked
        if not math.isfinite(gain) or not math.isfinite(excess):
            invalid_denominators += 1
            continue
        gains.append(float(gain))
        excesses.append(float(excess))

    gains.sort()
    if gains:
        lower_index, upper_index = _quantile_indices(len(gains), float(alpha))
        lower: float | None = float(gains[lower_index])
        upper: float | None = float(gains[upper_index])
        bootstrap_median: float | None = float(median(gains))
        centered_null = [value - float(observed_excess) for value in excesses]
        exceedances = sum(
            value >= float(observed_excess) for value in centered_null
        )
        p_value: float | None = (1.0 + exceedances) / (len(excesses) + 1.0)
    else:
        lower = upper = bootstrap_median = p_value = None

    denominator_valid = bool(
        invalid_denominators == 0 and len(gains) == samples
    )
    base_result.update(
        {
            "one_sided_p_value": p_value if denominator_valid else None,
            "one_sided_lower_relative_gain": lower if denominator_valid else None,
            "one_sided_upper_relative_gain": upper if denominator_valid else None,
            "bootstrap_median_relative_gain": (
                bootstrap_median if denominator_valid else None
            ),
            "valid_resamples": len(gains),
            "invalid_denominator_resamples": invalid_denominators,
            "denominator_valid": denominator_valid,
        }
    )
    return base_result


def _bootstrap_exp279_blocked_difference(
    simple_by_stratum: dict[str, list[float]],
    hybrid_by_stratum: dict[str, list[float]],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = PLANNING_ALPHA,
) -> dict[str, Any]:
    simple, hybrid = _validate_blocked_pair(simple_by_stratum, hybrid_by_stratum)
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-279 protected bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-279 protected bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-279 protected bootstrap alpha must be in (0, 0.5)")

    simple_means, hybrid_means, observed_simple, observed_hybrid = _blocked_means(
        simple,
        hybrid,
    )
    observed = observed_hybrid - observed_simple
    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    differences: list[float] = []
    for _ in range(samples):
        stratum_differences: list[float] = []
        for stratum in STRATA:
            n = len(simple[stratum])
            indices = [rng.randrange(n) for _ in range(n)]
            simple_mean = float(fmean(simple[stratum][index] for index in indices))
            hybrid_mean = float(fmean(hybrid[stratum][index] for index in indices))
            stratum_differences.append(hybrid_mean - simple_mean)
        differences.append(float(fmean(stratum_differences)))
    differences.sort()
    lower_index, upper_index = _quantile_indices(samples, float(alpha))
    return {
        "strata": list(STRATA),
        "block_weighting": BLOCK_WEIGHTING,
        "simple_stratum_means": simple_means,
        "hybrid_stratum_means": hybrid_means,
        "observed_simple_blocked_mean": observed_simple,
        "observed_hybrid_blocked_mean": observed_hybrid,
        "observed_difference": observed,
        "one_sided_lower_difference": float(differences[lower_index]),
        "one_sided_upper_difference": float(differences[upper_index]),
        "bootstrap_median_difference": float(median(differences)),
        "samples": samples,
        "alpha": float(alpha),
        "seed_digest": seed_digest,
    }


def holm_exp279_family(
    p_values: dict[str, float | None],
) -> dict[str, Any]:
    if not isinstance(p_values, dict) or set(p_values) != set(PRIMARY_CONTRASTS):
        raise ValueError("EXP-279 Holm family requires both frozen primary contrasts")
    canonical_order = {name: index for index, name in enumerate(PRIMARY_CONTRASTS)}
    normalized: dict[str, float | None] = {}
    for contrast in PRIMARY_CONTRASTS:
        value = p_values.get(contrast)
        if value is None:
            normalized[contrast] = None
        elif not _finite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("EXP-279 Holm p-values must be within [0, 1] or None")
        else:
            normalized[contrast] = float(value)

    ordered = sorted(
        PRIMARY_CONTRASTS,
        key=lambda name: (
            1.0 if normalized[name] is None else float(normalized[name]),
            canonical_order[name],
        ),
    )
    results: dict[str, dict[str, Any]] = {}
    active = True
    running_adjusted = 0.0
    total = len(PRIMARY_CONTRASTS)
    for rank, contrast in enumerate(ordered, start=1):
        raw_p = normalized[contrast]
        threshold = float(HOLM_THRESHOLDS[rank - 1])
        rejected = bool(active and raw_p is not None and raw_p <= threshold)
        if not rejected:
            active = False
        if raw_p is None:
            adjusted = None
        else:
            running_adjusted = max(
                running_adjusted,
                min(1.0, (total - rank + 1) * float(raw_p)),
            )
            adjusted = running_adjusted
        results[contrast] = {
            "rank": rank,
            "raw_p_value": raw_p,
            "threshold": threshold,
            "holm_adjusted_p_value": adjusted,
            "rejected": rejected,
            "estimable": raw_p is not None,
        }
    return {
        "family": MULTIPLICITY_FAMILY,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "step_down_thresholds": list(HOLM_THRESHOLDS),
        "ordered_contrasts": list(ordered),
        "results": results,
    }


def decide_exp279_confirmatory_outcome(
    *,
    selected_holm_rejected: bool,
    primary_lower: float | None,
    primary_upper: float | None,
    protected_lower: float | None,
    protected_upper: float | None,
    denominator_valid: bool,
) -> str:
    if not isinstance(selected_holm_rejected, bool):
        raise ValueError("EXP-279 Holm rejection flag must be boolean")
    if not isinstance(denominator_valid, bool):
        raise ValueError("EXP-279 denominator validity flag must be boolean")
    for label, value in (
        ("primary_lower", primary_lower),
        ("primary_upper", primary_upper),
        ("protected_lower", protected_lower),
        ("protected_upper", protected_upper),
    ):
        if value is not None and not _finite(value):
            raise ValueError(f"EXP-279 decision {label} must be finite or None")

    # No epsilon or denominator repair. If the frozen ratio estimand cannot be
    # formed, the primary claim is unresolved rather than silently converted
    # into a kill or promotion claim.
    if denominator_valid is not True:
        return "HOLD_UNSTABLE"

    if primary_upper is not None and float(primary_upper) < MESI:
        return "KILL_SUBSYSTEM"
    if (
        protected_upper is not None
        and float(protected_upper) < SOLUTION_FLOOR_DIFFERENCE
    ):
        return "KILL_SUBSYSTEM"

    if (
        selected_holm_rejected is True
        and primary_lower is not None
        and protected_lower is not None
        and float(primary_lower) >= MESI
        and float(protected_lower) >= SOLUTION_FLOOR_DIFFERENCE
    ):
        return "PROMOTE_TO_NEXT_STAGE"
    return "HOLD_UNSTABLE"


def _sealed_analysis_identity(raw: dict[str, Any]) -> tuple[str, str, str]:
    seal = raw.get("seal") or {}
    authorization = seal.get("authorization_snapshot") or {}
    analysis_code_digest = authorization.get("analysis_code_digest")
    frozen_analysis_digest = authorization.get("frozen_analysis_digest")
    protocol_digest = authorization.get("protocol_digest")
    if not isinstance(analysis_code_digest, str) or len(analysis_code_digest) != 64:
        raise ValueError("EXP-279 pre-beacon frozen analysis code identity missing")
    if not isinstance(frozen_analysis_digest, str) or len(frozen_analysis_digest) != 64:
        raise ValueError("EXP-279 pre-beacon frozen analysis contract digest missing")
    if not isinstance(protocol_digest, str) or len(protocol_digest) != 64:
        raise ValueError("EXP-279 pre-beacon frozen protocol identity missing")
    if frozen_analysis_digest != canonical_sha256(_frozen_analysis()):
        raise ValueError("EXP-279 pre-beacon frozen analysis contract digest drift")
    if PROTECTED_FLOOR != "hybrid >= best_simple - 0.01":
        raise ValueError("EXP-279 protected solution-rate contract drift")
    return analysis_code_digest, frozen_analysis_digest, protocol_digest


def _by_stratum(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    metric: str,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {stratum: [] for stratum in STRATA}
    for row in rows:
        stratum = row.get("stratum")
        if stratum not in result:
            raise ValueError("EXP-279 analysis raw stratum drift")
        value = (row.get(arm) or {}).get(metric)
        if not _finite(value):
            raise ValueError(f"EXP-279 analysis raw {arm} {metric} is non-finite")
        result[stratum].append(float(value))
    if any(not result[stratum] for stratum in STRATA):
        raise ValueError("EXP-279 analysis requires every frozen stratum")
    return result


def _expected_statistics(raw: dict[str, Any]) -> dict[str, Any]:
    _, frozen_analysis_digest, protocol_digest = _sealed_analysis_identity(raw)
    rows = raw.get("per_replicate")
    if not isinstance(rows, list) or not rows:
        raise ValueError("EXP-279 analysis requires non-empty raw replicate rows")

    hybrid_utility = _by_stratum(rows, arm="hybrid", metric=PRIMARY_METRIC)
    contrast_to_arm = {
        "hybrid_vs_propagation_only": "propagation_only",
        "hybrid_vs_branch_only": "branch_only",
    }
    contrasts: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, float | None] = {}
    for contrast in PRIMARY_CONTRASTS:
        simple_arm = contrast_to_arm[contrast]
        simple_utility = _by_stratum(rows, arm=simple_arm, metric=PRIMARY_METRIC)
        seed_material = (
            f"{protocol_digest}|{frozen_analysis_digest}|{EXPERIMENT_ID}|"
            f"{contrast}|{PRIMARY_BOOTSTRAP_STREAM}"
        )
        item = bootstrap_exp279_blocked_contrast(
            simple_utility,
            hybrid_utility,
            seed_material=seed_material,
            samples=BOOTSTRAP_SAMPLES,
            alpha=PLANNING_ALPHA,
            mesi=MESI,
        )
        item.update(
            {
                "contrast": contrast,
                "simple_arm": simple_arm,
                "candidate_arm": "hybrid",
                "metric": PRIMARY_METRIC,
                "effect_type": EFFECT_TYPE,
            }
        )
        contrasts[contrast] = item
        raw_p_values[contrast] = item["one_sided_p_value"]

    holm = holm_exp279_family(raw_p_values)
    for contrast in PRIMARY_CONTRASTS:
        family_item = holm["results"][contrast]
        contrasts[contrast].update(
            {
                "holm_rank": family_item["rank"],
                "holm_threshold": family_item["threshold"],
                "holm_adjusted_p_value": family_item["holm_adjusted_p_value"],
                "holm_rejected": family_item["rejected"],
            }
        )

    propagation_mean = float(
        contrasts["hybrid_vs_propagation_only"]["observed_simple_blocked_mean"]
    )
    branch_mean = float(
        contrasts["hybrid_vs_branch_only"]["observed_simple_blocked_mean"]
    )
    selected_arm = (
        "propagation_only" if propagation_mean >= branch_mean else "branch_only"
    )
    selected_contrast = (
        "hybrid_vs_propagation_only"
        if selected_arm == "propagation_only"
        else "hybrid_vs_branch_only"
    )
    selection = {
        "rule": (
            "higher equal-weight blocked mean primary utility; exact tie -> propagation_only"
        ),
        "propagation_only_blocked_mean_utility": propagation_mean,
        "branch_only_blocked_mean_utility": branch_mean,
        "selected_arm": selected_arm,
        "selected_contrast": selected_contrast,
        "selection_applied_after_both_contrasts_and_holm": True,
    }

    hybrid_solution = _by_stratum(
        rows,
        arm="hybrid",
        metric="verified_solution_rate",
    )
    selected_solution = _by_stratum(
        rows,
        arm=selected_arm,
        metric="verified_solution_rate",
    )
    protected_seed = (
        f"{protocol_digest}|{frozen_analysis_digest}|{EXPERIMENT_ID}|"
        f"hybrid_vs_{selected_arm}|{SOLUTION_BOOTSTRAP_STREAM}"
    )
    protected = _bootstrap_exp279_blocked_difference(
        selected_solution,
        hybrid_solution,
        seed_material=protected_seed,
        samples=BOOTSTRAP_SAMPLES,
        alpha=PLANNING_ALPHA,
    )
    protected.update(
        {
            "metric": "verified_solution_rate",
            "simple_arm": selected_arm,
            "candidate_arm": "hybrid",
            "floor_difference": SOLUTION_FLOOR_DIFFERENCE,
        }
    )

    selected = contrasts[selected_contrast]
    would_be_decision = decide_exp279_confirmatory_outcome(
        selected_holm_rejected=bool(selected["holm_rejected"]),
        primary_lower=selected["one_sided_lower_relative_gain"],
        primary_upper=selected["one_sided_upper_relative_gain"],
        protected_lower=protected["one_sided_lower_difference"],
        protected_upper=protected["one_sided_upper_difference"],
        denominator_valid=bool(selected["denominator_valid"]),
    )
    return {
        "contrasts": contrasts,
        "holm_family": holm,
        "best_simple_selection": selection,
        "protected_solution_rate": protected,
        "would_be_decision": would_be_decision,
    }


def build_exp279_confirmatory_analysis(
    *,
    raw_artifact: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    raw_errors = validate_exp279_confirmatory_raw(
        raw_artifact,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        raise ValueError("invalid EXP-279 raw artifact: " + "; ".join(raw_errors))
    if raw_artifact.get("status") == "INVALID_RUN" or raw_artifact.get(
        "decision"
    ) == "INVALID_RUN":
        raise ValueError("EXP-279 INVALID_RUN raw artifact cannot be analyzed")

    sealed_code_digest, frozen_analysis_digest, protocol_digest = (
        _sealed_analysis_identity(raw_artifact)
    )
    if analysis_code_digest != sealed_code_digest:
        raise ValueError(
            "EXP-279 analysis code identity differs from pre-beacon freeze"
        )

    statistics = _expected_statistics(raw_artifact)
    test_only = raw_artifact.get("test_only") is True
    scientific = bool(
        raw_artifact.get("scientific_evidence_eligible") is True and not test_only
    )
    if test_only:
        status = TEST_ONLY_STATUS
        evidence_level = "EV-E2"
        decision = "UNVERIFIED"
        decision_rule_executed = False
        test_only_decision_executed = True
        test_only_would_be_decision: str | None = statistics["would_be_decision"]
    elif scientific:
        status = SCIENTIFIC_STATUS
        evidence_level = "EV-E3"
        decision = statistics["would_be_decision"]
        decision_rule_executed = True
        test_only_decision_executed = False
        test_only_would_be_decision = None
    else:
        raise ValueError("EXP-279 raw evidence classification is not analyzable")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "evidence_level": evidence_level,
        "decision": decision,
        "test_only": test_only,
        "scientific_evidence_eligible": scientific,
        "confirmatory_data_consumed": bool(
            raw_artifact.get("confirmatory_data_consumed") is True
        ),
        "synthetic_challenge_data_consumed": bool(
            raw_artifact.get("synthetic_challenge_data_consumed") is True
        ),
        "challenge_materialized": True,
        "decision_rule_executed": decision_rule_executed,
        "test_only_decision_executed": test_only_decision_executed,
        "test_only_would_be_decision": test_only_would_be_decision,
        "analysis_code_digest": analysis_code_digest,
        "frozen_analysis_digest": frozen_analysis_digest,
        "protocol_digest": protocol_digest,
        "raw_artifact_digest": raw_artifact.get("artifact_digest"),
        "analysis_contract": deepcopy(_frozen_analysis()),
        "contrasts": statistics["contrasts"],
        "holm_family": statistics["holm_family"],
        "best_simple_selection": statistics["best_simple_selection"],
        "protected_solution_rate": statistics["protected_solution_rate"],
        "wall_energy_per_episode": deepcopy(WALL_ENERGY_REPORT),
        "analysis_digest": "",
    }
    payload["analysis_digest"] = _analysis_digest(payload)
    return payload


def validate_exp279_confirmatory_analysis(
    payload: dict[str, Any],
    *,
    raw_artifact: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-279 analysis artifact must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 confirmatory analysis identity")
    if payload.get("analysis_digest") != _analysis_digest(payload):
        errors.append("EXP-279 analysis artifact digest mismatch")

    raw_errors = validate_exp279_confirmatory_raw(
        raw_artifact,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        errors.append("invalid EXP-279 raw artifact: " + "; ".join(raw_errors))
        return errors
    if raw_artifact.get("status") == "INVALID_RUN":
        errors.append("EXP-279 INVALID_RUN raw artifact cannot have analysis")
        return errors

    try:
        sealed_code_digest, frozen_analysis_digest, protocol_digest = (
            _sealed_analysis_identity(raw_artifact)
        )
        expected_statistics = _expected_statistics(raw_artifact)
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    if payload.get("analysis_code_digest") != sealed_code_digest:
        errors.append("EXP-279 analysis code identity mismatch")
    if payload.get("frozen_analysis_digest") != frozen_analysis_digest:
        errors.append("EXP-279 frozen analysis digest mismatch")
    if payload.get("protocol_digest") != protocol_digest:
        errors.append("EXP-279 analysis protocol lineage mismatch")
    if payload.get("raw_artifact_digest") != raw_artifact.get("artifact_digest"):
        errors.append("EXP-279 analysis/raw artifact digest mismatch")
    if payload.get("analysis_contract") != _frozen_analysis():
        errors.append("EXP-279 frozen analysis contract surface drift")
    if payload.get("contrasts") != expected_statistics["contrasts"]:
        errors.append("EXP-279 analysis contrast statistics drift")
    if payload.get("holm_family") != expected_statistics["holm_family"]:
        errors.append("EXP-279 analysis Holm family drift")
    if payload.get("best_simple_selection") != expected_statistics[
        "best_simple_selection"
    ]:
        errors.append("EXP-279 analysis best-simple selection drift")
    if payload.get("protected_solution_rate") != expected_statistics[
        "protected_solution_rate"
    ]:
        errors.append("EXP-279 analysis protected solution-rate drift")
    if payload.get("wall_energy_per_episode") != WALL_ENERGY_REPORT:
        errors.append("EXP-279 wall-energy nonfabrication boundary drift")

    test_only = raw_artifact.get("test_only") is True
    scientific = bool(
        raw_artifact.get("scientific_evidence_eligible") is True and not test_only
    )
    expected_decision = expected_statistics["would_be_decision"]
    if test_only:
        if payload.get("status") != TEST_ONLY_STATUS:
            errors.append("EXP-279 TEST-ONLY analysis status drift")
        if payload.get("evidence_level") != "EV-E2" or payload.get(
            "decision"
        ) != "UNVERIFIED":
            errors.append("EXP-279 TEST-ONLY analysis cannot promote evidence")
        if payload.get("scientific_evidence_eligible") is not False:
            errors.append("EXP-279 TEST-ONLY analysis cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-279 TEST-ONLY analysis cannot consume confirmatory data")
        if payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("EXP-279 TEST-ONLY synthetic challenge flag missing")
        if payload.get("decision_rule_executed") is not False:
            errors.append("EXP-279 TEST-ONLY cannot execute scientific decision rule")
        if payload.get("test_only_decision_executed") is not True:
            errors.append("EXP-279 TEST-ONLY would-be decision was not evaluated")
        if payload.get("test_only_would_be_decision") != expected_decision:
            errors.append("EXP-279 TEST-ONLY would-be decision drift")
    elif scientific:
        if payload.get("status") != SCIENTIFIC_STATUS:
            errors.append("EXP-279 scientific analysis status drift")
        if payload.get("evidence_level") != "EV-E3":
            errors.append("EXP-279 scientific analysis must be EV-E3")
        if payload.get("decision") != expected_decision or payload.get(
            "decision"
        ) not in VALID_SCIENTIFIC_DECISIONS:
            errors.append("EXP-279 scientific decision drift")
        if payload.get("scientific_evidence_eligible") is not True:
            errors.append("EXP-279 scientific analysis evidence flag missing")
        if payload.get("confirmatory_data_consumed") is not True:
            errors.append("EXP-279 scientific analysis consumption flag missing")
        if payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("EXP-279 scientific analysis cannot mark synthetic data")
        if payload.get("decision_rule_executed") is not True:
            errors.append("EXP-279 scientific decision rule was not executed")
        if payload.get("test_only_decision_executed") is not False or payload.get(
            "test_only_would_be_decision"
        ) is not None:
            errors.append("EXP-279 scientific analysis leaked TEST-ONLY decision surface")
    else:
        errors.append("EXP-279 analysis raw evidence classification invalid")

    if payload.get("test_only") is not test_only:
        errors.append("EXP-279 analysis test_only classification mismatch")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-279 analysis must bind a materialized challenge")
    return errors
