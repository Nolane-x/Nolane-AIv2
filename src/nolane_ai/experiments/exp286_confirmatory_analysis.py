from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import random
from statistics import fmean
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import require_frozen_stage_a_v1_sha256
from .exp286_confirmatory_ceremony import validate_exp286_confirmatory_gate_a_seal
from .exp286_confirmatory_executor import validate_exp286_confirmatory_raw

SCHEMA = "NLM-EXP-286-CONFIRMATORY-ANALYSIS-V1"
EXPERIMENT_ID = "EXP-286"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_ANALYZED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_ANALYZED"
PRIMARY_ENDPOINT = "accounted_reasoning_flops_to_verified_solution"
PRIMARY_EFFECT_TYPE = "paired_log_cost_ratio_relative_reduction"
MESI_RELATIVE_REDUCTION = 0.15
ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10_000
SOLUTION_FLOOR_DIFFERENCE = -0.005
VALID_DECISIONS = {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}


def _analysis_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("analysis_digest", None)
    return canonical_sha256(clean)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _quantile_indices(count: int, alpha: float) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("EXP-286 bootstrap requires at least one resample")
    return (
        max(0, math.ceil(alpha * count) - 1),
        min(count - 1, math.ceil((1.0 - alpha) * count) - 1),
    )


def bootstrap_exp286_log_cost_reduction(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    if not baseline_values or len(baseline_values) != len(candidate_values):
        raise ValueError("EXP-286 primary bootstrap requires equal non-empty paired costs")
    if any(not _finite(value) or float(value) <= 0.0 for value in baseline_values + candidate_values):
        raise ValueError("EXP-286 primary costs must be finite and strictly positive")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-286 bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-286 bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-286 bootstrap alpha must be in (0, 0.5)")

    log_ratios = [
        math.log(float(candidate) / float(baseline))
        for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
    ]
    observed = 1.0 - math.exp(float(fmean(log_ratios)))
    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(log_ratios)
    reductions = [
        1.0 - math.exp(float(fmean(log_ratios[rng.randrange(n)] for _ in range(n))))
        for _ in range(samples)
    ]
    reductions.sort()
    lower_index, upper_index = _quantile_indices(samples, float(alpha))
    return {
        "observed_relative_reduction": float(observed),
        "one_sided_lower": float(reductions[lower_index]),
        "one_sided_upper": float(reductions[upper_index]),
        "samples": int(samples),
        "alpha": float(alpha),
        "paired_n": int(n),
        "seed_digest": seed_digest,
    }


def _bootstrap_solution_difference(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    if not baseline_values or len(baseline_values) != len(candidate_values):
        raise ValueError("EXP-286 protected bootstrap requires equal non-empty paired values")
    if any(not _finite(value) for value in baseline_values + candidate_values):
        raise ValueError("EXP-286 protected solution values must be finite")
    paired = [
        float(candidate) - float(baseline)
        for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
    ]
    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(paired)
    values = [
        float(fmean(paired[rng.randrange(n)] for _ in range(n)))
        for _ in range(samples)
    ]
    values.sort()
    lower_index, upper_index = _quantile_indices(samples, float(alpha))
    return {
        "observed_difference": float(fmean(paired)),
        "one_sided_lower": float(values[lower_index]),
        "one_sided_upper": float(values[upper_index]),
        "samples": int(samples),
        "alpha": float(alpha),
        "paired_n": int(n),
        "seed_digest": seed_digest,
        "floor_difference": SOLUTION_FLOOR_DIFFERENCE,
    }


def decide_exp286_confirmatory_outcome(
    *,
    primary_lower: float | None,
    primary_upper: float | None,
    solution_lower: float | None,
    solution_upper: float | None,
) -> str:
    for label, value in (
        ("primary_lower", primary_lower),
        ("primary_upper", primary_upper),
        ("solution_lower", solution_lower),
        ("solution_upper", solution_upper),
    ):
        if value is not None and not _finite(value):
            raise ValueError(f"EXP-286 decision {label} must be finite or None")
    if primary_upper is not None and float(primary_upper) < MESI_RELATIVE_REDUCTION:
        return "KILL_SUBSYSTEM"
    if solution_upper is not None and float(solution_upper) < SOLUTION_FLOOR_DIFFERENCE:
        return "KILL_SUBSYSTEM"
    if (
        primary_lower is not None
        and solution_lower is not None
        and float(primary_lower) >= MESI_RELATIVE_REDUCTION
        and float(solution_lower) >= SOLUTION_FLOOR_DIFFERENCE
    ):
        return "PROMOTE_TO_NEXT_STAGE"
    return "HOLD_UNSTABLE"


def _raw_mode(raw_artifact: dict[str, Any]) -> bool:
    test_only = raw_artifact.get("test_only")
    scientific = raw_artifact.get("scientific_evidence_eligible")
    if test_only is True and scientific is False:
        return True
    if test_only is False and scientific is True:
        if raw_artifact.get("confirmatory_data_consumed") is not True:
            raise ValueError("EXP-286 scientific raw artifact must consume confirmatory data")
        return False
    raise ValueError("EXP-286 raw artifact evidence mode is invalid")


def _compute_inferences(raw_artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    rows = raw_artifact["per_replicate"]
    baseline_cost = [float(row["chronological_failure"][PRIMARY_ENDPOINT]) for row in rows]
    candidate_cost = [float(row["oracle_conflict_core"][PRIMARY_ENDPOINT]) for row in rows]
    primary = bootstrap_exp286_log_cost_reduction(
        baseline_cost,
        candidate_cost,
        seed_material=f"{raw_artifact['artifact_digest']}|EXP-286|primary-log-cost",
    )
    baseline_solution = [float(row["chronological_failure"]["verified_solution_rate"]) for row in rows]
    candidate_solution = [float(row["oracle_conflict_core"]["verified_solution_rate"]) for row in rows]
    solution = _bootstrap_solution_difference(
        baseline_solution,
        candidate_solution,
        seed_material=f"{raw_artifact['artifact_digest']}|EXP-286|solution-rate",
    )
    decision = decide_exp286_confirmatory_outcome(
        primary_lower=primary["one_sided_lower"],
        primary_upper=primary["one_sided_upper"],
        solution_lower=solution["one_sided_lower"],
        solution_upper=solution["one_sided_upper"],
    )
    return primary, solution, decision


def build_exp286_confirmatory_analysis(
    *,
    raw_artifact: dict[str, Any],
    seal: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    raw_errors = validate_exp286_confirmatory_raw(raw_artifact)
    if raw_errors:
        raise ValueError("invalid EXP-286 raw evidence: " + "; ".join(raw_errors))
    seal_errors = validate_exp286_confirmatory_gate_a_seal(seal)
    if seal_errors:
        raise ValueError("invalid EXP-286 Gate-A seal: " + "; ".join(seal_errors))
    if raw_artifact.get("seal_digest") != seal.get("seal_digest"):
        raise ValueError("EXP-286 analysis/raw Gate-A seal lineage mismatch")
    if analysis_code_digest != seal.get("code_tree_digest"):
        raise ValueError("EXP-286 analysis code digest does not match sealed code tree")
    test_only = _raw_mode(raw_artifact)
    primary, solution, inferred_decision = _compute_inferences(raw_artifact)
    lineage = seal.get("lineage") or {}
    require_frozen_stage_a_v1_sha256(lineage.get("protocol_digest"))

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": not test_only,
        "evidence_level": "EV-E2" if test_only else "EV-E3",
        "decision": "UNVERIFIED" if test_only else inferred_decision,
        "confirmatory_data_consumed": False if test_only else True,
        "synthetic_challenge_data_consumed": True if test_only else False,
        "decision_rule_executed": False if test_only else True,
        "test_only_decision_executed": True if test_only else False,
        "test_only_would_be_decision": inferred_decision if test_only else None,
        "confirmatory_n": raw_artifact.get("confirmatory_n"),
        "raw_artifact_digest": raw_artifact.get("artifact_digest"),
        "seal_digest": seal.get("seal_digest"),
        "analysis_code_digest": analysis_code_digest,
        "primary_endpoint": {
            "metric": PRIMARY_ENDPOINT,
            "direction": "lower",
            "effect_type": PRIMARY_EFFECT_TYPE,
            "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
            "alpha": ALPHA,
            "samples": BOOTSTRAP_SAMPLES,
            "inference": primary,
        },
        "protected_solution_rate": solution,
        "lineage": {
            "protocol_digest": lineage.get("protocol_digest"),
            "gate_a_seal_digest": seal.get("seal_digest"),
            "raw_artifact_digest": raw_artifact.get("artifact_digest"),
            "beacon_receipt_digest": (raw_artifact.get("beacon_receipt") or {}).get("receipt_digest"),
            "analysis_code_digest": analysis_code_digest,
        },
        "analysis_digest": "",
    }
    payload["analysis_digest"] = _analysis_digest(payload)
    errors = validate_exp286_confirmatory_analysis(payload, raw_artifact=raw_artifact, seal=seal)
    if errors:
        raise RuntimeError("invalid EXP-286 Gate-B analysis: " + "; ".join(errors))
    return payload


def validate_exp286_confirmatory_analysis(
    payload: dict[str, Any],
    *,
    raw_artifact: dict[str, Any],
    seal: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-286 confirmatory analysis must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-286 Gate-B analysis identity")
    try:
        test_only = _raw_mode(raw_artifact)
    except ValueError as exc:
        errors.append(str(exc))
        test_only = None
    if test_only is True:
        expected = {
            "status": TEST_ONLY_STATUS,
            "scientific_evidence_eligible": False,
            "evidence_level": "EV-E2",
            "decision": "UNVERIFIED",
            "confirmatory_data_consumed": False,
            "synthetic_challenge_data_consumed": True,
            "decision_rule_executed": False,
            "test_only_decision_executed": True,
        }
    elif test_only is False:
        expected = {
            "status": SCIENTIFIC_STATUS,
            "scientific_evidence_eligible": True,
            "evidence_level": "EV-E3",
            "confirmatory_data_consumed": True,
            "synthetic_challenge_data_consumed": False,
            "decision_rule_executed": True,
            "test_only_decision_executed": False,
        }
    else:
        expected = {}
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"EXP-286 analysis evidence boundary drift: {key}")
    if test_only is False and payload.get("decision") not in VALID_DECISIONS:
        errors.append("EXP-286 scientific Gate-B decision is invalid")
    if payload.get("raw_artifact_digest") != raw_artifact.get("artifact_digest"):
        errors.append("EXP-286 analysis/raw lineage mismatch")
    if payload.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-286 analysis/seal lineage mismatch")
    if payload.get("confirmatory_n") != raw_artifact.get("confirmatory_n"):
        errors.append("EXP-286 analysis confirmatory_n mismatch")
    if payload.get("analysis_code_digest") != seal.get("code_tree_digest"):
        errors.append("EXP-286 analysis code-tree closure mismatch")

    primary = payload.get("primary_endpoint") or {}
    frozen_primary = {
        "metric": PRIMARY_ENDPOINT,
        "direction": "lower",
        "effect_type": PRIMARY_EFFECT_TYPE,
        "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
        "alpha": ALPHA,
        "samples": BOOTSTRAP_SAMPLES,
    }
    for key, value in frozen_primary.items():
        if primary.get(key) != value:
            errors.append(f"EXP-286 frozen primary analysis drift: {key}")
    solution = payload.get("protected_solution_rate") or {}
    if solution.get("floor_difference") != SOLUTION_FLOOR_DIFFERENCE:
        errors.append("EXP-286 protected solution floor drift")
    try:
        require_frozen_stage_a_v1_sha256((payload.get("lineage") or {}).get("protocol_digest"))
    except ValueError as exc:
        errors.append(str(exc))

    if not errors:
        try:
            expected_primary, expected_solution, expected_decision = _compute_inferences(raw_artifact)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"EXP-286 analysis semantic reconstruction failed: {exc}")
        else:
            if primary.get("inference") != expected_primary:
                errors.append("EXP-286 primary analysis semantic reconstruction mismatch")
            if solution != expected_solution:
                errors.append("EXP-286 protected analysis semantic reconstruction mismatch")
            if test_only is True:
                if payload.get("decision") != "UNVERIFIED" or payload.get("test_only_would_be_decision") != expected_decision:
                    errors.append("EXP-286 TEST-ONLY decision firewall mismatch")
            elif test_only is False:
                if payload.get("decision") != expected_decision or payload.get("test_only_would_be_decision") is not None:
                    errors.append("EXP-286 scientific decision semantic reconstruction mismatch")
    if payload.get("analysis_digest") != _analysis_digest(payload):
        errors.append("EXP-286 analysis digest mismatch")
    return errors
