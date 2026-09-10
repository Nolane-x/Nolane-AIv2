from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import random
from statistics import fmean
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import require_frozen_stage_a_v1_sha256
from .exp289_confirmatory_executor import validate_exp289_confirmatory_raw


SCHEMA = "NLM-EXP-289-CONFIRMATORY-ANALYSIS-V1"
EXPERIMENT_ID = "EXP-289"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_ANALYZED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_ANALYZED"
PRIMARY_ENDPOINT = "repeat_dead_end_rate"
PRIMARY_EFFECT_TYPE = "paired_relative_rder_reduction"
MESI_RELATIVE_REDUCTION = 0.25
ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10_000
DENOMINATOR_POLICY = "strictly_positive_no_epsilon"
OVERPRUNE_CEILING = 0.005
SOLUTION_FLOOR_DIFFERENCE = -0.01
VALID_DECISIONS = {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}


def _analysis_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("analysis_digest", None)
    return canonical_sha256(clean)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _quantile_indices(count: int, alpha: float) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("EXP-289 bootstrap requires at least one resample")
    lower = max(0, math.ceil(alpha * count) - 1)
    upper = min(count - 1, math.ceil((1.0 - alpha) * count) - 1)
    return lower, upper


def bootstrap_exp289_relative_rder(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    if (
        not isinstance(baseline_values, list)
        or not isinstance(candidate_values, list)
        or not baseline_values
        or len(baseline_values) != len(candidate_values)
    ):
        raise ValueError("EXP-289 paired RDER bootstrap requires equal non-empty paired values")
    if any(not _finite(value) for value in baseline_values + candidate_values):
        raise ValueError("EXP-289 paired RDER bootstrap requires finite values")
    if any(float(value) <= 0.0 for value in baseline_values):
        raise ValueError("EXP-289 baseline denominator must be strictly positive; epsilon rescue is forbidden")
    if any(float(value) < 0.0 for value in candidate_values):
        raise ValueError("EXP-289 candidate RDER must be non-negative")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-289 bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-289 bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-289 bootstrap alpha must be in (0, 0.5)")

    baseline = [float(value) for value in baseline_values]
    candidate = [float(value) for value in candidate_values]
    paired_effects = [
        (base - cand) / base
        for base, cand in zip(baseline, candidate, strict=True)
    ]
    observed = float(fmean(paired_effects))
    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(paired_effects)
    resampled = [
        float(fmean(paired_effects[rng.randrange(n)] for _ in range(n)))
        for _ in range(samples)
    ]
    resampled.sort()
    lower_index, upper_index = _quantile_indices(samples, float(alpha))
    return {
        "observed_relative_reduction": observed,
        "one_sided_lower": float(resampled[lower_index]),
        "one_sided_upper": float(resampled[upper_index]),
        "samples": int(samples),
        "valid_resamples": int(samples),
        "invalid_denominator_resamples": 0,
        "alpha": float(alpha),
        "seed_digest": seed_digest,
        "denominator_policy": DENOMINATOR_POLICY,
        "denominator_valid": True,
        "paired_n": int(n),
    }


def _bootstrap_paired_difference(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    if (
        not baseline_values
        or len(baseline_values) != len(candidate_values)
        or any(not _finite(value) for value in baseline_values + candidate_values)
    ):
        raise ValueError("EXP-289 protected solution bootstrap requires equal finite paired values")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-289 protected bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-289 protected bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-289 protected bootstrap alpha must be in (0, 0.5)")

    paired = [
        float(candidate) - float(baseline)
        for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
    ]
    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(paired)
    resampled = [
        float(fmean(paired[rng.randrange(n)] for _ in range(n)))
        for _ in range(samples)
    ]
    resampled.sort()
    lower_index, upper_index = _quantile_indices(samples, float(alpha))
    return {
        "observed_difference": float(fmean(paired)),
        "one_sided_lower": float(resampled[lower_index]),
        "one_sided_upper": float(resampled[upper_index]),
        "samples": int(samples),
        "alpha": float(alpha),
        "seed_digest": seed_digest,
        "paired_n": int(n),
    }


def decide_exp289_confirmatory_outcome(
    *,
    primary_lower: float | None,
    primary_upper: float | None,
    overprune_rate: float,
    solution_lower: float | None,
    solution_upper: float | None,
    denominator_valid: bool,
) -> str:
    if not isinstance(denominator_valid, bool):
        raise ValueError("EXP-289 denominator validity must be boolean")
    if not _finite(overprune_rate) or float(overprune_rate) < 0.0:
        raise ValueError("EXP-289 over-prune rate must be finite and non-negative")
    for label, value in (
        ("primary_lower", primary_lower),
        ("primary_upper", primary_upper),
        ("solution_lower", solution_lower),
        ("solution_upper", solution_upper),
    ):
        if value is not None and not _finite(value):
            raise ValueError(f"EXP-289 decision {label} must be finite or None")

    if primary_upper is not None and float(primary_upper) < MESI_RELATIVE_REDUCTION:
        return "KILL_SUBSYSTEM"
    if float(overprune_rate) > OVERPRUNE_CEILING:
        return "KILL_SUBSYSTEM"
    if solution_upper is not None and float(solution_upper) < SOLUTION_FLOOR_DIFFERENCE:
        return "KILL_SUBSYSTEM"
    if denominator_valid is not True:
        return "HOLD_UNSTABLE"
    if (
        primary_lower is not None
        and solution_lower is not None
        and float(primary_lower) >= MESI_RELATIVE_REDUCTION
        and float(overprune_rate) <= OVERPRUNE_CEILING
        and float(solution_lower) >= SOLUTION_FLOOR_DIFFERENCE
    ):
        return "PROMOTE_TO_NEXT_STAGE"
    return "HOLD_UNSTABLE"


def _compute_inferences(raw_artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    rows = raw_artifact["per_replicate"]
    baseline_rder = [float(row["no_nogood"]["repeat_dead_end_rate"]) for row in rows]
    candidate_rder = [float(row["local_nogood"]["repeat_dead_end_rate"]) for row in rows]
    primary = bootstrap_exp289_relative_rder(
        baseline_rder,
        candidate_rder,
        seed_material=f"{raw_artifact['artifact_digest']}|EXP-289|primary-rder",
        samples=BOOTSTRAP_SAMPLES,
        alpha=ALPHA,
    )

    invalid_prunes = sum(int(row["local_nogood"]["invalid_valid_state_prune_count"]) for row in rows)
    prune_events = sum(int(row["local_nogood"]["nogood_prune_event_count"]) for row in rows)
    overprune_rate = float(invalid_prunes / prune_events) if prune_events > 0 else 0.0
    overprune = {
        "ceiling": OVERPRUNE_CEILING,
        "invalid_valid_state_prune_count": int(invalid_prunes),
        "nogood_prune_event_count": int(prune_events),
        "observed_rate": overprune_rate,
        "guard_passed": overprune_rate <= OVERPRUNE_CEILING,
    }

    baseline_solution = [float(row["no_nogood"]["verified_solution_rate"]) for row in rows]
    candidate_solution = [float(row["local_nogood"]["verified_solution_rate"]) for row in rows]
    solution = _bootstrap_paired_difference(
        baseline_solution,
        candidate_solution,
        seed_material=f"{raw_artifact['artifact_digest']}|EXP-289|solution-rate",
        samples=BOOTSTRAP_SAMPLES,
        alpha=ALPHA,
    )
    solution["floor_difference"] = SOLUTION_FLOOR_DIFFERENCE
    solution["guard_passed"] = solution["one_sided_lower"] >= SOLUTION_FLOOR_DIFFERENCE

    decision = decide_exp289_confirmatory_outcome(
        primary_lower=primary["one_sided_lower"],
        primary_upper=primary["one_sided_upper"],
        overprune_rate=overprune_rate,
        solution_lower=solution["one_sided_lower"],
        solution_upper=solution["one_sided_upper"],
        denominator_valid=bool(primary["denominator_valid"]),
    )
    return primary, overprune, solution, decision


def _raw_mode(raw_artifact: dict[str, Any]) -> bool:
    test_only = raw_artifact.get("test_only")
    scientific = raw_artifact.get("scientific_evidence_eligible")
    if test_only is True and scientific is False:
        return True
    if test_only is False and scientific is True:
        if raw_artifact.get("confirmatory_data_consumed") is not True:
            raise ValueError("EXP-289 scientific raw artifact must consume confirmatory data")
        return False
    raise ValueError("EXP-289 raw artifact evidence mode is invalid")


def build_exp289_confirmatory_analysis(
    *,
    raw_artifact: dict[str, Any],
    seal: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    raw_errors = validate_exp289_confirmatory_raw(
        raw_artifact,
        seal=seal,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        raise ValueError("invalid EXP-289 raw/reconstruction evidence: " + "; ".join(raw_errors))
    test_only = _raw_mode(raw_artifact)
    if analysis_code_digest != seal.get("code_tree_digest"):
        raise ValueError("EXP-289 analysis code digest does not match sealed code tree")

    primary, overprune, solution, inferred_decision = _compute_inferences(raw_artifact)
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
        "reserved_replicate_ids": deepcopy(raw_artifact.get("reserved_replicate_ids")),
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
            "denominator_policy": DENOMINATOR_POLICY,
            "inference": primary,
        },
        "protected_overprune": overprune,
        "protected_solution_rate": solution,
        "lineage": {
            "protocol_digest": (seal.get("lineage") or {}).get("protocol_digest"),
            "gate_a_seal_digest": seal.get("seal_digest"),
            "pre_beacon_binding_digest": seal.get("pre_beacon_binding_digest"),
            "raw_artifact_digest": raw_artifact.get("artifact_digest"),
            "beacon_receipt_digest": (raw_artifact.get("beacon_receipt") or {}).get("receipt_digest"),
            "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
            "analysis_code_digest": analysis_code_digest,
        },
        "analysis_digest": "",
    }
    payload["analysis_digest"] = _analysis_digest(payload)
    errors = validate_exp289_confirmatory_analysis(
        payload,
        raw_artifact=raw_artifact,
        seal=seal,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if errors:
        raise RuntimeError("invalid EXP-289 Gate-B analysis: " + "; ".join(errors))
    return payload


def validate_exp289_confirmatory_analysis(
    payload: dict[str, Any],
    *,
    raw_artifact: dict[str, Any],
    seal: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-289 confirmatory analysis must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-289 Gate-B analysis identity")

    try:
        raw_test_only = _raw_mode(raw_artifact)
    except ValueError as exc:
        errors.append(str(exc))
        raw_test_only = None

    if raw_test_only is True:
        expected_status = TEST_ONLY_STATUS
        expected_boundary = {
            "test_only": True,
            "scientific_evidence_eligible": False,
            "evidence_level": "EV-E2",
            "decision": "UNVERIFIED",
            "confirmatory_data_consumed": False,
            "synthetic_challenge_data_consumed": True,
            "decision_rule_executed": False,
            "test_only_decision_executed": True,
        }
    elif raw_test_only is False:
        expected_status = SCIENTIFIC_STATUS
        expected_boundary = {
            "test_only": False,
            "scientific_evidence_eligible": True,
            "evidence_level": "EV-E3",
            "confirmatory_data_consumed": True,
            "synthetic_challenge_data_consumed": False,
            "decision_rule_executed": True,
            "test_only_decision_executed": False,
        }
    else:
        expected_status = None
        expected_boundary = {}
    if payload.get("status") != expected_status:
        errors.append("EXP-289 Gate-B analysis status/evidence-mode mismatch")
    for key, expected in expected_boundary.items():
        if payload.get(key) != expected:
            errors.append(f"EXP-289 analysis evidence boundary drift: {key}")
    if raw_test_only is False and payload.get("decision") not in VALID_DECISIONS:
        errors.append("EXP-289 scientific Gate-B decision is invalid")
    if raw_test_only is False and payload.get("test_only_would_be_decision") is not None:
        errors.append("EXP-289 scientific Gate-B cannot report a TEST-ONLY would-be decision")

    raw_errors = validate_exp289_confirmatory_raw(
        raw_artifact,
        seal=seal,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        errors.append("EXP-289 analysis raw semantic reconstruction invalid: " + "; ".join(raw_errors))
        return errors
    if payload.get("raw_artifact_digest") != raw_artifact.get("artifact_digest"):
        errors.append("EXP-289 analysis/raw artifact lineage mismatch")
    if payload.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-289 analysis/seal lineage mismatch")
    if payload.get("confirmatory_n") != raw_artifact.get("confirmatory_n"):
        errors.append("EXP-289 analysis confirmatory_n mismatch")
    if payload.get("reserved_replicate_ids") != raw_artifact.get("reserved_replicate_ids"):
        errors.append("EXP-289 analysis reserved replicate lineage mismatch")
    if payload.get("analysis_code_digest") != seal.get("code_tree_digest"):
        errors.append("EXP-289 analysis code-tree closure mismatch")

    primary = payload.get("primary_endpoint") or {}
    expected_primary = {
        "metric": PRIMARY_ENDPOINT,
        "direction": "lower",
        "effect_type": PRIMARY_EFFECT_TYPE,
        "mesi_relative_reduction": MESI_RELATIVE_REDUCTION,
        "alpha": ALPHA,
        "samples": BOOTSTRAP_SAMPLES,
        "denominator_policy": DENOMINATOR_POLICY,
    }
    for key, expected in expected_primary.items():
        if primary.get(key) != expected:
            errors.append(f"EXP-289 frozen primary analysis drift: {key}")
    overprune = payload.get("protected_overprune") or {}
    if overprune.get("ceiling") != OVERPRUNE_CEILING:
        errors.append("EXP-289 protected over-prune ceiling drift")
    solution = payload.get("protected_solution_rate") or {}
    if solution.get("floor_difference") != SOLUTION_FLOOR_DIFFERENCE:
        errors.append("EXP-289 protected solution floor drift")

    lineage = payload.get("lineage") or {}
    try:
        require_frozen_stage_a_v1_sha256(lineage.get("protocol_digest"))
    except ValueError as exc:
        errors.append(str(exc))
    expected_lineage = {
        "gate_a_seal_digest": seal.get("seal_digest"),
        "pre_beacon_binding_digest": seal.get("pre_beacon_binding_digest"),
        "raw_artifact_digest": raw_artifact.get("artifact_digest"),
        "beacon_receipt_digest": (raw_artifact.get("beacon_receipt") or {}).get("receipt_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "analysis_code_digest": seal.get("code_tree_digest"),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            errors.append(f"EXP-289 analysis lineage mismatch: {key}")

    if not errors:
        try:
            expected_primary_inference, expected_overprune, expected_solution, expected_decision = _compute_inferences(raw_artifact)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"EXP-289 analysis semantic reconstruction failed: {exc}")
        else:
            if primary.get("inference") != expected_primary_inference:
                errors.append("EXP-289 primary analysis semantic reconstruction mismatch")
            if overprune != expected_overprune:
                errors.append("EXP-289 over-prune analysis semantic reconstruction mismatch")
            if solution != expected_solution:
                errors.append("EXP-289 solution analysis semantic reconstruction mismatch")
            if expected_decision not in VALID_DECISIONS:
                errors.append("EXP-289 reconstructed decision is outside frozen decision set")
            elif raw_test_only is True:
                if (
                    payload.get("test_only_would_be_decision") != expected_decision
                    or payload.get("decision") != "UNVERIFIED"
                ):
                    errors.append("EXP-289 TEST-ONLY decision semantic reconstruction mismatch")
            elif raw_test_only is False:
                if (
                    payload.get("test_only_would_be_decision") is not None
                    or payload.get("decision") != expected_decision
                ):
                    errors.append("EXP-289 scientific decision semantic reconstruction mismatch")
    if payload.get("analysis_digest") != _analysis_digest(payload):
        errors.append("EXP-289 analysis digest mismatch")
    return errors
