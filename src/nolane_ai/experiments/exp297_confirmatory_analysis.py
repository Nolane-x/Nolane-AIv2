from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import random
from statistics import NormalDist, fmean, median
from typing import Any

from nolane_ai.experiments.exp297_confirmatory_executor import (
    validate_exp297_confirmatory_raw,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    ALPHA_PER_ENDPOINT,
    BOOTSTRAP_SAMPLES,
    FAITHFUL_REJECTION_LIMIT,
    MESI,
    PRIMARY_ENDPOINT,
    WRONG_AUTHORITY_LIMIT,
    _frozen_analysis,
)
from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-297-CONFIRMATORY-ANALYSIS-V1"
EXPERIMENT_ID = "EXP-297"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_ANALYZED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_ANALYZED"
VALID_SCIENTIFIC_DECISIONS = {
    "PROMOTE_TO_NEXT_STAGE",
    "HOLD_UNSTABLE",
    "KILL_SUBSYSTEM",
}
BOOTSTRAP_SEED_RULE = (
    "SHA256(protocol_digest|frozen_analysis_digest|EXP-297|paired-bootstrap-v1)"
)


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


def _close(left: Any, right: Any, *, tol: float = 1e-12) -> bool:
    return _finite(left) and _finite(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tol
    )


def bootstrap_exp297_paired_gain(
    values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ALPHA_PER_ENDPOINT,
) -> dict[str, Any]:
    if not values or any(not _finite(value) for value in values):
        raise ValueError("EXP-297 paired bootstrap requires non-empty finite effects")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("EXP-297 bootstrap seed material is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("EXP-297 bootstrap sample count must be positive")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-297 bootstrap alpha must be in (0, 0.5)")

    seed_digest = sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(values)
    means = [
        fmean(float(values[rng.randrange(n)]) for _ in range(n))
        for _ in range(samples)
    ]
    means.sort()
    alpha_f = float(alpha)
    lower_index = max(0, math.ceil(alpha_f * samples) - 1)
    upper_index = min(samples - 1, math.ceil((1.0 - alpha_f) * samples) - 1)
    return {
        "observed_mean": float(fmean(values)),
        "median": float(median(values)),
        "one_sided_lower": float(means[lower_index]),
        "one_sided_upper": float(means[upper_index]),
        "positive_count": sum(float(value) > 0.0 for value in values),
        "zero_count": sum(float(value) == 0.0 for value in values),
        "negative_count": sum(float(value) < 0.0 for value in values),
        "samples": samples,
        "alpha": alpha_f,
        "seed_digest": seed_digest,
    }


def wilson_upper_bound(events: int, total: int, *, alpha: float) -> float:
    if (
        not isinstance(events, int)
        or isinstance(events, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or events < 0
        or events > total
    ):
        raise ValueError("EXP-297 Wilson counts are invalid")
    if not _finite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("EXP-297 Wilson alpha must be in (0, 0.5)")
    z = NormalDist().inv_cdf(1.0 - float(alpha))
    n = float(total)
    phat = events / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
        / denominator
    )
    return min(1.0, max(0.0, center + margin))


def decide_exp297_confirmatory_outcome(
    *,
    primary_lower: float,
    primary_upper: float,
    wrong_authority_upper: float,
    faithful_rejection_upper: float,
) -> str:
    values = (
        primary_lower,
        primary_upper,
        wrong_authority_upper,
        faithful_rejection_upper,
    )
    if any(not _finite(value) for value in values):
        raise ValueError("EXP-297 confirmatory decision inputs must be finite")
    safety_pass = (
        float(wrong_authority_upper) <= WRONG_AUTHORITY_LIMIT
        and float(faithful_rejection_upper) <= FAITHFUL_REJECTION_LIMIT
    )
    if float(primary_lower) >= MESI and safety_pass:
        return "PROMOTE_TO_NEXT_STAGE"
    if not safety_pass or float(primary_upper) < MESI:
        return "KILL_SUBSYSTEM"
    return "HOLD_UNSTABLE"


def _arm_balanced_accuracy(rows: list[dict[str, Any]], arm_id: str) -> dict[str, Any]:
    faithful = [row for row in rows if row.get("is_faithful") is True]
    wrong = [row for row in rows if row.get("is_faithful") is False]
    if len(faithful) != 8 or len(wrong) != 8:
        raise ValueError("EXP-297 analysis replicate must contain 8 faithful and 8 wrong candidates")
    faithful_accept = sum(
        bool(((row.get("arms") or {}).get(arm_id) or {}).get("authority_granted"))
        for row in faithful
    )
    wrong_reject = sum(
        not bool(((row.get("arms") or {}).get(arm_id) or {}).get("authority_granted"))
        for row in wrong
    )
    faithful_reject = len(faithful) - faithful_accept
    wrong_authority = len(wrong) - wrong_reject
    tpr = faithful_accept / len(faithful)
    tnr = wrong_reject / len(wrong)
    return {
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "faithful_accept": faithful_accept,
        "faithful_reject": faithful_reject,
        "faithful_total": len(faithful),
        "wrong_reject": wrong_reject,
        "wrong_authority": wrong_authority,
        "wrong_total": len(wrong),
    }


def _reconstruct_metrics(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    reserved = raw.get("reserved_replicate_ids") or []
    rows = raw.get("raw_candidates") or []
    if not isinstance(reserved, list) or not isinstance(rows, list):
        raise ValueError("EXP-297 raw replicate/candidate evidence is missing")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        replicate = row.get("replicate")
        if not isinstance(replicate, int) or isinstance(replicate, bool):
            raise ValueError("EXP-297 raw analysis replicate identity invalid")
        grouped.setdefault(replicate, []).append(row)
    if list(grouped) != reserved:
        raise ValueError("EXP-297 raw analysis replicate order does not match reservation")

    per_replicate: list[dict[str, Any]] = []
    effects: list[float] = []
    wrong_events = wrong_total = faithful_events = faithful_total = 0
    for replicate in reserved:
        replicate_rows = grouped.get(replicate) or []
        if len(replicate_rows) != 16:
            raise ValueError("EXP-297 analysis replicate must contain exactly 16 candidates")
        if [row.get("candidate_index") for row in replicate_rows] != list(range(16)):
            raise ValueError("EXP-297 analysis candidate order drift")
        compile_metrics = _arm_balanced_accuracy(replicate_rows, "compile_only")
        fidelity_metrics = _arm_balanced_accuracy(replicate_rows, "fidelity_court")
        effect = float(fidelity_metrics["balanced_accuracy"] - compile_metrics["balanced_accuracy"])
        effects.append(effect)
        wrong_events += int(fidelity_metrics["wrong_authority"])
        wrong_total += int(fidelity_metrics["wrong_total"])
        faithful_events += int(fidelity_metrics["faithful_reject"])
        faithful_total += int(fidelity_metrics["faithful_total"])
        per_replicate.append(
            {
                "replicate": replicate,
                "compile_only": compile_metrics,
                "fidelity_court": fidelity_metrics,
                "fidelity_court_minus_compile_only_balanced_accuracy": effect,
            }
        )
    return per_replicate, effects, {
        "wrong_authority_events": wrong_events,
        "wrong_authority_total": wrong_total,
        "faithful_rejection_events": faithful_events,
        "faithful_rejection_total": faithful_total,
    }


def _safety_summary(*, metric: str, events: int, total: int, limit: float) -> dict[str, Any]:
    upper = wilson_upper_bound(events, total, alpha=ALPHA_PER_ENDPOINT)
    return {
        "metric": metric,
        "events": events,
        "total": total,
        "rate": events / total,
        "one_sided_upper": upper,
        "limit": limit,
        "alpha": ALPHA_PER_ENDPOINT,
        "pass": upper <= limit,
    }


def _expected_frozen_contract(raw: dict[str, Any]) -> dict[str, Any]:
    reconstruction = raw.get("reconstruction_authorization") or {}
    frozen = _frozen_analysis()
    observed_digest = reconstruction.get("frozen_analysis_digest")
    if observed_digest != canonical_sha256(frozen):
        raise ValueError("EXP-297 raw reconstruction frozen-analysis digest drift")
    contract = deepcopy(frozen)
    contract["bootstrap_seed_rule"] = BOOTSTRAP_SEED_RULE
    return contract


def _frozen_analysis_code_digest(raw: dict[str, Any]) -> str:
    reconstruction = raw.get("reconstruction_authorization") or {}
    authorization = reconstruction.get("execution_authorization") or {}
    digest = (authorization.get("lineage") or {}).get("analysis_code_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("EXP-297 frozen analysis code digest missing")
    return digest


def _expected_statistics(raw: dict[str, Any]) -> dict[str, Any]:
    per_replicate, effects, counts = _reconstruct_metrics(raw)
    reconstruction = raw.get("reconstruction_authorization") or {}
    protocol_digest = (reconstruction.get("lineage") or {}).get("protocol_digest")
    frozen_analysis_digest = reconstruction.get("frozen_analysis_digest")
    if not isinstance(protocol_digest, str) or not protocol_digest:
        raise ValueError("EXP-297 analysis protocol lineage missing")
    if not isinstance(frozen_analysis_digest, str) or not frozen_analysis_digest:
        raise ValueError("EXP-297 frozen analysis digest missing")
    seed_material = (
        f"{protocol_digest}|{frozen_analysis_digest}|{EXPERIMENT_ID}|paired-bootstrap-v1"
    )
    primary = bootstrap_exp297_paired_gain(
        effects,
        seed_material=seed_material,
        samples=BOOTSTRAP_SAMPLES,
        alpha=ALPHA_PER_ENDPOINT,
    )
    primary.update(
        {
            "metric": PRIMARY_ENDPOINT,
            "effect_type": "absolute_gain",
            "baseline_arm": "compile_only",
            "candidate_arm": "fidelity_court",
            "mesi_absolute_gain": MESI,
            "replicates": len(effects),
        }
    )
    wrong = _safety_summary(
        metric="wrong_formalization_authority_rate",
        events=int(counts["wrong_authority_events"]),
        total=int(counts["wrong_authority_total"]),
        limit=WRONG_AUTHORITY_LIMIT,
    )
    faithful = _safety_summary(
        metric="faithful_formalization_rejection_rate",
        events=int(counts["faithful_rejection_events"]),
        total=int(counts["faithful_rejection_total"]),
        limit=FAITHFUL_REJECTION_LIMIT,
    )
    would_be = decide_exp297_confirmatory_outcome(
        primary_lower=float(primary["one_sided_lower"]),
        primary_upper=float(primary["one_sided_upper"]),
        wrong_authority_upper=float(wrong["one_sided_upper"]),
        faithful_rejection_upper=float(faithful["one_sided_upper"]),
    )
    return {
        "per_replicate": per_replicate,
        "primary": primary,
        "wrong": wrong,
        "faithful": faithful,
        "would_be_decision": would_be,
    }


def build_exp297_confirmatory_analysis(
    *,
    raw_artifact: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    raw_errors = validate_exp297_confirmatory_raw(raw_artifact)
    if raw_errors:
        raise ValueError("invalid EXP-297 confirmatory raw artifact: " + "; ".join(raw_errors))
    frozen_code = _frozen_analysis_code_digest(raw_artifact)
    if analysis_code_digest != frozen_code:
        raise ValueError("EXP-297 analysis code digest does not match pre-beacon freeze")

    frozen_contract = _expected_frozen_contract(raw_artifact)
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
        test_only_would_be = stats["would_be_decision"]
    else:
        evidence_level = "EV-E3"
        decision = stats["would_be_decision"]
        status = SCIENTIFIC_STATUS
        scientific_eligible = raw_artifact.get("scientific_evidence_eligible") is True
        confirmatory_consumed = True
        synthetic_consumed = False
        decision_rule_executed = True
        test_only_decision_executed = False
        test_only_would_be = None

    reconstruction = raw_artifact.get("reconstruction_authorization") or {}
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
        "test_only_decision_rule_executed": test_only_decision_executed,
        "semantic_authority_promoted": False,
        "test_only_would_be_decision": test_only_would_be,
        "confirmatory_n": raw_artifact.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(raw_artifact.get("reserved_replicate_ids") or []),
        "frozen_contract": frozen_contract,
        "primary_effect": stats["primary"],
        "protected_endpoints": {
            "wrong_authority": stats["wrong"],
            "faithful_rejection": stats["faithful"],
        },
        "raw_per_replicate_metrics": stats["per_replicate"],
        "raw_artifact": deepcopy(raw_artifact),
        "lineage": {
            "protocol_digest": (reconstruction.get("lineage") or {}).get("protocol_digest"),
            "frozen_analysis_digest": reconstruction.get("frozen_analysis_digest"),
            "sample_size_freeze_digest": reconstruction.get("sample_size_freeze_digest"),
            "reconstruction_digest": reconstruction.get("reconstruction_digest"),
            "raw_artifact_digest": raw_artifact.get("artifact_digest"),
            "beacon_receipt_digest": (raw_artifact.get("lineage") or {}).get("beacon_receipt_digest"),
            "executor_code_digest": (raw_artifact.get("lineage") or {}).get("executor_code_digest"),
            "analysis_code_digest": analysis_code_digest,
        },
        "analysis_digest": "",
    }
    payload["analysis_digest"] = _analysis_digest(payload)
    errors = validate_exp297_confirmatory_analysis(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 confirmatory analysis: " + "; ".join(errors))
    return payload


def validate_exp297_confirmatory_analysis(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-297 confirmatory analysis identity")
    if payload.get("analysis_digest") != _analysis_digest(payload):
        errors.append("EXP-297 confirmatory analysis digest mismatch")
    if payload.get("semantic_authority_promoted") is not False:
        errors.append("EXP-297 confirmatory analysis cannot promote unrestricted semantic authority")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-297 confirmatory analysis challenge-materialization boundary drift")

    raw = payload.get("raw_artifact")
    if not isinstance(raw, dict):
        errors.append("EXP-297 confirmatory analysis raw artifact missing")
        return errors
    raw_errors = validate_exp297_confirmatory_raw(raw)
    if raw_errors:
        errors.append("embedded EXP-297 raw artifact invalid: " + "; ".join(raw_errors))
        return errors

    raw_test_only = raw.get("test_only") is True
    if payload.get("test_only") is not raw_test_only:
        errors.append("EXP-297 analysis/raw TEST-ONLY classification mismatch")
    if raw_test_only:
        if payload.get("status") != TEST_ONLY_STATUS:
            errors.append("EXP-297 TEST-ONLY analysis status drift")
        if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
            errors.append("EXP-297 TEST-ONLY analysis cannot promote scientific evidence")
        if payload.get("scientific_evidence_eligible") is not False:
            errors.append("EXP-297 TEST-ONLY analysis cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-297 TEST-ONLY analysis cannot consume confirmatory data")
        if payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("EXP-297 TEST-ONLY analysis synthetic-consumption flag drift")
        if payload.get("decision_rule_executed") is not False:
            errors.append("EXP-297 TEST-ONLY analysis cannot execute scientific decision rule")
        if payload.get("test_only_decision_rule_executed") is not True:
            errors.append("EXP-297 TEST-ONLY diagnostic decision-rule flag drift")
    else:
        if payload.get("status") != SCIENTIFIC_STATUS:
            errors.append("EXP-297 scientific analysis status drift")
        if payload.get("evidence_level") != "EV-E3":
            errors.append("EXP-297 scientific confirmatory analysis must be EV-E3")
        if payload.get("decision") not in VALID_SCIENTIFIC_DECISIONS:
            errors.append("EXP-297 scientific confirmatory decision invalid")
        if payload.get("scientific_evidence_eligible") is not True:
            errors.append("EXP-297 scientific analysis eligibility missing")
        if payload.get("confirmatory_data_consumed") is not True:
            errors.append("EXP-297 scientific analysis confirmatory-consumption flag missing")
        if payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("EXP-297 scientific analysis cannot claim synthetic consumption")
        if payload.get("decision_rule_executed") is not True:
            errors.append("EXP-297 scientific analysis must record decision-rule execution")
        if payload.get("test_only_decision_rule_executed") is not False:
            errors.append("EXP-297 scientific analysis TEST-ONLY rule flag drift")
        if payload.get("test_only_would_be_decision") is not None:
            errors.append("EXP-297 scientific analysis cannot expose TEST-ONLY decision field")

    try:
        expected_contract = _expected_frozen_contract(raw)
        expected_code = _frozen_analysis_code_digest(raw)
        expected_stats = _expected_statistics(raw)
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"EXP-297 confirmatory analysis reconstruction failed: {exc}")
        return errors

    if payload.get("frozen_contract") != expected_contract:
        errors.append("EXP-297 confirmatory frozen analysis contract drift")
    if payload.get("confirmatory_n") != raw.get("confirmatory_n"):
        errors.append("EXP-297 confirmatory analysis n mismatch")
    if payload.get("reserved_replicate_ids") != raw.get("reserved_replicate_ids"):
        errors.append("EXP-297 confirmatory analysis reserved replicate mismatch")
    if payload.get("raw_per_replicate_metrics") != expected_stats["per_replicate"]:
        errors.append("EXP-297 confirmatory per-replicate metric reconstruction mismatch")
    if payload.get("primary_effect") != expected_stats["primary"]:
        errors.append("EXP-297 confirmatory primary bootstrap reconstruction mismatch")
    protected = payload.get("protected_endpoints") or {}
    if protected.get("wrong_authority") != expected_stats["wrong"]:
        errors.append("EXP-297 wrong-authority safety reconstruction mismatch")
    if protected.get("faithful_rejection") != expected_stats["faithful"]:
        errors.append("EXP-297 faithful-rejection safety reconstruction mismatch")

    expected_decision = expected_stats["would_be_decision"]
    if raw_test_only:
        if payload.get("test_only_would_be_decision") != expected_decision:
            errors.append("EXP-297 TEST-ONLY would-be decision mismatch")
    elif payload.get("decision") != expected_decision:
        errors.append("EXP-297 scientific decision does not match frozen rule")

    reconstruction = raw.get("reconstruction_authorization") or {}
    lineage = payload.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": (reconstruction.get("lineage") or {}).get("protocol_digest"),
        "frozen_analysis_digest": reconstruction.get("frozen_analysis_digest"),
        "sample_size_freeze_digest": reconstruction.get("sample_size_freeze_digest"),
        "reconstruction_digest": reconstruction.get("reconstruction_digest"),
        "raw_artifact_digest": raw.get("artifact_digest"),
        "beacon_receipt_digest": (raw.get("lineage") or {}).get("beacon_receipt_digest"),
        "executor_code_digest": (raw.get("lineage") or {}).get("executor_code_digest"),
        "analysis_code_digest": expected_code,
    }
    if lineage != expected_lineage:
        errors.append("EXP-297 confirmatory analysis lineage mismatch")
    return errors
