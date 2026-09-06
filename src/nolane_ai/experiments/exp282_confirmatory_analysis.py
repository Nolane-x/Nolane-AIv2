from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import random
from statistics import fmean, median
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import require_canonical_stage_a_v1_digest
from .exp282_analysis_lineage import validate_exp282_prep_pilot_lineage
from .exp282_confirmatory_executor import validate_exp282_confirmatory_open_raw
from .exp282_confirmatory_prep import validate_exp282_confirmatory_prep
from .exp282_paired_runner import validate_exp282_paired_development
from .exp282_reconstruction_court import validate_exp282_confirmatory_reconstruction
from .matched_belief_arms import account_matched_belief_arm_pair, build_matched_belief_arm_pair

SCHEMA = "NLM-EXP-282-CONFIRMATORY-ANALYSIS-V1"
CLAIM_ID = "H-BELIEF-01"
EXPERIMENT_ID = "EXP-282"
PRIMARY_ENDPOINT = "grounded_decision_accuracy"
MESI = 0.03
FAMILYWISE_ALPHA = 0.05
BRIER_MARGIN = 0.02
COMPUTE_MARGIN = 0.05
BOOTSTRAP_SAMPLES = 10_000
ANALYSIS_METHOD = "paired accuracy difference with bootstrap CI plus calibration guard"
BRIER_GUARD = "explicit_belief <= recurrent_hidden + 0.02"
COMPUTE_GUARD = "difference <= 0.05 relative unless included in primary cost normalization"
STATUS = "CONFIRMATORY_OPEN_ANALYZED"
VALID_DECISIONS = {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}
REMAINING_BLOCKERS = [
    "100M post-freeze challenge replication remains open for EV-E4",
    "independent clean-room replication remains open for EV-E5",
]


def _analysis_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("analysis_digest", None)
    return canonical_sha256(clean)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _close(left: Any, right: Any, *, tol: float = 1e-12) -> bool:
    return _finite(left) and _finite(right) and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)


def _frozen_exp282(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "NLM-REASONING-STAGE-A-CONFIRMATORY-V1":
        raise ValueError("invalid frozen protocol id")
    if protocol.get("status") != "FROZEN_V1":
        raise ValueError("protocol is not FROZEN_V1")
    experiment = next(
        (item for item in (protocol.get("experiments") or []) if item.get("experiment_id") == EXPERIMENT_ID),
        None,
    )
    if experiment is None:
        raise ValueError("frozen protocol EXP-282 is missing")
    primary = experiment.get("primary_endpoint") or {}
    if primary.get("metric") != PRIMARY_ENDPOINT or primary.get("direction") != "higher":
        raise ValueError("frozen EXP-282 primary endpoint drift")
    mesi = experiment.get("mesi") or {}
    if mesi.get("type") != "absolute_gain" or not _close(mesi.get("value"), MESI):
        raise ValueError("frozen EXP-282 MESI drift")
    sample = experiment.get("sample_size_plan") or {}
    if int(sample.get("min_n", -1)) != 32 or int(sample.get("max_n", -1)) != 128 or sample.get("paired") is not True:
        raise ValueError("frozen EXP-282 sample-size contract drift")
    if experiment.get("analysis_method") != ANALYSIS_METHOD:
        raise ValueError("frozen EXP-282 analysis method drift")
    if experiment.get("multiplicity_family") != "BELIEF_STATE":
        raise ValueError("frozen EXP-282 multiplicity family drift")
    protected = {item.get("metric"): item.get("floor") for item in (experiment.get("protected_endpoints") or [])}
    if protected.get("brier_score") != BRIER_GUARD:
        raise ValueError("frozen EXP-282 Brier guard drift")
    if protected.get("accounted_flops") != COMPUTE_GUARD:
        raise ValueError("frozen EXP-282 compute guard drift")
    global_plan = protocol.get("global_sample_size_plan") or {}
    if not _close(global_plan.get("familywise_alpha"), FAMILYWISE_ALPHA):
        raise ValueError("frozen familywise alpha drift")


def _frozen_analysis_digest(prep_artifact: dict[str, Any]) -> str:
    frozen = prep_artifact.get("frozen_analysis") or {}
    expected = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": "higher",
        "mesi_absolute_gain": MESI,
        "power_target": 0.90,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "inference_tail": "one_sided_lower_bound",
        "analysis_method": ANALYSIS_METHOD,
        "multiplicity_family": "BELIEF_STATE",
        "brier_guard": BRIER_GUARD,
        "compute_guard": COMPUTE_GUARD,
    }
    for key, expected_value in expected.items():
        observed = frozen.get(key)
        if isinstance(expected_value, float):
            if not _close(observed, expected_value):
                raise ValueError(f"frozen analysis {key} drift")
        elif observed != expected_value:
            raise ValueError(f"frozen analysis {key} drift")
    return canonical_sha256(frozen)


def bootstrap_paired_effect(
    values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    if not values or any(not _finite(value) for value in values):
        raise ValueError("paired bootstrap requires a non-empty finite effect vector")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("bootstrap alpha must be in (0, 0.5)")
    seed_digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(seed_digest)[:8], "big"))
    n = len(values)
    means = [fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    means.sort()
    lower_index = max(0, math.ceil(alpha * samples) - 1)
    upper_index = min(samples - 1, math.ceil((1.0 - alpha) * samples) - 1)
    return {
        "observed_mean": fmean(values),
        "median": float(median(values)),
        "one_sided_lower_95": float(means[lower_index]),
        "one_sided_upper_95": float(means[upper_index]),
        "positive_count": sum(value > 0.0 for value in values),
        "zero_count": sum(value == 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "samples": samples,
        "alpha": alpha,
        "seed_digest": seed_digest,
    }


def _paired_effects(rows: list[dict[str, Any]]) -> list[float]:
    effects: list[float] = []
    for row in rows:
        recurrent = float((row.get("recurrent_hidden") or {}).get(PRIMARY_ENDPOINT, math.nan))
        explicit = float((row.get("explicit_belief") or {}).get(PRIMARY_ENDPOINT, math.nan))
        declared = row.get("explicit_minus_recurrent_accuracy")
        effect = explicit - recurrent
        if not _finite(declared) or not math.isclose(float(declared), effect, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("confirmatory paired accuracy contrast mismatch")
        effects.append(effect)
    return effects


def _brier_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recurrent_sum = explicit_sum = 0.0
    recurrent_count = explicit_count = 0
    for row in rows:
        recurrent = row.get("recurrent_hidden") or {}
        explicit = row.get("explicit_belief") or {}
        recurrent_sum += float(recurrent.get("brier_sum", math.nan))
        explicit_sum += float(explicit.get("brier_sum", math.nan))
        recurrent_count += int(recurrent.get("brier_count", 0) or 0)
        explicit_count += int(explicit.get("brier_count", 0) or 0)
    if not all(math.isfinite(value) for value in (recurrent_sum, explicit_sum)):
        raise ValueError("confirmatory Brier sufficient statistics are non-finite")
    if recurrent_count <= 0 or explicit_count <= 0:
        raise ValueError("confirmatory Brier sufficient-statistic counts are invalid")
    recurrent_score = recurrent_sum / recurrent_count
    explicit_score = explicit_sum / explicit_count
    delta = explicit_score - recurrent_score
    return {
        "metric": "brier_score",
        "guard": BRIER_GUARD,
        "margin": BRIER_MARGIN,
        "recurrent_brier_sum": recurrent_sum,
        "recurrent_brier_count": recurrent_count,
        "recurrent_brier_score": recurrent_score,
        "explicit_brier_sum": explicit_sum,
        "explicit_brier_count": explicit_count,
        "explicit_brier_score": explicit_score,
        "delta": delta,
        "pass": delta <= BRIER_MARGIN,
    }


def _compute_summary(execution_contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_geometry = deepcopy(execution_contract.get("arm_geometry") or {})
    world_geometry = deepcopy(execution_contract.get("world_geometry") or {})
    for key in ("d_model", "hidden_size", "target_parameters"):
        if int(arm_geometry.get(key, 0) or 0) <= 0:
            raise ValueError(f"analysis arm geometry {key} is invalid")
    for key in ("timesteps", "variables"):
        if int(world_geometry.get(key, 0) or 0) <= 0:
            raise ValueError(f"analysis world geometry {key} is invalid")
    recurrent, explicit = build_matched_belief_arm_pair(
        d_model=int(arm_geometry["d_model"]),
        hidden_size=int(arm_geometry["hidden_size"]),
        target_parameters=int(arm_geometry["target_parameters"]),
        device="meta",
    )
    ledger = account_matched_belief_arm_pair(
        recurrent,
        explicit,
        timesteps=int(world_geometry["timesteps"]),
        variables=int(world_geometry["variables"]),
    )
    relative = float(ledger["relative_accounted_flop_difference"])
    passed = (
        ledger.get("primitive_operation_match") is True
        and relative <= COMPUTE_MARGIN
        and ledger.get("hardware_profiler_flops_claimed") is False
    )
    return {
        "metric": "accounted_flops",
        "guard": COMPUTE_GUARD,
        "margin": COMPUTE_MARGIN,
        "relative_accounted_flop_difference": relative,
        "primitive_operation_match": ledger.get("primitive_operation_match"),
        "accounted_flops_match": ledger.get("accounted_flops_match"),
        "hardware_profiler_flops_claimed": ledger.get("hardware_profiler_flops_claimed"),
        "pass": passed,
        "ledger": ledger,
    }, {"arm_geometry": arm_geometry, "world_geometry": world_geometry}


def _decision(*, lower: float, upper: float, brier_pass: bool, compute_pass: bool) -> str:
    if lower >= MESI and brier_pass and compute_pass:
        return "PROMOTE_TO_NEXT_STAGE"
    if upper < MESI:
        return "KILL_SUBSYSTEM"
    return "HOLD_UNSTABLE"


def _lineage(
    *,
    protocol_digest: str,
    prep_artifact: dict[str, Any],
    paired_execution_artifact: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    raw_artifact: dict[str, Any],
    analysis_code_digest: str,
) -> tuple[str, str, int, list[int]]:
    if not analysis_code_digest:
        raise ValueError("analysis_code_digest is required")
    prep_lineage = prep_artifact.get("lineage") or {}
    paired_digest = paired_execution_artifact.get("artifact_digest")
    checkpoint = paired_execution_artifact.get("checkpoint") or {}
    checkpoint_sha = checkpoint.get("checkpoint_sha256")
    reconstruction_lineage = reconstruction_authorization.get("lineage") or {}
    raw_lineage = raw_artifact.get("lineage") or {}

    if any(
        digest != protocol_digest
        for digest in (
            prep_lineage.get("protocol_digest"),
            paired_execution_artifact.get("protocol_digest"),
            reconstruction_lineage.get("protocol_digest"),
            raw_lineage.get("protocol_digest"),
        )
    ):
        raise ValueError("protocol digest lineage mismatch")
    if prep_lineage.get("analysis_code_digest") != analysis_code_digest:
        raise ValueError("frozen analysis code digest mismatch")
    if not paired_digest or any(
        digest != paired_digest
        for digest in (
            prep_lineage.get("execution_artifact_digest"),
            reconstruction_lineage.get("paired_execution_artifact_digest"),
            raw_lineage.get("paired_execution_artifact_digest"),
        )
    ):
        raise ValueError("paired execution artifact lineage mismatch")
    if not checkpoint_sha or any(
        digest != checkpoint_sha
        for digest in (
            prep_lineage.get("paired_checkpoint_sha256"),
            reconstruction_authorization.get("checkpoint_sha256"),
            reconstruction_lineage.get("paired_checkpoint_sha256"),
            raw_lineage.get("paired_checkpoint_sha256"),
        )
    ):
        raise ValueError("paired checkpoint lineage mismatch")

    reconstruction_digest = reconstruction_authorization.get("reconstruction_digest")
    if not reconstruction_digest or raw_lineage.get("reconstruction_digest") != reconstruction_digest:
        raise ValueError("raw reconstruction digest mismatch")
    contract_digest = reconstruction_authorization.get("execution_contract_digest")
    if not contract_digest or raw_lineage.get("execution_contract_digest") != contract_digest:
        raise ValueError("raw execution contract digest mismatch")
    if checkpoint.get("execution_contract_digest") != contract_digest:
        raise ValueError("paired checkpoint execution contract digest mismatch")

    frozen_analysis_digest = canonical_sha256(prep_artifact.get("frozen_analysis") or {})
    if reconstruction_authorization.get("frozen_analysis_digest") != frozen_analysis_digest:
        raise ValueError("reconstruction frozen-analysis digest mismatch")
    sample_size_freeze_digest = canonical_sha256(prep_artifact.get("sample_size_freeze") or {})
    if reconstruction_authorization.get("sample_size_freeze_digest") != sample_size_freeze_digest:
        raise ValueError("reconstruction sample-size freeze digest mismatch")

    confirmatory_n = int((prep_artifact.get("sample_size_freeze") or {}).get("confirmatory_n", 0) or 0)
    reserved = list((prep_artifact.get("confirmatory_lineage") or {}).get("reserved_replicate_ids") or [])
    if reconstruction_authorization.get("confirmatory_n") != confirmatory_n or raw_artifact.get("confirmatory_n") != confirmatory_n:
        raise ValueError("confirmatory n lineage mismatch")
    if reconstruction_authorization.get("reserved_replicate_ids") != reserved or raw_artifact.get("reserved_replicate_ids") != reserved:
        raise ValueError("reserved confirmatory replicate lineage mismatch")
    if [row.get("replicate") for row in (raw_artifact.get("per_replicate") or [])] != reserved:
        raise ValueError("raw replicate order does not match frozen reserved order")
    return frozen_analysis_digest, sample_size_freeze_digest, confirmatory_n, reserved


def build_exp282_confirmatory_analysis(
    *,
    protocol: dict[str, Any],
    protocol_digest: str,
    prep_artifact: dict[str, Any],
    paired_execution_artifact: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    raw_artifact: dict[str, Any],
    analysis_code_digest: str,
) -> dict[str, Any]:
    require_canonical_stage_a_v1_digest(protocol_digest)
    _frozen_exp282(protocol)

    validators = (
        ("confirmatory prep artifact", validate_exp282_confirmatory_prep(prep_artifact)),
        ("paired execution artifact", validate_exp282_paired_development(paired_execution_artifact)),
        ("reconstruction authorization", validate_exp282_confirmatory_reconstruction(reconstruction_authorization)),
        ("confirmatory raw artifact", validate_exp282_confirmatory_open_raw(raw_artifact)),
    )
    for label, errors in validators:
        if errors:
            raise ValueError(f"invalid {label}: " + "; ".join(errors))
    pilot_lineage_errors = validate_exp282_prep_pilot_lineage(prep_artifact, paired_execution_artifact)
    if pilot_lineage_errors:
        raise ValueError("pilot summary does not match paired execution: " + "; ".join(pilot_lineage_errors))
    if raw_artifact.get("confirmatory_data_consumed") is not True or raw_artifact.get("challenge_materialized") is not False:
        raise ValueError("confirmatory raw evidence boundary drift")

    frozen_analysis_digest = _frozen_analysis_digest(prep_artifact)
    frozen_digest2, sample_size_freeze_digest, confirmatory_n, reserved = _lineage(
        protocol_digest=protocol_digest,
        prep_artifact=prep_artifact,
        paired_execution_artifact=paired_execution_artifact,
        reconstruction_authorization=reconstruction_authorization,
        raw_artifact=raw_artifact,
        analysis_code_digest=analysis_code_digest,
    )
    if frozen_digest2 != frozen_analysis_digest:
        raise RuntimeError("internal frozen-analysis digest disagreement")

    rows = list(raw_artifact["per_replicate"])
    effects = _paired_effects(rows)
    seed_material = f"{protocol_digest}|{frozen_analysis_digest}|{EXPERIMENT_ID}|paired-bootstrap-v1"
    primary = bootstrap_paired_effect(effects, seed_material=seed_material)
    primary.update(
        {
            "metric": PRIMARY_ENDPOINT,
            "effect_type": "absolute_gain",
            "baseline_arm": "recurrent_hidden",
            "candidate_arm": "explicit_belief",
            "mesi_absolute_gain": MESI,
            "replicates": len(effects),
        }
    )
    brier = _brier_summary(rows)
    execution_contract = deepcopy(reconstruction_authorization.get("execution_contract") or {})
    compute, geometry = _compute_summary(execution_contract)
    decision = _decision(
        lower=float(primary["one_sided_lower_95"]),
        upper=float(primary["one_sided_upper_95"]),
        brier_pass=bool(brier["pass"]),
        compute_pass=bool(compute["pass"]),
    )
    checkpoint = paired_execution_artifact.get("checkpoint") or {}
    raw_lineage = raw_artifact.get("lineage") or {}
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_id": CLAIM_ID,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E3",
        "decision": decision,
        "status": STATUS,
        "scope": "exp282-stage-a-confirmatory-open-analysis",
        "confirmatory_data_consumed": True,
        "challenge_materialized": False,
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "frozen_contract": {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "primary_direction": "higher",
            "mesi_absolute_gain": MESI,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "analysis_method": ANALYSIS_METHOD,
            "brier_guard": BRIER_GUARD,
            "compute_guard": COMPUTE_GUARD,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed_rule": "SHA256(protocol_digest|frozen_analysis_digest|EXP-282|paired-bootstrap-v1)",
        },
        "primary_effect": primary,
        "protected_endpoints": {"brier": brier, "compute": compute},
        "execution_contract": execution_contract,
        "execution_geometry": geometry,
        "raw_artifact": deepcopy(raw_artifact),
        "raw_per_replicate_metrics": deepcopy(rows),
        "lineage": {
            "protocol_digest": protocol_digest,
            "prep_digest": prep_artifact.get("prep_digest"),
            "paired_execution_artifact_digest": paired_execution_artifact.get("artifact_digest"),
            "reconstruction_digest": reconstruction_authorization.get("reconstruction_digest"),
            "raw_artifact_digest": raw_artifact.get("artifact_digest"),
            "paired_checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
            "execution_contract_digest": reconstruction_authorization.get("execution_contract_digest"),
            "frozen_analysis_digest": frozen_analysis_digest,
            "sample_size_freeze_digest": sample_size_freeze_digest,
            "prep_analysis_code_digest": (prep_artifact.get("lineage") or {}).get("analysis_code_digest"),
            "analysis_code_digest": analysis_code_digest,
            "executor_code_digest": raw_lineage.get("executor_code_digest"),
        },
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "analysis_digest": "",
    }
    payload["analysis_digest"] = _analysis_digest(payload)
    errors = validate_exp282_confirmatory_analysis(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 confirmatory analysis artifact: " + "; ".join(errors))
    return payload


def validate_exp282_confirmatory_analysis(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-282 confirmatory analysis schema")
    if payload.get("claim_id") != CLAIM_ID or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("confirmatory analysis claim identity drift")
    if payload.get("evidence_level") != "EV-E3":
        errors.append("confirmatory analysis evidence level must be EV-E3")
    if payload.get("decision") not in VALID_DECISIONS:
        errors.append("invalid confirmatory analysis decision")
    if payload.get("status") != STATUS:
        errors.append("confirmatory analysis status drift")
    if payload.get("confirmatory_data_consumed") is not True or payload.get("challenge_materialized") is not False:
        errors.append("confirmatory analysis evidence boundary drift")
    if payload.get("remaining_blockers") != REMAINING_BLOCKERS:
        errors.append("confirmatory analysis remaining-blocker boundary drift")

    frozen = payload.get("frozen_contract") or {}
    if frozen.get("primary_endpoint") != PRIMARY_ENDPOINT or frozen.get("primary_direction") != "higher":
        errors.append("confirmatory analysis primary contract drift")
    if not _close(frozen.get("mesi_absolute_gain"), MESI) or not _close(frozen.get("familywise_alpha"), FAMILYWISE_ALPHA):
        errors.append("confirmatory analysis MESI/alpha drift")
    if frozen.get("analysis_method") != ANALYSIS_METHOD:
        errors.append("confirmatory analysis method drift")
    if frozen.get("brier_guard") != BRIER_GUARD or frozen.get("compute_guard") != COMPUTE_GUARD:
        errors.append("confirmatory analysis protected-endpoint contract drift")
    if frozen.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("confirmatory analysis bootstrap sample-count drift")

    confirmatory_n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    rows = payload.get("raw_per_replicate_metrics") or []
    if not isinstance(confirmatory_n, int) or not (32 <= confirmatory_n <= 128):
        errors.append("confirmatory analysis n is outside frozen bounds")
    elif len(reserved) != confirmatory_n or len(rows) != confirmatory_n:
        errors.append("confirmatory analysis replicate count mismatch")
    if [row.get("replicate") for row in rows] != reserved:
        errors.append("confirmatory analysis raw replicate order drift")

    lineage = payload.get("lineage") or {}
    required_lineage = (
        "protocol_digest",
        "prep_digest",
        "paired_execution_artifact_digest",
        "reconstruction_digest",
        "raw_artifact_digest",
        "paired_checkpoint_sha256",
        "execution_contract_digest",
        "frozen_analysis_digest",
        "sample_size_freeze_digest",
        "prep_analysis_code_digest",
        "analysis_code_digest",
        "executor_code_digest",
    )
    for key in required_lineage:
        if not lineage.get(key):
            errors.append(f"missing confirmatory analysis lineage {key}")
    if lineage.get("prep_analysis_code_digest") != lineage.get("analysis_code_digest"):
        errors.append("confirmatory analysis code digest does not match frozen prep digest")
    try:
        require_canonical_stage_a_v1_digest(str(lineage.get("protocol_digest") or ""))
    except ValueError:
        errors.append("confirmatory analysis protocol digest is not canonical frozen V1")

    raw_artifact = payload.get("raw_artifact") or {}
    raw_errors = validate_exp282_confirmatory_open_raw(raw_artifact) if isinstance(raw_artifact, dict) else ["raw artifact is not an object"]
    if raw_errors:
        errors.append("embedded confirmatory raw artifact is invalid: " + "; ".join(raw_errors))
    else:
        if raw_artifact.get("artifact_digest") != lineage.get("raw_artifact_digest"):
            errors.append("confirmatory analysis raw artifact digest lineage mismatch")
        if raw_artifact.get("per_replicate") != rows:
            errors.append("confirmatory analysis raw rows do not match embedded raw artifact")
        raw_lineage = raw_artifact.get("lineage") or {}
        if raw_lineage.get("protocol_digest") != lineage.get("protocol_digest"):
            errors.append("confirmatory analysis/raw protocol lineage mismatch")
        if raw_lineage.get("execution_contract_digest") != lineage.get("execution_contract_digest"):
            errors.append("confirmatory analysis/raw execution-contract lineage mismatch")

    execution_contract = payload.get("execution_contract") or {}
    if not isinstance(execution_contract, dict) or canonical_sha256(execution_contract) != lineage.get("execution_contract_digest"):
        errors.append("confirmatory analysis execution contract digest mismatch")
    geometry = payload.get("execution_geometry") or {}
    if isinstance(execution_contract, dict):
        expected_geometry = {
            "arm_geometry": execution_contract.get("arm_geometry") or {},
            "world_geometry": execution_contract.get("world_geometry") or {},
        }
        if geometry != expected_geometry:
            errors.append("confirmatory analysis execution geometry does not match execution contract")

    expected_primary = expected_brier = expected_compute = None
    primary = payload.get("primary_effect") or {}
    try:
        effects = _paired_effects(rows)
        seed_material = (
            f"{lineage.get('protocol_digest')}|{lineage.get('frozen_analysis_digest')}|"
            f"{EXPERIMENT_ID}|paired-bootstrap-v1"
        )
        expected_primary = bootstrap_paired_effect(effects, seed_material=seed_material)
        for key in ("observed_mean", "median", "one_sided_lower_95", "one_sided_upper_95", "alpha"):
            if not _close(primary.get(key), expected_primary[key]):
                errors.append(f"confirmatory analysis primary {key} mismatch")
        for key in ("positive_count", "zero_count", "negative_count", "samples", "seed_digest"):
            if primary.get(key) != expected_primary[key]:
                errors.append(f"confirmatory analysis primary {key} mismatch")
        if primary.get("metric") != PRIMARY_ENDPOINT or primary.get("effect_type") != "absolute_gain":
            errors.append("confirmatory analysis primary metric/effect drift")
        if primary.get("baseline_arm") != "recurrent_hidden" or primary.get("candidate_arm") != "explicit_belief":
            errors.append("confirmatory analysis primary arm identity drift")
        if not _close(primary.get("mesi_absolute_gain"), MESI) or primary.get("replicates") != len(effects):
            errors.append("confirmatory analysis primary MESI/replicate drift")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid confirmatory analysis primary evidence: {exc}")

    brier = (payload.get("protected_endpoints") or {}).get("brier") or {}
    try:
        expected_brier = _brier_summary(rows)
        for key in (
            "margin",
            "recurrent_brier_sum",
            "recurrent_brier_score",
            "explicit_brier_sum",
            "explicit_brier_score",
            "delta",
        ):
            if not _close(brier.get(key), expected_brier[key]):
                errors.append(f"confirmatory analysis Brier {key} mismatch")
        for key in ("metric", "guard", "recurrent_brier_count", "explicit_brier_count", "pass"):
            if brier.get(key) != expected_brier[key]:
                errors.append(f"confirmatory analysis Brier {key} mismatch")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid confirmatory analysis Brier evidence: {exc}")

    compute = (payload.get("protected_endpoints") or {}).get("compute") or {}
    try:
        expected_compute, _ = _compute_summary(execution_contract)
        if compute.get("ledger") != expected_compute["ledger"]:
            errors.append("confirmatory analysis compute ledger mismatch")
        for key in (
            "metric",
            "guard",
            "primitive_operation_match",
            "accounted_flops_match",
            "hardware_profiler_flops_claimed",
            "pass",
        ):
            if compute.get(key) != expected_compute[key]:
                errors.append(f"confirmatory analysis compute {key} mismatch")
        for key in ("margin", "relative_accounted_flop_difference"):
            if not _close(compute.get(key), expected_compute[key]):
                errors.append(f"confirmatory analysis compute {key} mismatch")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid confirmatory analysis compute evidence: {exc}")

    if expected_primary is not None and expected_brier is not None and expected_compute is not None:
        expected_decision = _decision(
            lower=float(expected_primary["one_sided_lower_95"]),
            upper=float(expected_primary["one_sided_upper_95"]),
            brier_pass=bool(expected_brier["pass"]),
            compute_pass=bool(expected_compute["pass"]),
        )
        if payload.get("decision") != expected_decision:
            errors.append("confirmatory analysis decision does not match frozen rule")

    if payload.get("analysis_digest") not in (None, "") and payload.get("analysis_digest") != _analysis_digest(payload):
        errors.append("confirmatory analysis digest mismatch")
    return errors
