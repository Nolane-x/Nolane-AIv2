from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXP301_SUPERSEDED_V1_DIGEST = "110701d1054fe0743bac2b65d16faa602182a1d90ab85bdc7d3d426de0212253"
EXP301_EXPECTED_DIGEST = "1660a728990290f1c605941d531be1d52fe82591c619a327d104e4b87c1e6bad"
EXP301_SCHEMA_VERSION = "nlm-v017-prereg-v2"
EXP301_ARM_IDS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EXP301_ROOTS = (0, 1, 2, 3)
EXP301_TRAINING_LOOPS = (1, 2, 4, 8)
EXP301_CHALLENGE_LOOPS = (1, 2, 4, 8, 12, 16)
EXP301_PARAMETER_CEILING = 10_000_000
EXP301_AUTHORITY_PARENT = "a7d14a97ae74ace9a0f4be4e72fade434cbaa96b"
EXP301_MAX_RELATIVE_FLOP_MISMATCH = 0.002


def canonical_registration_payload(payload: Mapping[str, Any]) -> bytes:
    scientific_payload = dict(payload)
    scientific_payload.pop("registration_digest", None)
    return json.dumps(
        scientific_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def registration_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_registration_payload(payload)).hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"EXP-301 {label} drifted: expected {expected!r}, got {actual!r}")


def validate_exp301_registration(payload: Mapping[str, Any]) -> None:
    _require_equal("schema_version", payload.get("schema_version"), EXP301_SCHEMA_VERSION)
    _require_equal("experiment_id", payload.get("experiment_id"), "EXP-301")

    # The immutable scientific digest boundary comes before most field-level
    # diagnostics. A mutation that keeps/recomputes its own self-digest is not
    # allowed to redefine the frozen V2 authority.
    declared = payload.get("registration_digest")
    _require_equal("registration_digest", declared, f"sha256:{EXP301_EXPECTED_DIGEST}")
    actual_digest = registration_digest(payload)
    if actual_digest != EXP301_EXPECTED_DIGEST:
        raise ValueError(
            "EXP-301 registration digest mismatch: "
            f"expected {EXP301_EXPECTED_DIGEST}, got {actual_digest}"
        )

    authority = payload.get("authority_parent")
    if not isinstance(authority, Mapping):
        raise ValueError("EXP-301 authority_parent must be an object")
    _require_equal("authority_parent.main", authority.get("main"), EXP301_AUTHORITY_PARENT)

    amendment = payload.get("protocol_amendment")
    if not isinstance(amendment, Mapping):
        raise ValueError("EXP-301 protocol_amendment must be an object")
    _require_equal(
        "protocol_amendment.amendment_kind",
        amendment.get("amendment_kind"),
        "PRE_DATA_FAIRNESS_CORRECTION",
    )
    _require_equal(
        "protocol_amendment.supersedes_registration_digest",
        amendment.get("supersedes_registration_digest"),
        f"sha256:{EXP301_SUPERSEDED_V1_DIGEST}",
    )
    _require_equal("protocol_amendment.scientific_outcome_consumed", amendment.get("scientific_outcome_consumed"), False)
    _require_equal("protocol_amendment.challenge_materialized", amendment.get("challenge_materialized"), False)
    _require_equal("protocol_amendment.hypothesis_changed", amendment.get("hypothesis_changed"), False)

    architecture = payload.get("architecture_candidate")
    if not isinstance(architecture, Mapping):
        raise ValueError("EXP-301 architecture_candidate must be an object")
    _require_equal("max_resident_parameters", architecture.get("max_resident_parameters"), EXP301_PARAMETER_CEILING)
    _require_equal("core_parameter_count_before_capacity_exchange", architecture.get("core_parameter_count_before_capacity_exchange"), 9_120_832)
    _require_equal("capacity_exchange_envelope", architecture.get("capacity_exchange_envelope"), 879_168)
    _require_equal("vocab_size", architecture.get("vocab_size"), 4_608)
    _require_equal("d_model", architecture.get("d_model"), 448)
    _require_equal("n_heads", architecture.get("n_heads"), 7)
    _require_equal("head_dim", architecture.get("head_dim"), 64)
    _require_equal("shared_layers", architecture.get("shared_layers"), 3)
    _require_equal("d_ff", architecture.get("d_ff"), 1_152)

    arms = payload.get("arms")
    if not isinstance(arms, list):
        raise ValueError("EXP-301 arms must be a list")
    _require_equal("arm ids", tuple(arm.get("id") for arm in arms), EXP301_ARM_IDS)

    compute = payload.get("compute_matching")
    if not isinstance(compute, Mapping):
        raise ValueError("EXP-301 compute_matching must be an object")
    _require_equal("compute effort multipliers", tuple(compute.get("effort_multipliers", ())), EXP301_CHALLENGE_LOOPS)
    _require_equal("max relative FLOP mismatch", compute.get("max_relative_flop_mismatch"), EXP301_MAX_RELATIVE_FLOP_MISMATCH)
    _require_equal(
        "unmatched budget policy",
        compute.get("unmatched_budget_policy"),
        "INVALID_COMPUTE_MATCH for all arms at that operating point; never interpolate or repair after outcome",
    )

    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("EXP-301 training must be an object")
    _require_equal("roots", tuple(training.get("roots", ())), EXP301_ROOTS)
    _require_equal("training loops", tuple(training.get("loop_training_distribution", ())), EXP301_TRAINING_LOOPS)
    _require_equal("challenge loops", tuple(training.get("challenge_loop_budgets", ())), EXP301_CHALLENGE_LOOPS)


def load_exp301_registration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EXP-301 registration root must be an object")
    validate_exp301_registration(payload)
    return payload
