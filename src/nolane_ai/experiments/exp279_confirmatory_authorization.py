from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp279_checkpoint import validate_exp279_checkpoint_receipt
from .exp279_confirmatory_prep import (
    BOOTSTRAP_SAMPLES,
    FAMILYWISE_ALPHA,
    HOLM_THRESHOLDS,
    PLANNING_ALPHA,
    PRIMARY_CONTRASTS,
    _frozen_analysis,
    validate_exp279_confirmatory_prep,
)
from .exp279_paired_runner import validate_exp279_paired_development
from .exp279_routing_worlds import STRATA


AUTH_SCHEMA = "NLM-EXP-279-CONFIRMATORY-GATE-A-AUTH-V1"
SEAL_SCHEMA = "NLM-EXP-279-CONFIRMATORY-GATE-A-SEAL-V1"
EXPERIMENT_ID = "EXP-279"
AUTH_STATUS = "AUTHORIZED_NOT_EXECUTED"
SEAL_STATUS = "CONFIRMATORY_GATE_A_SEALED"
READINESS = "FROZEN_MACHINERY_AND_CHECKPOINT_READY_FOR_FUTURE_BEACON_ONLY"
_REQUIRED_MACHINERY = {
    "challenge_generator_digest",
    "beacon_seed_derivation_digest",
    "executor_code_digest",
    "reconstruction_code_digest",
}
_FORBIDDEN_PREFREEZE_TERMS = (
    "beacon_receipt",
    "challenge_seed",
    "challenge_batch",
    "challenge_entropy",
)
_EXPECTED_INFORMATION_RECEIPT = {
    "artifact": "constraint_variable_incidence",
    "ground_truth": True,
    "delivered_to": ["propagation_only", "hybrid"],
    "withheld_from": ["branch_only"],
    "branch_only_received_incidence": False,
}


def _authorization_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("authorization_digest", None)
    return canonical_sha256(clean)


def _seal_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("seal_digest", None)
    return canonical_sha256(clean)


def _authorization_binding(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "protocol_digest": payload.get("protocol_digest"),
            "source_tree_digest": payload.get("source_tree_digest"),
            "freeze_commit_sha": payload.get("freeze_commit_sha"),
            "freeze_commit_timestamp_utc": payload.get("freeze_commit_timestamp_utc"),
            "checkpoint_seal_created_at_utc": payload.get("checkpoint_seal_created_at_utc"),
            "development_execution_digest": payload.get("development_execution_digest"),
            "development_code_digest": payload.get("development_code_digest"),
            "arm_registry_digest": payload.get("arm_registry_digest"),
            "prep_digest": payload.get("prep_digest"),
            "frozen_analysis_digest": payload.get("frozen_analysis_digest"),
            "analysis_code_digest": payload.get("analysis_code_digest"),
            "sample_size_freeze_digest": payload.get("sample_size_freeze_digest"),
            "primary_contrasts": payload.get("primary_contrasts"),
            "familywise_alpha": payload.get("familywise_alpha"),
            "planning_alpha": payload.get("planning_alpha"),
            "holm_step_down_thresholds": payload.get("holm_step_down_thresholds"),
            "bootstrap_samples": payload.get("bootstrap_samples"),
            "pilot_best_simple_not_carried_into_confirmatory": payload.get(
                "pilot_best_simple_not_carried_into_confirmatory"
            ),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "stratum_schedule": payload.get("stratum_schedule"),
            "stratum_schedule_rule": payload.get("stratum_schedule_rule"),
            "route_threshold": payload.get("route_threshold"),
            "checkpoint": payload.get("checkpoint"),
            "model_geometry": payload.get("model_geometry"),
            "resource_court": payload.get("resource_court"),
            "information_separation": payload.get("information_separation"),
            "machinery_digests": payload.get("machinery_digests"),
        }
    )


def _seal_binding(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "authorization_digest": payload.get("authorization_digest"),
            "authorization_binding_digest": payload.get(
                "authorization_binding_digest"
            ),
            "authorization_snapshot_digest": payload.get(
                "authorization_snapshot_digest"
            ),
            "protocol_digest": payload.get("protocol_digest"),
            "source_tree_digest": payload.get("source_tree_digest"),
            "freeze_commit_sha": payload.get("freeze_commit_sha"),
            "freeze_commit_timestamp_utc": payload.get("freeze_commit_timestamp_utc"),
            "checkpoint_seal_created_at_utc": payload.get(
                "checkpoint_seal_created_at_utc"
            ),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "stratum_schedule": payload.get("stratum_schedule"),
            "route_threshold": payload.get("route_threshold"),
            "readiness": payload.get("readiness"),
        }
    )


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("timestamp is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _validate_digest_label(value: Any, *, label: str, size: int = 64) -> None:
    if not isinstance(value, str) or len(value) != size:
        raise ValueError(f"{label} must be a {size}-character digest")


def _validate_prefreeze_surface(payload: dict[str, Any], *, label: str) -> list[str]:
    rendered = repr(payload).lower()
    return [
        f"{label} contains forbidden pre-freeze material: {term}"
        for term in _FORBIDDEN_PREFREEZE_TERMS
        if term in rendered
    ]


def _validate_development_bindings(
    *,
    prep_artifact: dict[str, Any],
    development_execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    prep_errors = validate_exp279_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError(
            "invalid EXP-279 confirmatory prep: " + "; ".join(prep_errors)
        )
    if (
        prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED"
        or prep_artifact.get("confirmatory_ready") is not True
    ):
        raise ValueError("EXP-279 confirmatory prep is not executable")

    execution = development_execution_artifact
    execution_errors = validate_exp279_paired_development(execution)
    if execution_errors:
        raise ValueError(
            "invalid EXP-279 development execution: "
            + "; ".join(execution_errors)
        )
    if (
        execution.get("evidence_level") != "EV-E2"
        or execution.get("decision") != "UNVERIFIED"
    ):
        raise ValueError(
            "EXP-279 development execution must remain EV-E2 / UNVERIFIED"
        )
    if (
        execution.get("confirmatory_data_consumed") is not False
        or execution.get("challenge_materialized") is not False
        or execution.get("decision_rule_executed") is not False
    ):
        raise ValueError(
            "EXP-279 development execution consumed confirmatory/challenge data"
        )

    lineage = prep_artifact.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": execution.get("protocol_digest"),
        "development_execution_digest": execution.get("artifact_digest"),
        "development_code_digest": execution.get("code_digest"),
        "arm_registry_digest": arm_registry.get("registry_digest"),
        "pair_audit_digest": canonical_sha256(
            (execution.get("resource_match") or {}).get("pair_audit") or {}
        ),
        "route_threshold": (execution.get("route_config") or {}).get("threshold"),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(f"EXP-279 prep/development lineage mismatch: {key}")

    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-279 arm registry identity drift")
    if (
        arm_registry.get("evidence_level") != "EV-E2"
        or arm_registry.get("decision") != "UNVERIFIED"
    ):
        raise ValueError("EXP-279 arm registry must remain EV-E2 / UNVERIFIED")
    if arm_registry.get("protocol_digest") != execution.get("protocol_digest"):
        raise ValueError("EXP-279 registry/development protocol digest mismatch")
    registry_item = (arm_registry.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    if registry_item.get("development_match_status") != "PAIRED_ROUTING_DEV_READY":
        raise ValueError("EXP-279 registry development match is not ready")
    registry_execution = registry_item.get("paired_execution_evidence") or {}
    if registry_execution.get("artifact_digest") != execution.get("artifact_digest"):
        raise ValueError("EXP-279 registry/development execution digest mismatch")
    if registry_execution.get("code_digest") != execution.get("code_digest"):
        raise ValueError("EXP-279 registry/development code digest mismatch")

    resource = execution.get("resource_match") or {}
    for field in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "same_world_lineage",
        "compute_budget_closed",
    ):
        if resource.get(field) is not True:
            raise ValueError(f"EXP-279 resource court is open: {field}")
    pair_audit = resource.get("pair_audit") or {}
    if pair_audit.get("optimizer_visible_parameter_match") is not True:
        raise ValueError(
            "EXP-279 resource court optimizer-visible parameter match is open"
        )
    if pair_audit.get("structure_fit_strata") != list(STRATA):
        raise ValueError("EXP-279 resource court structure-fit strata drift")
    if resource.get("structure_fit_strata") != list(STRATA):
        raise ValueError("EXP-279 execution structure-fit strata drift")
    if resource.get("pair_audit_digest") != canonical_sha256(pair_audit):
        raise ValueError("EXP-279 resource pair-audit digest mismatch")

    route_config = execution.get("route_config") or {}
    try:
        route_threshold = float(route_config.get("threshold"))
        arm_threshold = float((execution.get("arm_geometry") or {}).get("route_threshold"))
        audit_threshold = float(pair_audit.get("route_threshold"))
        prep_threshold = float(lineage.get("route_threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("EXP-279 route threshold binding invalid") from exc
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("EXP-279 route threshold is invalid")
    if any(
        abs(item - route_threshold) > 1e-12
        for item in (arm_threshold, audit_threshold, prep_threshold)
    ):
        raise ValueError("EXP-279 route threshold binding drift")
    if (
        route_config.get("strategy")
        != "propagation_then_branch_on_residual_uncertainty"
        or route_config.get("frozen_before_evaluation") is not True
    ):
        raise ValueError("EXP-279 routing contract drift")

    information = execution.get("information_receipt") or {}
    if information != _EXPECTED_INFORMATION_RECEIPT:
        raise ValueError("EXP-279 information-separation receipt drift")

    declared = resource.get("declared_max_accounted_flops_per_episode")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        raise ValueError("EXP-279 resource court compute ceiling missing")
    ledger = pair_audit.get("compute_ledger") or {}
    if set(ledger) != {"propagation_only", "branch_only", "hybrid"}:
        raise ValueError("EXP-279 resource court compute ledger surface drift")
    for arm_id in ("propagation_only", "branch_only", "hybrid"):
        arm_ledger = ledger.get(arm_id) or {}
        maximum = arm_ledger.get("max_accounted_flops_per_episode")
        if (
            arm_ledger.get("hardware_profiler_flops_claimed") is not False
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or maximum <= 0
            or maximum > declared
        ):
            raise ValueError(f"EXP-279 resource court {arm_id} compute ledger invalid")
    hybrid_ledger = ledger["hybrid"]
    stop_cost = hybrid_ledger.get("stop_accounted_flops_per_episode")
    branch_cost = hybrid_ledger.get("branch_accounted_flops_per_episode")
    if (
        not isinstance(stop_cost, int)
        or isinstance(stop_cost, bool)
        or not isinstance(branch_cost, int)
        or isinstance(branch_cost, bool)
        or stop_cost <= 0
        or branch_cost < stop_cost
        or branch_cost > declared
    ):
        raise ValueError("EXP-279 hybrid route-dependent compute ledger invalid")
    return resource, pair_audit, information, route_threshold


def _validate_checkpoint_binding(
    *,
    checkpoint_receipt: dict[str, Any],
    development_execution_artifact: dict[str, Any],
    route_threshold: float,
) -> None:
    errors = validate_exp279_checkpoint_receipt(checkpoint_receipt)
    if errors:
        raise ValueError("invalid EXP-279 checkpoint receipt: " + "; ".join(errors))
    if checkpoint_receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("EXP-279 checkpoint identity mismatch")
    contract = checkpoint_receipt.get("replay_contract") or {}
    execution = development_execution_artifact
    if contract.get("protocol_digest") != execution.get("protocol_digest"):
        raise ValueError("EXP-279 checkpoint protocol binding mismatch")
    if contract.get("development_code_digest") != execution.get("code_digest"):
        raise ValueError("EXP-279 checkpoint development code binding mismatch")
    if contract.get("final_state") != execution.get("final_state"):
        raise ValueError("EXP-279 checkpoint final-state binding mismatch")
    if checkpoint_receipt.get("final_state_digest") != execution.get("final_state_digest"):
        raise ValueError("EXP-279 checkpoint final-state digest binding mismatch")
    if contract.get("arm_geometry") != execution.get("arm_geometry"):
        raise ValueError("EXP-279 checkpoint arm-geometry binding mismatch")
    if contract.get("world_geometry") != execution.get("world_geometry"):
        raise ValueError("EXP-279 checkpoint world-geometry binding mismatch")
    if contract.get("model_init_seed") != execution.get("model_init_seed"):
        raise ValueError("EXP-279 checkpoint model-init binding mismatch")
    if contract.get("resource_pair_audit_digest") != (
        execution.get("resource_match") or {}
    ).get("pair_audit_digest"):
        raise ValueError("EXP-279 checkpoint resource pair-audit binding mismatch")
    try:
        checkpoint_threshold = float(
            (contract.get("route_config") or {}).get("threshold")
        )
        checkpoint_arm_threshold = float(
            (contract.get("arm_geometry") or {}).get("route_threshold")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("EXP-279 checkpoint route threshold binding invalid") from exc
    if (
        abs(checkpoint_threshold - route_threshold) > 1e-12
        or abs(checkpoint_arm_threshold - route_threshold) > 1e-12
    ):
        raise ValueError("EXP-279 checkpoint route threshold binding drift")


def _accounted_flops_contract(
    *,
    pair_audit: dict[str, Any],
) -> dict[str, Any]:
    ledger = pair_audit["compute_ledger"]
    return {
        "propagation_only": {
            "mode": "fixed",
            "accounted_flops_per_episode": ledger["propagation_only"][
                "accounted_flops_per_episode"
            ],
        },
        "branch_only": {
            "mode": "fixed",
            "accounted_flops_per_episode": ledger["branch_only"][
                "accounted_flops_per_episode"
            ],
        },
        "hybrid": {
            "mode": "route_dependent",
            "stop_accounted_flops_per_episode": ledger["hybrid"][
                "stop_accounted_flops_per_episode"
            ],
            "branch_accounted_flops_per_episode": ledger["hybrid"][
                "branch_accounted_flops_per_episode"
            ],
        },
    }


def build_exp279_gate_a_authorization(
    *,
    prep_artifact: dict[str, Any],
    checkpoint_receipt: dict[str, Any],
    development_execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
    source_tree_digest: str,
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
    checkpoint_seal_created_at_utc: str,
    machinery_digests: dict[str, str],
) -> dict[str, Any]:
    resource, pair_audit, information, route_threshold = (
        _validate_development_bindings(
            prep_artifact=prep_artifact,
            development_execution_artifact=development_execution_artifact,
            arm_registry=arm_registry,
        )
    )
    _validate_checkpoint_binding(
        checkpoint_receipt=checkpoint_receipt,
        development_execution_artifact=development_execution_artifact,
        route_threshold=route_threshold,
    )
    _validate_digest_label(source_tree_digest, label="source_tree_digest")
    _validate_digest_label(freeze_commit_sha, label="freeze_commit_sha", size=40)
    if set(machinery_digests) != _REQUIRED_MACHINERY:
        raise ValueError("EXP-279 machinery digest surface drift")
    for name, digest in machinery_digests.items():
        _validate_digest_label(digest, label=name)

    try:
        freeze_time = _parse_utc(freeze_commit_timestamp_utc)
        checkpoint_time = _parse_utc(checkpoint_seal_created_at_utc)
    except ValueError as exc:
        raise ValueError("EXP-279 Gate A timestamps must be UTC ISO-8601") from exc
    if checkpoint_time <= freeze_time:
        raise ValueError(
            "EXP-279 checkpoint seal must be created strictly after Gate A freeze commit"
        )

    frozen = prep_artifact.get("frozen_analysis") or {}
    expected_frozen = _frozen_analysis()
    if frozen != expected_frozen:
        raise ValueError("EXP-279 frozen analysis contract drift")
    sample = prep_artifact.get("sample_size_freeze") or {}
    lineage = prep_artifact.get("confirmatory_lineage") or {}
    confirmatory_n = sample.get("confirmatory_n")
    reserved = list(lineage.get("reserved_replicate_ids") or [])
    if (
        not isinstance(confirmatory_n, int)
        or isinstance(confirmatory_n, bool)
        or not 32 <= confirmatory_n <= 128
    ):
        raise ValueError("EXP-279 Gate A has no executable confirmatory n")
    if len(reserved) != confirmatory_n or len(set(reserved)) != confirmatory_n:
        raise ValueError("EXP-279 reserved confirmatory lineage invalid")
    if reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        raise ValueError("EXP-279 reserved confirmatory IDs must be contiguous")
    if lineage.get("seed_materialization_status") != "NOT_EXECUTED":
        raise ValueError("EXP-279 Gate A prep already materialized challenge seeds")
    if lineage.get("stratum_schedule") != list(STRATA):
        raise ValueError("EXP-279 Gate A stratum schedule drift")

    analysis_code_digest = (prep_artifact.get("lineage") or {}).get(
        "analysis_code_digest"
    )
    _validate_digest_label(analysis_code_digest, label="analysis_code_digest")

    checkpoint_snapshot = {
        "receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "scientific_identity_digest": checkpoint_receipt.get(
            "scientific_identity_digest"
        ),
        "replay_contract_digest": checkpoint_receipt.get(
            "replay_contract_digest"
        ),
        "final_state_digest": checkpoint_receipt.get("final_state_digest"),
        "final_state": deepcopy(checkpoint_receipt.get("final_state") or {}),
        "checkpoint_file_sha256": checkpoint_receipt.get(
            "checkpoint_file_sha256"
        ),
        "state_policy": checkpoint_receipt.get("state_policy"),
    }
    resource_snapshot = {
        "parameter_match": resource.get("parameter_match"),
        "functional_parameter_match": resource.get("functional_parameter_match"),
        "active_functional_parameter_match": resource.get(
            "active_functional_parameter_match"
        ),
        "optimizer_visible_parameter_match": pair_audit.get(
            "optimizer_visible_parameter_match"
        ),
        "reclaimed_parameter_assignment_closed": resource.get(
            "reclaimed_parameter_assignment_closed"
        ),
        "same_world_lineage": resource.get("same_world_lineage"),
        "compute_budget_closed": resource.get("compute_budget_closed"),
        "structure_fit_strata": deepcopy(resource.get("structure_fit_strata")),
        "declared_max_accounted_flops_per_episode": resource.get(
            "declared_max_accounted_flops_per_episode"
        ),
        "pair_audit_digest": canonical_sha256(pair_audit),
        "route_threshold": route_threshold,
        "accounted_flops_contract": _accounted_flops_contract(
            pair_audit=pair_audit
        ),
    }
    payload: dict[str, Any] = {
        "schema": AUTH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": AUTH_STATUS,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "decision_rule_executed": False,
        "protocol_digest": development_execution_artifact.get("protocol_digest"),
        "source_tree_digest": source_tree_digest,
        "freeze_commit_sha": freeze_commit_sha,
        "freeze_commit_timestamp_utc": freeze_commit_timestamp_utc,
        "checkpoint_seal_created_at_utc": checkpoint_seal_created_at_utc,
        "development_execution_digest": development_execution_artifact.get(
            "artifact_digest"
        ),
        "development_code_digest": development_execution_artifact.get(
            "code_digest"
        ),
        "arm_registry_digest": arm_registry.get("registry_digest"),
        "prep_digest": prep_artifact.get("prep_digest"),
        "frozen_analysis_digest": canonical_sha256(frozen),
        "analysis_code_digest": analysis_code_digest,
        "sample_size_freeze_digest": canonical_sha256(sample),
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "planning_alpha": PLANNING_ALPHA,
        "holm_step_down_thresholds": list(HOLM_THRESHOLDS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "pilot_best_simple_not_carried_into_confirmatory": True,
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "stratum_schedule": list(STRATA),
        "stratum_schedule_rule": lineage.get("stratum_schedule_rule"),
        "route_threshold": route_threshold,
        "checkpoint": checkpoint_snapshot,
        "model_geometry": {
            "arm": deepcopy(development_execution_artifact.get("arm_geometry") or {}),
            "world": deepcopy(development_execution_artifact.get("world_geometry") or {}),
        },
        "resource_court": resource_snapshot,
        "information_separation": deepcopy(information),
        "machinery_digests": deepcopy(machinery_digests),
        "binding_digest": "",
        "authorization_digest": "",
    }
    payload["binding_digest"] = _authorization_binding(payload)
    payload["authorization_digest"] = _authorization_digest(payload)
    errors = validate_exp279_gate_a_authorization(payload)
    if errors:
        raise RuntimeError(
            "invalid EXP-279 Gate A authorization: " + "; ".join(errors)
        )
    return payload


def validate_exp279_gate_a_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != AUTH_SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 Gate A authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 Gate A authorization cannot promote evidence")
    if payload.get("status") != AUTH_STATUS:
        errors.append("EXP-279 Gate A authorization status drift")
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(flag) is not False:
            errors.append(f"EXP-279 Gate A authorization forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-279 Gate A authorization cannot materialize challenge seeds")
    errors.extend(
        _validate_prefreeze_surface(payload, label="EXP-279 Gate A authorization")
    )

    if payload.get("primary_contrasts") != list(PRIMARY_CONTRASTS):
        errors.append("EXP-279 Gate A primary contrast family drift")
    if payload.get("familywise_alpha") != FAMILYWISE_ALPHA:
        errors.append("EXP-279 Gate A familywise alpha drift")
    if payload.get("planning_alpha") != PLANNING_ALPHA:
        errors.append("EXP-279 Gate A planning alpha drift")
    if payload.get("holm_step_down_thresholds") != list(HOLM_THRESHOLDS):
        errors.append("EXP-279 Gate A Holm thresholds drift")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("EXP-279 Gate A bootstrap count drift")
    if payload.get("pilot_best_simple_not_carried_into_confirmatory") is not True:
        errors.append("EXP-279 Gate A pilot best-simple carryover is forbidden")
    if payload.get("stratum_schedule") != list(STRATA):
        errors.append("EXP-279 Gate A stratum schedule drift")
    try:
        threshold = float(payload.get("route_threshold"))
    except (TypeError, ValueError):
        threshold = -1.0
    if not 0.0 <= threshold <= 1.0:
        errors.append("EXP-279 Gate A route threshold invalid")

    for label, size in (
        ("analysis_code_digest", 64),
        ("source_tree_digest", 64),
        ("freeze_commit_sha", 40),
        ("protocol_digest", 64),
        ("development_execution_digest", 64),
        ("development_code_digest", 64),
        ("arm_registry_digest", 64),
        ("prep_digest", 64),
        ("frozen_analysis_digest", 64),
        ("sample_size_freeze_digest", 64),
    ):
        value = payload.get(label)
        if not isinstance(value, str) or len(value) != size:
            errors.append(f"EXP-279 Gate A {label} invalid")

    try:
        freeze_time = _parse_utc(payload.get("freeze_commit_timestamp_utc"))
        checkpoint_time = _parse_utc(payload.get("checkpoint_seal_created_at_utc"))
    except ValueError:
        errors.append("EXP-279 Gate A timestamps invalid or not UTC")
    else:
        if checkpoint_time <= freeze_time:
            errors.append("EXP-279 checkpoint seal timestamp must be after freeze commit")

    n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(n, int) or isinstance(n, bool) or not 32 <= n <= 128:
        errors.append("EXP-279 Gate A confirmatory n invalid")
    elif len(reserved) != n or len(set(reserved)) != n:
        errors.append("EXP-279 Gate A reserved replicate count/uniqueness mismatch")
    elif reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("EXP-279 Gate A reserved replicates are not contiguous")

    machinery = payload.get("machinery_digests") or {}
    if set(machinery) != _REQUIRED_MACHINERY or any(
        not isinstance(value, str) or len(value) != 64 for value in machinery.values()
    ):
        errors.append("EXP-279 Gate A machinery digest surface invalid")

    checkpoint = payload.get("checkpoint") or {}
    for field in (
        "receipt_digest",
        "scientific_identity_digest",
        "replay_contract_digest",
        "final_state_digest",
        "checkpoint_file_sha256",
    ):
        if not isinstance(checkpoint.get(field), str) or len(checkpoint.get(field, "")) != 64:
            errors.append(f"EXP-279 Gate A checkpoint binding invalid: {field}")
    final_state = checkpoint.get("final_state") or {}
    if set(final_state) != {
        "propagation_only_digest",
        "branch_only_digest",
        "hybrid_digest",
    } or any(
        not isinstance(value, str) or len(value) != 64 for value in final_state.values()
    ):
        errors.append("EXP-279 Gate A checkpoint final-state binding invalid")
    if checkpoint.get("state_policy") != "functional-only":
        errors.append("EXP-279 Gate A checkpoint state policy drift")

    resource = payload.get("resource_court") or {}
    for field in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "same_world_lineage",
        "compute_budget_closed",
    ):
        if resource.get(field) is not True:
            errors.append(f"EXP-279 Gate A resource court open: {field}")
    if resource.get("structure_fit_strata") != list(STRATA):
        errors.append("EXP-279 Gate A resource strata drift")
    if resource.get("route_threshold") != payload.get("route_threshold"):
        errors.append("EXP-279 Gate A resource/route threshold binding drift")
    ceiling = resource.get("declared_max_accounted_flops_per_episode")
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
        errors.append("EXP-279 Gate A compute ceiling invalid")
    if not isinstance(resource.get("pair_audit_digest"), str) or len(
        resource.get("pair_audit_digest", "")
    ) != 64:
        errors.append("EXP-279 Gate A pair-audit binding invalid")

    flops = resource.get("accounted_flops_contract") or {}
    if set(flops) != {"propagation_only", "branch_only", "hybrid"}:
        errors.append("EXP-279 Gate A accounted FLOP contract surface drift")
    else:
        for arm_id in ("propagation_only", "branch_only"):
            item = flops.get(arm_id) or {}
            value = item.get("accounted_flops_per_episode")
            if (
                item.get("mode") != "fixed"
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or (isinstance(ceiling, int) and value > ceiling)
            ):
                errors.append(f"EXP-279 Gate A {arm_id} fixed FLOP contract invalid")
        hybrid = flops.get("hybrid") or {}
        stop_cost = hybrid.get("stop_accounted_flops_per_episode")
        branch_cost = hybrid.get("branch_accounted_flops_per_episode")
        if (
            hybrid.get("mode") != "route_dependent"
            or not isinstance(stop_cost, int)
            or isinstance(stop_cost, bool)
            or not isinstance(branch_cost, int)
            or isinstance(branch_cost, bool)
            or stop_cost <= 0
            or branch_cost < stop_cost
            or (isinstance(ceiling, int) and branch_cost > ceiling)
        ):
            errors.append("EXP-279 Gate A hybrid route-dependent FLOP contract invalid")

    if payload.get("information_separation") != _EXPECTED_INFORMATION_RECEIPT:
        errors.append("EXP-279 Gate A information-separation drift")

    if payload.get("binding_digest") != _authorization_binding(payload):
        errors.append("EXP-279 Gate A semantic binding digest mismatch")
    if payload.get("authorization_digest") != _authorization_digest(payload):
        errors.append("EXP-279 Gate A authorization digest mismatch")
    return errors


def build_exp279_gate_a_seal(*, authorization: dict[str, Any]) -> dict[str, Any]:
    errors = validate_exp279_gate_a_authorization(authorization)
    if errors:
        raise ValueError(
            "invalid EXP-279 Gate A authorization: " + "; ".join(errors)
        )
    snapshot = deepcopy(authorization)
    payload: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": SEAL_STATUS,
        "readiness": READINESS,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "decision_rule_executed": False,
        "protocol_digest": authorization.get("protocol_digest"),
        "source_tree_digest": authorization.get("source_tree_digest"),
        "freeze_commit_sha": authorization.get("freeze_commit_sha"),
        "freeze_commit_timestamp_utc": authorization.get(
            "freeze_commit_timestamp_utc"
        ),
        "checkpoint_seal_created_at_utc": authorization.get(
            "checkpoint_seal_created_at_utc"
        ),
        "confirmatory_n": authorization.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(
            authorization.get("reserved_replicate_ids") or []
        ),
        "stratum_schedule": deepcopy(authorization.get("stratum_schedule") or []),
        "route_threshold": authorization.get("route_threshold"),
        "authorization_digest": authorization.get("authorization_digest"),
        "authorization_binding_digest": authorization.get("binding_digest"),
        "authorization_snapshot": snapshot,
        "authorization_snapshot_digest": canonical_sha256(snapshot),
        "seal_binding_digest": "",
        "seal_digest": "",
    }
    payload["seal_binding_digest"] = _seal_binding(payload)
    payload["seal_digest"] = _seal_digest(payload)
    errors = validate_exp279_gate_a_seal(payload)
    if errors:
        raise RuntimeError("invalid EXP-279 Gate A seal: " + "; ".join(errors))
    return payload


def validate_exp279_gate_a_seal(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SEAL_SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 Gate A seal identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 Gate A seal cannot promote evidence")
    if payload.get("status") != SEAL_STATUS or payload.get("readiness") != READINESS:
        errors.append("EXP-279 Gate A seal status/readiness drift")
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(flag) is not False:
            errors.append(f"EXP-279 Gate A seal forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-279 Gate A seal cannot materialize challenge seeds")
    errors.extend(_validate_prefreeze_surface(payload, label="EXP-279 Gate A seal"))

    if payload.get("stratum_schedule") != list(STRATA):
        errors.append("EXP-279 Gate A seal stratum schedule drift")
    try:
        threshold = float(payload.get("route_threshold"))
    except (TypeError, ValueError):
        threshold = -1.0
    if not 0.0 <= threshold <= 1.0:
        errors.append("EXP-279 Gate A seal route threshold invalid")

    n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(n, int) or isinstance(n, bool) or not 32 <= n <= 128:
        errors.append("EXP-279 Gate A seal confirmatory n invalid")
    elif len(reserved) != n or len(set(reserved)) != n:
        errors.append("EXP-279 Gate A seal reserved replicate mismatch")
    elif reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("EXP-279 Gate A seal reserved replicates are not contiguous")

    try:
        freeze_time = _parse_utc(payload.get("freeze_commit_timestamp_utc"))
        checkpoint_time = _parse_utc(payload.get("checkpoint_seal_created_at_utc"))
    except ValueError:
        errors.append("EXP-279 Gate A seal timestamps invalid or not UTC")
    else:
        if checkpoint_time <= freeze_time:
            errors.append("EXP-279 Gate A seal checkpoint timestamp must be after freeze")

    snapshot = payload.get("authorization_snapshot")
    if not isinstance(snapshot, dict):
        errors.append("EXP-279 Gate A seal authorization snapshot missing")
    else:
        snapshot_errors = validate_exp279_gate_a_authorization(snapshot)
        if snapshot_errors:
            errors.append(
                "EXP-279 Gate A seal authorization snapshot invalid: "
                + "; ".join(snapshot_errors)
            )
        if payload.get("authorization_snapshot_digest") != canonical_sha256(snapshot):
            errors.append("EXP-279 Gate A seal authorization snapshot digest mismatch")
        if payload.get("authorization_digest") != snapshot.get("authorization_digest"):
            errors.append("EXP-279 Gate A seal authorization digest binding mismatch")
        if payload.get("authorization_binding_digest") != snapshot.get("binding_digest"):
            errors.append("EXP-279 Gate A seal authorization binding mismatch")
        for field in (
            "protocol_digest",
            "source_tree_digest",
            "freeze_commit_sha",
            "freeze_commit_timestamp_utc",
            "checkpoint_seal_created_at_utc",
            "confirmatory_n",
            "reserved_replicate_ids",
            "stratum_schedule",
            "route_threshold",
        ):
            if payload.get(field) != snapshot.get(field):
                errors.append(f"EXP-279 Gate A seal/snapshot binding drift: {field}")

    if payload.get("seal_binding_digest") != _seal_binding(payload):
        errors.append("EXP-279 Gate A seal semantic binding digest mismatch")
    if payload.get("seal_digest") != _seal_digest(payload):
        errors.append("EXP-279 Gate A seal digest mismatch")
    return errors
