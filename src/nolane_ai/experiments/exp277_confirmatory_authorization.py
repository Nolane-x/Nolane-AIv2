from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp277_beacon import _parse_utc
from .exp277_confirmatory_prep import validate_exp277_confirmatory_prep


AUTH_SCHEMA = "NLM-EXP-277-CONFIRMATORY-GATE-A-AUTH-V1"
SEAL_SCHEMA = "NLM-EXP-277-CONFIRMATORY-GATE-A-SEAL-V1"
EXPERIMENT_ID = "EXP-277"
AUTH_STATUS = "AUTHORIZED_NOT_EXECUTED"
SEAL_STATUS = "CONFIRMATORY_GATE_A_SEALED"
READINESS = "FROZEN_MACHINERY_AND_CHECKPOINT_READY_FOR_FUTURE_BEACON_ONLY"
ENDPOINT_ALPHA = 0.025
BOOTSTRAP_SAMPLES = 10_000
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
            "sample_size_freeze_digest": payload.get("sample_size_freeze_digest"),
            "endpoint_alpha": payload.get("endpoint_alpha"),
            "bootstrap_samples": payload.get("bootstrap_samples"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "checkpoint": payload.get("checkpoint"),
            "model_geometry": payload.get("model_geometry"),
            "resource_court": payload.get("resource_court"),
            "oracle_information_separation": payload.get("oracle_information_separation"),
            "machinery_digests": payload.get("machinery_digests"),
            "wall_energy_policy": payload.get("wall_energy_policy"),
        }
    )


def _seal_binding(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "authorization_digest": payload.get("authorization_digest"),
            "authorization_binding_digest": payload.get("authorization_binding_digest"),
            "authorization_snapshot_digest": payload.get("authorization_snapshot_digest"),
            "protocol_digest": payload.get("protocol_digest"),
            "source_tree_digest": payload.get("source_tree_digest"),
            "freeze_commit_sha": payload.get("freeze_commit_sha"),
            "freeze_commit_timestamp_utc": payload.get("freeze_commit_timestamp_utc"),
            "checkpoint_seal_created_at_utc": payload.get("checkpoint_seal_created_at_utc"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "readiness": payload.get("readiness"),
        }
    )


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


def _per_arm_flops_from_resource(
    resource: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[dict[str, int], str]:
    pair_audit = resource.get("pair_audit") or {}
    ledger = pair_audit.get("compute_ledger") or {}
    values: dict[str, int] = {}
    if isinstance(ledger, dict):
        for arm in ("arcs_branch", "oracle_cbrf"):
            arm_ledger = ledger.get(arm) or {}
            value = arm_ledger.get("accounted_flops_per_episode")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                values[arm] = value
    if set(values) == {"arcs_branch", "oracle_cbrf"}:
        return values, "pair_audit_compute_ledger"

    rows = list((execution.get("evaluation") or {}).get("per_replicate") or [])
    fallback: dict[str, int] = {}
    for arm in ("arcs_branch", "oracle_cbrf"):
        observed = {
            row.get(arm, {}).get("accounted_flops_per_episode")
            for row in rows
            if isinstance(row, dict)
        }
        if (
            len(observed) == 1
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in observed)
        ):
            fallback[arm] = int(next(iter(observed)))
    if set(fallback) != {"arcs_branch", "oracle_cbrf"}:
        raise ValueError("EXP-277 resource court per-arm accounted FLOPs missing")
    return fallback, "development_evaluation_consistent_fallback"


def _validate_development_bindings(
    *,
    prep_artifact: dict[str, Any],
    development_execution_artifact: dict[str, Any],
    arm_registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prep_errors = validate_exp277_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid EXP-277 confirmatory prep: " + "; ".join(prep_errors))
    if prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED" or prep_artifact.get("confirmatory_ready") is not True:
        raise ValueError("EXP-277 confirmatory prep is not executable")

    execution = development_execution_artifact
    if execution.get("schema") != "NLM-EXP-277-PAIRED-DEV-EVAL-V1":
        raise ValueError("EXP-277 development execution identity drift")
    if execution.get("evidence_level") != "EV-E2" or execution.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-277 development execution must remain EV-E2 / UNVERIFIED")
    if execution.get("confirmatory_data_consumed") is not False or execution.get("challenge_materialized") is not False:
        raise ValueError("EXP-277 development execution consumed confirmatory/challenge data")

    lineage = prep_artifact.get("lineage") or {}
    if execution.get("artifact_digest") != lineage.get("execution_artifact_digest"):
        raise ValueError("EXP-277 development execution digest mismatch")
    if execution.get("code_digest") != lineage.get("development_code_digest"):
        raise ValueError("EXP-277 development code digest mismatch")
    if execution.get("protocol_digest") != lineage.get("protocol_digest"):
        raise ValueError("EXP-277 development protocol digest mismatch")

    if arm_registry.get("schema") != "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1":
        raise ValueError("EXP-277 arm registry identity drift")
    if arm_registry.get("registry_digest") != lineage.get("arm_registry_digest"):
        raise ValueError("EXP-277 arm registry digest mismatch")
    if arm_registry.get("protocol_digest") != execution.get("protocol_digest"):
        raise ValueError("EXP-277 registry protocol digest mismatch")
    registry_exp = (arm_registry.get("experiments") or {}).get("EXP-277") or {}
    registry_execution = registry_exp.get("paired_execution_evidence") or {}
    if registry_execution.get("artifact_digest") != execution.get("artifact_digest"):
        raise ValueError("EXP-277 registry/development execution digest mismatch")
    if registry_execution.get("code_digest") not in (None, execution.get("code_digest")):
        raise ValueError("EXP-277 registry/development code digest mismatch")

    resource = execution.get("resource_match") or {}
    for field in ("parameter_match", "functional_parameter_match", "same_world_lineage", "compute_budget_closed"):
        if resource.get(field) is not True:
            raise ValueError(f"EXP-277 resource court is open: {field}")
    pair_audit = resource.get("pair_audit") or {}
    if pair_audit.get("oracle_information_separation") is not True:
        raise ValueError("EXP-277 resource court oracle-information separation is open")
    if pair_audit.get("compute_budget_closed") is not True:
        raise ValueError("EXP-277 resource court compute budget is open")
    declared = resource.get("declared_max_accounted_flops_per_episode")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        raise ValueError("EXP-277 resource court compute ceiling missing")
    arm_flops, _ = _per_arm_flops_from_resource(resource, execution)
    if any(value > declared for value in arm_flops.values()):
        raise ValueError("EXP-277 resource court per-arm FLOPs exceed declared ceiling")

    oracle = execution.get("oracle_information_receipt") or {}
    expected_oracle = {
        "artifact": "oracle_incidence",
        "ground_truth": True,
        "delivered_to": ["oracle_cbrf"],
        "withheld_from": ["arcs_branch"],
        "arcs_received_oracle_incidence": False,
    }
    if oracle != expected_oracle:
        raise ValueError("EXP-277 oracle-information separation drift")
    return resource, oracle


def _validate_checkpoint_binding(
    *,
    checkpoint_receipt: dict[str, Any],
    development_execution_artifact: dict[str, Any],
) -> None:
    from .exp277_checkpoint import validate_exp277_checkpoint_receipt

    errors = validate_exp277_checkpoint_receipt(checkpoint_receipt)
    if errors:
        raise ValueError("invalid EXP-277 checkpoint receipt: " + "; ".join(errors))
    if checkpoint_receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("EXP-277 checkpoint identity mismatch")
    contract = checkpoint_receipt.get("execution_contract") or {}
    execution = development_execution_artifact
    if contract.get("protocol_digest") != execution.get("protocol_digest"):
        raise ValueError("EXP-277 checkpoint protocol binding mismatch")
    if contract.get("development_code_digest") != execution.get("code_digest"):
        raise ValueError("EXP-277 checkpoint development code binding mismatch")
    if contract.get("development_final_state") != execution.get("final_state"):
        raise ValueError("EXP-277 checkpoint final-state binding mismatch")
    if contract.get("arm_geometry") != execution.get("arm_geometry") or contract.get("world_geometry") != execution.get("world_geometry"):
        raise ValueError("EXP-277 checkpoint geometry binding mismatch")
    if contract.get("model_init_seed") != execution.get("model_init_seed"):
        raise ValueError("EXP-277 checkpoint model-init binding mismatch")


def build_exp277_gate_a_authorization(
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
    resource, oracle = _validate_development_bindings(
        prep_artifact=prep_artifact,
        development_execution_artifact=development_execution_artifact,
        arm_registry=arm_registry,
    )
    _validate_checkpoint_binding(
        checkpoint_receipt=checkpoint_receipt,
        development_execution_artifact=development_execution_artifact,
    )
    _validate_digest_label(source_tree_digest, label="source_tree_digest")
    _validate_digest_label(freeze_commit_sha, label="freeze_commit_sha", size=40)
    if set(machinery_digests) != _REQUIRED_MACHINERY:
        raise ValueError("EXP-277 machinery digest surface drift")
    for name, digest in machinery_digests.items():
        _validate_digest_label(digest, label=name)

    try:
        freeze_time = _parse_utc(freeze_commit_timestamp_utc)
        checkpoint_time = _parse_utc(checkpoint_seal_created_at_utc)
    except ValueError as exc:
        raise ValueError("EXP-277 Gate A timestamps must be UTC ISO-8601") from exc
    if checkpoint_time <= freeze_time:
        raise ValueError("EXP-277 checkpoint seal must be created strictly after Gate A freeze commit")

    frozen = prep_artifact.get("frozen_analysis") or {}
    if float(frozen.get("endpoint_alpha", -1.0)) != ENDPOINT_ALPHA:
        raise ValueError("EXP-277 frozen endpoint alpha drift")
    if int(frozen.get("bootstrap_samples", -1)) != BOOTSTRAP_SAMPLES:
        raise ValueError("EXP-277 frozen bootstrap count drift")
    sample = prep_artifact.get("sample_size_freeze") or {}
    confirmatory_n = sample.get("confirmatory_n")
    reserved = list((prep_artifact.get("confirmatory_lineage") or {}).get("reserved_replicate_ids") or [])
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not (32 <= confirmatory_n <= 128):
        raise ValueError("EXP-277 Gate A has no executable confirmatory n")
    if len(reserved) != confirmatory_n or len(set(reserved)) != confirmatory_n:
        raise ValueError("EXP-277 reserved confirmatory lineage invalid")
    if reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        raise ValueError("EXP-277 reserved confirmatory IDs must be contiguous")
    if (prep_artifact.get("confirmatory_lineage") or {}).get("seed_materialization_status") != "NOT_EXECUTED":
        raise ValueError("EXP-277 Gate A prep already materialized challenge seeds")

    checkpoint_snapshot = {
        "receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "execution_contract_digest": checkpoint_receipt.get("execution_contract_digest"),
        "arcs_branch_final_digest": checkpoint_receipt.get("arcs_branch_final_digest"),
        "oracle_cbrf_final_digest": checkpoint_receipt.get("oracle_cbrf_final_digest"),
        "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
        "state_policy": checkpoint_receipt.get("state_policy"),
    }
    arm_flops, arm_flops_source = _per_arm_flops_from_resource(resource, development_execution_artifact)
    resource_snapshot = {
        "parameter_match": resource.get("parameter_match"),
        "functional_parameter_match": resource.get("functional_parameter_match"),
        "same_world_lineage": resource.get("same_world_lineage"),
        "compute_budget_closed": resource.get("compute_budget_closed"),
        "declared_max_accounted_flops_per_episode": resource.get("declared_max_accounted_flops_per_episode"),
        "arm_accounted_flops_per_episode": arm_flops,
        "arm_accounted_flops_source": arm_flops_source,
        "pair_audit_digest": canonical_sha256(resource.get("pair_audit") or {}),
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
        "development_execution_digest": development_execution_artifact.get("artifact_digest"),
        "development_code_digest": development_execution_artifact.get("code_digest"),
        "arm_registry_digest": arm_registry.get("registry_digest"),
        "prep_digest": prep_artifact.get("prep_digest"),
        "frozen_analysis_digest": canonical_sha256(frozen),
        "sample_size_freeze_digest": canonical_sha256(sample),
        "endpoint_alpha": ENDPOINT_ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "checkpoint": checkpoint_snapshot,
        "model_geometry": {
            "arm": deepcopy(development_execution_artifact.get("arm_geometry") or {}),
            "world": deepcopy(development_execution_artifact.get("world_geometry") or {}),
        },
        "resource_court": resource_snapshot,
        "oracle_information_separation": deepcopy(oracle),
        "machinery_digests": deepcopy(machinery_digests),
        "wall_energy_policy": {
            "metric": "wall_energy_per_episode",
            "stage_a_role": "report-only",
            "fabrication_forbidden": True,
            "decision_use": False,
        },
        "binding_digest": "",
        "authorization_digest": "",
    }
    payload["binding_digest"] = _authorization_binding(payload)
    payload["authorization_digest"] = _authorization_digest(payload)
    errors = validate_exp277_gate_a_authorization(payload)
    if errors:
        raise RuntimeError("invalid EXP-277 Gate A authorization: " + "; ".join(errors))
    return payload


def validate_exp277_gate_a_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != AUTH_SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-277 Gate A authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-277 Gate A authorization cannot promote evidence")
    if payload.get("status") != AUTH_STATUS:
        errors.append("EXP-277 Gate A authorization status drift")
    for flag in ("confirmatory_ready", "confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(flag) is not False:
            errors.append(f"EXP-277 Gate A authorization forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-277 Gate A authorization cannot materialize challenge seeds")
    errors.extend(_validate_prefreeze_surface(payload, label="EXP-277 Gate A authorization"))

    if payload.get("endpoint_alpha") != ENDPOINT_ALPHA:
        errors.append("EXP-277 Gate A endpoint alpha drift")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("EXP-277 Gate A bootstrap count drift")
    if not isinstance(payload.get("source_tree_digest"), str) or len(payload.get("source_tree_digest", "")) != 64:
        errors.append("EXP-277 Gate A source-tree digest invalid")
    if not isinstance(payload.get("freeze_commit_sha"), str) or len(payload.get("freeze_commit_sha", "")) != 40:
        errors.append("EXP-277 Gate A freeze commit SHA invalid")
    try:
        freeze_time = _parse_utc(payload.get("freeze_commit_timestamp_utc"))
        checkpoint_time = _parse_utc(payload.get("checkpoint_seal_created_at_utc"))
    except ValueError:
        errors.append("EXP-277 Gate A timestamps invalid or not UTC")
    else:
        if checkpoint_time <= freeze_time:
            errors.append("EXP-277 checkpoint seal timestamp must be after freeze commit")

    n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(n, int) or isinstance(n, bool) or not (32 <= n <= 128):
        errors.append("EXP-277 Gate A confirmatory n invalid")
    elif len(reserved) != n or len(set(reserved)) != n:
        errors.append("EXP-277 Gate A reserved replicate count/uniqueness mismatch")
    elif reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("EXP-277 Gate A reserved replicates are not contiguous")

    machinery = payload.get("machinery_digests") or {}
    if set(machinery) != _REQUIRED_MACHINERY or any(
        not isinstance(value, str) or len(value) != 64 for value in machinery.values()
    ):
        errors.append("EXP-277 Gate A machinery digest surface invalid")

    checkpoint = payload.get("checkpoint") or {}
    for field in (
        "receipt_digest",
        "scientific_identity_digest",
        "execution_contract_digest",
        "arcs_branch_final_digest",
        "oracle_cbrf_final_digest",
        "checkpoint_file_sha256",
    ):
        if not isinstance(checkpoint.get(field), str) or not checkpoint.get(field):
            errors.append(f"EXP-277 Gate A checkpoint binding missing: {field}")
    if checkpoint.get("state_policy") != "functional-only":
        errors.append("EXP-277 Gate A checkpoint state policy drift")

    resource = payload.get("resource_court") or {}
    for field in ("parameter_match", "functional_parameter_match", "same_world_lineage", "compute_budget_closed"):
        if resource.get(field) is not True:
            errors.append(f"EXP-277 Gate A resource court open: {field}")
    ceiling = resource.get("declared_max_accounted_flops_per_episode")
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
        errors.append("EXP-277 Gate A resource court compute ceiling invalid")
    arm_flops = resource.get("arm_accounted_flops_per_episode")
    if not isinstance(arm_flops, dict) or set(arm_flops) != {"arcs_branch", "oracle_cbrf"}:
        errors.append("EXP-277 Gate A per-arm FLOP binding missing")
    else:
        for arm, value in arm_flops.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"EXP-277 Gate A {arm} accounted FLOPs invalid")
            elif isinstance(ceiling, int) and value > ceiling:
                errors.append(f"EXP-277 Gate A {arm} accounted FLOPs exceed ceiling")
    if resource.get("arm_accounted_flops_source") not in {
        "pair_audit_compute_ledger",
        "development_evaluation_consistent_fallback",
    }:
        errors.append("EXP-277 Gate A per-arm FLOP source invalid")
    if not isinstance(resource.get("pair_audit_digest"), str) or not resource.get("pair_audit_digest"):
        errors.append("EXP-277 Gate A pair-audit binding missing")

    oracle = payload.get("oracle_information_separation") or {}
    if (
        oracle.get("arcs_received_oracle_incidence") is not False
        or oracle.get("delivered_to") != ["oracle_cbrf"]
        or oracle.get("withheld_from") != ["arcs_branch"]
    ):
        errors.append("EXP-277 Gate A oracle-information separation drift")

    wall = payload.get("wall_energy_policy") or {}
    if wall != {
        "metric": "wall_energy_per_episode",
        "stage_a_role": "report-only",
        "fabrication_forbidden": True,
        "decision_use": False,
    }:
        errors.append("EXP-277 Gate A wall-energy policy drift")

    if payload.get("binding_digest") != _authorization_binding(payload):
        errors.append("EXP-277 Gate A semantic binding digest mismatch")
    if payload.get("authorization_digest") != _authorization_digest(payload):
        errors.append("EXP-277 Gate A authorization digest mismatch")
    return errors


def build_exp277_gate_a_seal(*, authorization: dict[str, Any]) -> dict[str, Any]:
    errors = validate_exp277_gate_a_authorization(authorization)
    if errors:
        raise ValueError("invalid EXP-277 Gate A authorization: " + "; ".join(errors))
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
        "freeze_commit_timestamp_utc": authorization.get("freeze_commit_timestamp_utc"),
        "checkpoint_seal_created_at_utc": authorization.get("checkpoint_seal_created_at_utc"),
        "confirmatory_n": authorization.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(authorization.get("reserved_replicate_ids") or []),
        "authorization_digest": authorization.get("authorization_digest"),
        "authorization_binding_digest": authorization.get("binding_digest"),
        "authorization_snapshot": snapshot,
        "authorization_snapshot_digest": canonical_sha256(snapshot),
        "seal_binding_digest": "",
        "seal_digest": "",
    }
    payload["seal_binding_digest"] = _seal_binding(payload)
    payload["seal_digest"] = _seal_digest(payload)
    errors = validate_exp277_gate_a_seal(payload)
    if errors:
        raise RuntimeError("invalid EXP-277 Gate A seal: " + "; ".join(errors))
    return payload


def validate_exp277_gate_a_seal(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SEAL_SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-277 Gate A seal identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-277 Gate A seal cannot promote evidence")
    if payload.get("status") != SEAL_STATUS or payload.get("readiness") != READINESS:
        errors.append("EXP-277 Gate A seal status/readiness drift")
    for flag in ("confirmatory_ready", "confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(flag) is not False:
            errors.append(f"EXP-277 Gate A seal forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-277 Gate A seal cannot materialize challenge seeds")
    errors.extend(_validate_prefreeze_surface(payload, label="EXP-277 Gate A seal"))

    snapshot = payload.get("authorization_snapshot")
    if not isinstance(snapshot, dict):
        errors.append("EXP-277 Gate A seal authorization snapshot missing")
        return errors
    authorization_errors = validate_exp277_gate_a_authorization(snapshot)
    if authorization_errors:
        errors.append("invalid sealed EXP-277 authorization: " + "; ".join(authorization_errors))
    if payload.get("authorization_snapshot_digest") != canonical_sha256(snapshot):
        errors.append("EXP-277 Gate A seal authorization snapshot digest mismatch")
    if payload.get("authorization_digest") != snapshot.get("authorization_digest"):
        errors.append("EXP-277 Gate A seal authorization digest mismatch")
    if payload.get("authorization_binding_digest") != snapshot.get("binding_digest"):
        errors.append("EXP-277 Gate A seal authorization binding mismatch")

    for field in (
        "protocol_digest",
        "source_tree_digest",
        "freeze_commit_sha",
        "freeze_commit_timestamp_utc",
        "checkpoint_seal_created_at_utc",
        "confirmatory_n",
        "reserved_replicate_ids",
    ):
        if payload.get(field) != snapshot.get(field):
            errors.append(f"EXP-277 Gate A seal snapshot mismatch: {field}")
    if payload.get("seal_binding_digest") != _seal_binding(payload):
        errors.append("EXP-277 Gate A seal binding digest mismatch")
    if payload.get("seal_digest") != _seal_digest(payload):
        errors.append("EXP-277 Gate A seal digest mismatch")
    return errors
