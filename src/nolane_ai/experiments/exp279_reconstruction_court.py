from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp279_confirmatory_authorization import validate_exp279_gate_a_seal
from .exp279_routing_worlds import STRATA


SCHEMA = "NLM-EXP-279-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
STATUS = "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
EXPERIMENT_ID = "EXP-279"
_FORBIDDEN_PREFREEZE_TERMS = (
    "beacon_receipt",
    "challenge_seed",
    "challenge_batch",
    "challenge_entropy",
)


def _reconstruction_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("reconstruction_digest", None)
    return canonical_sha256(clean)


def _binding_digest(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "seal_digest": payload.get("seal_digest"),
            "seal_snapshot_digest": payload.get("seal_snapshot_digest"),
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
            "checkpoint_scientific_identity_digest": payload.get(
                "checkpoint_scientific_identity_digest"
            ),
            "checkpoint_receipt_digest": payload.get(
                "checkpoint_receipt_digest"
            ),
            "checkpoint_replay_contract_digest": payload.get(
                "checkpoint_replay_contract_digest"
            ),
            "checkpoint_final_state_digest": payload.get(
                "checkpoint_final_state_digest"
            ),
            "checkpoint_file_sha256": payload.get("checkpoint_file_sha256"),
            "accounted_flops_contract": payload.get("accounted_flops_contract"),
            "model_geometry": payload.get("model_geometry"),
            "information_separation": payload.get("information_separation"),
            "reconstruction_code_digest": payload.get(
                "reconstruction_code_digest"
            ),
        }
    )


def _valid_digest(value: Any, *, size: int = 64) -> bool:
    return isinstance(value, str) and len(value) == size


def _prefreeze_errors(payload: dict[str, Any]) -> list[str]:
    rendered = repr(payload).lower()
    return [
        f"EXP-279 reconstruction contains forbidden pre-freeze material: {term}"
        for term in _FORBIDDEN_PREFREEZE_TERMS
        if term in rendered
    ]


def _sealed_authorization(seal: dict[str, Any]) -> dict[str, Any]:
    errors = validate_exp279_gate_a_seal(seal)
    if errors:
        raise ValueError("invalid EXP-279 Gate A seal: " + "; ".join(errors))
    snapshot = seal.get("authorization_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("EXP-279 Gate A seal authorization snapshot missing")
    return snapshot


def _validate_accounted_flops_contract(
    contract: Any,
    *,
    ceiling: Any,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict) or set(contract) != {
        "propagation_only",
        "branch_only",
        "hybrid",
    }:
        return ["EXP-279 reconstruction accounted FLOP contract surface invalid"]
    valid_ceiling = (
        isinstance(ceiling, int)
        and not isinstance(ceiling, bool)
        and ceiling > 0
    )
    for arm_id in ("propagation_only", "branch_only"):
        item = contract.get(arm_id) or {}
        value = item.get("accounted_flops_per_episode")
        if (
            item.get("mode") != "fixed"
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or (valid_ceiling and value > ceiling)
        ):
            errors.append(
                f"EXP-279 reconstruction {arm_id} fixed FLOP contract invalid"
            )
    hybrid = contract.get("hybrid") or {}
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
        or (valid_ceiling and branch_cost > ceiling)
    ):
        errors.append(
            "EXP-279 reconstruction hybrid route-dependent FLOP contract invalid"
        )
    return errors


def build_exp279_reconstruction_authorization(
    *,
    seal: dict[str, Any],
    reconstruction_code_digest: str,
) -> dict[str, Any]:
    sealed = _sealed_authorization(seal)
    if not _valid_digest(reconstruction_code_digest):
        raise ValueError(
            "EXP-279 reconstruction code digest must be a 64-character digest"
        )
    machinery = sealed.get("machinery_digests") or {}
    if machinery.get("reconstruction_code_digest") != reconstruction_code_digest:
        raise ValueError(
            "EXP-279 reconstruction code digest does not match Gate A machinery freeze"
        )

    checkpoint = sealed.get("checkpoint") or {}
    resource = sealed.get("resource_court") or {}
    accounted_flops = deepcopy(resource.get("accounted_flops_contract") or {})
    flop_errors = _validate_accounted_flops_contract(
        accounted_flops,
        ceiling=resource.get("declared_max_accounted_flops_per_episode"),
    )
    if flop_errors:
        raise ValueError("; ".join(flop_errors))
    if sealed.get("stratum_schedule") != list(STRATA):
        raise ValueError("EXP-279 reconstruction Gate A stratum schedule drift")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": STATUS,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "decision_rule_executed": False,
        "seal_digest": seal.get("seal_digest"),
        "seal_snapshot": deepcopy(seal),
        "seal_snapshot_digest": canonical_sha256(seal),
        "protocol_digest": sealed.get("protocol_digest"),
        "source_tree_digest": sealed.get("source_tree_digest"),
        "freeze_commit_sha": sealed.get("freeze_commit_sha"),
        "freeze_commit_timestamp_utc": sealed.get("freeze_commit_timestamp_utc"),
        "checkpoint_seal_created_at_utc": sealed.get(
            "checkpoint_seal_created_at_utc"
        ),
        "confirmatory_n": sealed.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(
            sealed.get("reserved_replicate_ids") or []
        ),
        "stratum_schedule": deepcopy(sealed.get("stratum_schedule") or []),
        "route_threshold": sealed.get("route_threshold"),
        "checkpoint_scientific_identity_digest": checkpoint.get(
            "scientific_identity_digest"
        ),
        "checkpoint_receipt_digest": checkpoint.get("receipt_digest"),
        "checkpoint_replay_contract_digest": checkpoint.get(
            "replay_contract_digest"
        ),
        "checkpoint_final_state_digest": checkpoint.get("final_state_digest"),
        "checkpoint_file_sha256": checkpoint.get("checkpoint_file_sha256"),
        "accounted_flops_contract": accounted_flops,
        "model_geometry": deepcopy(sealed.get("model_geometry") or {}),
        "information_separation": deepcopy(
            sealed.get("information_separation") or {}
        ),
        "reconstruction_code_digest": reconstruction_code_digest,
        "binding_digest": "",
        "reconstruction_digest": "",
    }
    payload["binding_digest"] = _binding_digest(payload)
    payload["reconstruction_digest"] = _reconstruction_digest(payload)
    errors = validate_exp279_reconstruction_authorization(payload)
    if errors:
        raise RuntimeError(
            "invalid EXP-279 reconstruction authorization: " + "; ".join(errors)
        )
    return payload


def validate_exp279_reconstruction_authorization(
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 reconstruction authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 reconstruction authorization cannot promote evidence")
    if payload.get("status") != STATUS:
        errors.append("EXP-279 reconstruction authorization status drift")
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(flag) is not False:
            errors.append(
                f"EXP-279 reconstruction authorization forbidden flag enabled: {flag}"
            )
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-279 reconstruction authorization cannot materialize seeds")
    errors.extend(_prefreeze_errors(payload))

    seal = payload.get("seal_snapshot")
    if not isinstance(seal, dict):
        errors.append("EXP-279 reconstruction seal snapshot missing")
        return errors
    seal_errors = validate_exp279_gate_a_seal(seal)
    if seal_errors:
        errors.append(
            "invalid EXP-279 reconstruction seal snapshot: "
            + "; ".join(seal_errors)
        )
        return errors
    if payload.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-279 reconstruction seal digest mismatch")
    if payload.get("seal_snapshot_digest") != canonical_sha256(seal):
        errors.append("EXP-279 reconstruction seal snapshot digest mismatch")

    sealed = seal.get("authorization_snapshot") or {}
    checkpoint = sealed.get("checkpoint") or {}
    resource = sealed.get("resource_court") or {}
    expected = {
        "protocol_digest": sealed.get("protocol_digest"),
        "source_tree_digest": sealed.get("source_tree_digest"),
        "freeze_commit_sha": sealed.get("freeze_commit_sha"),
        "freeze_commit_timestamp_utc": sealed.get("freeze_commit_timestamp_utc"),
        "checkpoint_seal_created_at_utc": sealed.get(
            "checkpoint_seal_created_at_utc"
        ),
        "confirmatory_n": sealed.get("confirmatory_n"),
        "reserved_replicate_ids": sealed.get("reserved_replicate_ids"),
        "stratum_schedule": sealed.get("stratum_schedule"),
        "route_threshold": sealed.get("route_threshold"),
        "checkpoint_scientific_identity_digest": checkpoint.get(
            "scientific_identity_digest"
        ),
        "checkpoint_receipt_digest": checkpoint.get("receipt_digest"),
        "checkpoint_replay_contract_digest": checkpoint.get(
            "replay_contract_digest"
        ),
        "checkpoint_final_state_digest": checkpoint.get("final_state_digest"),
        "checkpoint_file_sha256": checkpoint.get("checkpoint_file_sha256"),
        "accounted_flops_contract": resource.get("accounted_flops_contract"),
        "model_geometry": sealed.get("model_geometry"),
        "information_separation": sealed.get("information_separation"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            label = "FLOP cost" if field == "accounted_flops_contract" else field
            errors.append(
                f"EXP-279 reconstruction Gate A lineage mismatch: {label}"
            )

    if payload.get("stratum_schedule") != list(STRATA):
        errors.append("EXP-279 reconstruction stratum schedule drift")
    try:
        route_threshold = float(payload.get("route_threshold"))
        sealed_threshold = float(sealed.get("route_threshold"))
    except (TypeError, ValueError):
        route_threshold = -1.0
        sealed_threshold = -2.0
    if (
        not 0.0 <= route_threshold <= 1.0
        or abs(route_threshold - sealed_threshold) > 1e-12
    ):
        errors.append("EXP-279 reconstruction route threshold drift")

    n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(n, int) or isinstance(n, bool) or not 32 <= n <= 128:
        errors.append("EXP-279 reconstruction confirmatory n invalid")
    elif len(reserved) != n or len(set(reserved)) != n:
        errors.append("EXP-279 reconstruction reserved replicate mismatch")
    elif reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("EXP-279 reconstruction reserved replicates are not contiguous")

    errors.extend(
        _validate_accounted_flops_contract(
            payload.get("accounted_flops_contract"),
            ceiling=resource.get("declared_max_accounted_flops_per_episode"),
        )
    )

    machinery = sealed.get("machinery_digests") or {}
    if payload.get("reconstruction_code_digest") != machinery.get(
        "reconstruction_code_digest"
    ):
        errors.append("EXP-279 reconstruction machinery code digest mismatch")
    if not _valid_digest(payload.get("reconstruction_code_digest")):
        errors.append("EXP-279 reconstruction code digest invalid")

    for field in (
        "checkpoint_scientific_identity_digest",
        "checkpoint_receipt_digest",
        "checkpoint_replay_contract_digest",
        "checkpoint_final_state_digest",
        "checkpoint_file_sha256",
    ):
        if not _valid_digest(payload.get(field)):
            errors.append(f"EXP-279 reconstruction checkpoint binding invalid: {field}")

    if payload.get("binding_digest") != _binding_digest(payload):
        errors.append("EXP-279 reconstruction semantic binding digest mismatch")
    if payload.get("reconstruction_digest") != _reconstruction_digest(payload):
        errors.append("EXP-279 reconstruction authorization digest mismatch")
    return errors
