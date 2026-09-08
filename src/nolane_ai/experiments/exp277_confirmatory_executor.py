from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp277_beacon import validate_exp277_beacon_receipt
from .exp277_checkpoint import validate_exp277_checkpoint_receipt
from .exp277_confirmatory_authorization import validate_exp277_gate_a_seal
from .exp277_reconstruction_court import (
    _row_digest,
    reconstruct_exp277_expected_row,
    validate_exp277_raw_row_against_reconstruction,
    validate_exp277_reconstruction_authorization,
)


SCHEMA = "NLM-EXP-277-CONFIRMATORY-CHALLENGE-RAW-V1"
EXPERIMENT_ID = "EXP-277"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
INVALID_STATUS = "INVALID_RUN"
CHALLENGE_STREAM = "challenge"


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _arm_input_receipt() -> dict[str, bool]:
    return {
        "same_surface_events": True,
        "same_variable_states": True,
        "arcs_received_oracle_incidence": False,
        "oracle_cbrf_received_oracle_incidence": True,
        "evaluator_targets_withheld_from_arms": True,
    }


def _analytical_cost_receipt(accounted_flops: int) -> dict[str, Any]:
    return {
        "accounting_semantics": "analytical scalar arithmetic FLOPs for frozen neural geometry; not hardware-profiler FLOPs",
        "accounted_flops_per_episode": int(accounted_flops),
        "hardware_profiler_flops_claimed": False,
    }


def _invalid_artifact(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
    integrity_errors: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "INVALID_RUN",
        "status": INVALID_STATUS,
        "test_only": bool(beacon_receipt.get("test_only") is True),
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "synthetic_challenge_data_consumed": False,
        "challenge_materialized": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "decision_rule_executed": False,
        "confirmatory_n": reconstruction_authorization.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(reconstruction_authorization.get("reserved_replicate_ids") or []),
        "challenge_stream": CHALLENGE_STREAM,
        "per_replicate": [],
        "integrity_errors": list(integrity_errors),
        "current_source_tree_digest": current_source_tree_digest,
        "executor_code_digest": executor_code_digest,
        "seal_digest": seal.get("seal_digest"),
        "reconstruction_digest": reconstruction_authorization.get("reconstruction_digest"),
        "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def _preflight_integrity_errors(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> list[str]:
    errors: list[str] = []

    reconstruction_errors = validate_exp277_reconstruction_authorization(reconstruction_authorization)
    if reconstruction_errors:
        errors.append("EXP-277 reconstruction authorization invalid: " + "; ".join(reconstruction_errors))

    seal_errors = validate_exp277_gate_a_seal(seal)
    if seal_errors:
        errors.append("EXP-277 Gate A seal invalid: " + "; ".join(seal_errors))

    if reconstruction_authorization.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-277 reconstruction/seal lineage mismatch")

    frozen_source = reconstruction_authorization.get("source_tree_digest")
    if current_source_tree_digest != frozen_source:
        errors.append("EXP-277 current source tree digest does not match frozen source tree")

    sealed = seal.get("authorization_snapshot") or {}
    machinery = sealed.get("machinery_digests") or {}
    if executor_code_digest != machinery.get("executor_code_digest"):
        errors.append("EXP-277 executor code digest does not match frozen executor machinery")

    checkpoint_errors = validate_exp277_checkpoint_receipt(checkpoint_receipt)
    if checkpoint_errors:
        errors.append("EXP-277 checkpoint receipt invalid: " + "; ".join(checkpoint_errors))
    sealed_checkpoint = sealed.get("checkpoint") or {}
    for field in (
        "receipt_digest",
        "scientific_identity_digest",
        "execution_contract_digest",
        "arcs_branch_final_digest",
        "oracle_cbrf_final_digest",
        "checkpoint_file_sha256",
        "state_policy",
    ):
        if checkpoint_receipt.get(field) != sealed_checkpoint.get(field):
            errors.append(f"EXP-277 checkpoint Gate A binding mismatch: {field}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        errors.append("EXP-277 checkpoint file is missing")

    beacon_errors = validate_exp277_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=reconstruction_authorization.get("freeze_commit_timestamp_utc"),
        checkpoint_seal_created_at_utc=reconstruction_authorization.get("checkpoint_seal_created_at_utc"),
    )
    if beacon_errors:
        errors.append("EXP-277 beacon receipt invalid: " + "; ".join(beacon_errors))

    reserved = reconstruction_authorization.get("reserved_replicate_ids") or []
    confirmatory_n = reconstruction_authorization.get("confirmatory_n")
    if (
        not isinstance(confirmatory_n, int)
        or isinstance(confirmatory_n, bool)
        or not 32 <= confirmatory_n <= 128
        or not isinstance(reserved, list)
        or len(reserved) != confirmatory_n
        or len(set(reserved)) != confirmatory_n
        or (reserved and reserved != list(range(reserved[0], reserved[0] + len(reserved))))
    ):
        errors.append("EXP-277 frozen confirmatory replicate lineage invalid")

    return errors


def execute_exp277_confirmatory_challenge(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> dict[str, Any]:
    errors = _preflight_integrity_errors(
        reconstruction_authorization=reconstruction_authorization,
        seal=seal,
        beacon_receipt=beacon_receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source_tree_digest,
        executor_code_digest=executor_code_digest,
    )
    if errors:
        return _invalid_artifact(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            current_source_tree_digest=current_source_tree_digest,
            executor_code_digest=executor_code_digest,
            integrity_errors=errors,
        )

    reserved = list(reconstruction_authorization["reserved_replicate_ids"])
    arm_flops = reconstruction_authorization["arm_accounted_flops_per_episode"]
    rows: list[dict[str, Any]] = []
    for replicate in reserved:
        try:
            row = reconstruct_exp277_expected_row(
                reconstruction_authorization=reconstruction_authorization,
                seal=seal,
                beacon_receipt=beacon_receipt,
                checkpoint_path=checkpoint_path,
                checkpoint_receipt=checkpoint_receipt,
                replicate=replicate,
            )
        except (ValueError, RuntimeError) as exc:
            return _invalid_artifact(
                reconstruction_authorization=reconstruction_authorization,
                seal=seal,
                beacon_receipt=beacon_receipt,
                current_source_tree_digest=current_source_tree_digest,
                executor_code_digest=executor_code_digest,
                integrity_errors=[f"EXP-277 confirmatory row reconstruction failed before publication: {exc}"],
            )
        row["arm_input_receipt"] = _arm_input_receipt()
        for arm in ("arcs_branch", "oracle_cbrf"):
            row[arm]["analytical_cost_receipt"] = _analytical_cost_receipt(int(arm_flops[arm]))
        row["row_digest"] = _row_digest(row)
        rows.append(row)

    test_only = beacon_receipt.get("test_only") is True
    scientific_eligible = bool(beacon_receipt.get("scientific_evidence_eligible") is True and not test_only)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": scientific_eligible,
        "confirmatory_data_consumed": bool(scientific_eligible),
        "synthetic_challenge_data_consumed": bool(test_only),
        "challenge_materialized": True,
        "seed_materialization_status": "TEST_ONLY_EXECUTED" if test_only else "EXECUTED",
        "decision_rule_executed": False,
        "confirmatory_n": len(reserved),
        "reserved_replicate_ids": reserved,
        "challenge_stream": CHALLENGE_STREAM,
        "per_replicate": rows,
        "integrity_errors": [],
        "current_source_tree_digest": current_source_tree_digest,
        "executor_code_digest": executor_code_digest,
        "seal": deepcopy(seal),
        "reconstruction_authorization": deepcopy(reconstruction_authorization),
        "beacon_receipt": deepcopy(beacon_receipt),
        "lineage": {
            "protocol_digest": reconstruction_authorization.get("protocol_digest"),
            "source_tree_digest": reconstruction_authorization.get("source_tree_digest"),
            "seal_digest": seal.get("seal_digest"),
            "reconstruction_digest": reconstruction_authorization.get("reconstruction_digest"),
            "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
            "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "executor_code_digest": executor_code_digest,
        },
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    validation_errors = validate_exp277_confirmatory_raw(
        payload,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if validation_errors:
        return _invalid_artifact(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            current_source_tree_digest=current_source_tree_digest,
            executor_code_digest=executor_code_digest,
            integrity_errors=validation_errors,
        )
    return payload


def validate_exp277_confirmatory_raw(
    payload: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_receipt: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-277 raw artifact must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-277 confirmatory raw identity")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-277 confirmatory raw artifact digest mismatch")

    status = payload.get("status")
    decision = payload.get("decision")
    if status == INVALID_STATUS or decision == "INVALID_RUN":
        if status != INVALID_STATUS or decision != "INVALID_RUN":
            errors.append("EXP-277 invalid raw status/decision mismatch")
        if payload.get("scientific_evidence_eligible") is not False:
            errors.append("EXP-277 invalid run cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-277 invalid run cannot consume confirmatory data")
        if payload.get("challenge_materialized") is not False:
            errors.append("EXP-277 invalid run must fail before challenge materialization")
        if payload.get("decision_rule_executed") is not False:
            errors.append("EXP-277 invalid run cannot execute decision rule")
        if payload.get("per_replicate") != []:
            errors.append("EXP-277 invalid run cannot publish confirmatory rows")
        if not payload.get("integrity_errors"):
            errors.append("EXP-277 invalid run must record integrity errors")
        return errors

    if payload.get("evidence_level") != "EV-E2" or decision != "UNVERIFIED":
        errors.append("EXP-277 raw executor cannot promote scientific evidence")
    if payload.get("decision_rule_executed") is not False:
        errors.append("EXP-277 raw executor cannot execute frozen decision rule")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-277 valid raw executor must materialize challenge")
    if payload.get("challenge_stream") != CHALLENGE_STREAM:
        errors.append("EXP-277 raw challenge stream drift")
    if payload.get("integrity_errors") != []:
        errors.append("EXP-277 valid raw artifact cannot contain integrity errors")

    test_only = payload.get("test_only")
    scientific = payload.get("scientific_evidence_eligible")
    if test_only is True:
        if status != TEST_ONLY_STATUS:
            errors.append("EXP-277 TEST-ONLY raw status drift")
        if scientific is not False:
            errors.append("EXP-277 TEST-ONLY raw cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-277 TEST-ONLY raw cannot consume confirmatory data")
        if payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("EXP-277 TEST-ONLY raw synthetic-data flag missing")
        if payload.get("seed_materialization_status") != "TEST_ONLY_EXECUTED":
            errors.append("EXP-277 TEST-ONLY seed materialization status drift")
    elif test_only is False:
        if status != SCIENTIFIC_STATUS:
            errors.append("EXP-277 scientific raw status drift")
        if scientific is not True:
            errors.append("EXP-277 scientific raw must be evidence-eligible")
        if payload.get("confirmatory_data_consumed") is not True:
            errors.append("EXP-277 scientific raw must record confirmatory consumption")
        if payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("EXP-277 scientific raw cannot mark synthetic challenge consumption")
        if payload.get("seed_materialization_status") != "EXECUTED":
            errors.append("EXP-277 scientific seed materialization status drift")
    else:
        errors.append("EXP-277 raw test_only classification missing")

    reconstruction = payload.get("reconstruction_authorization")
    seal = payload.get("seal")
    beacon = payload.get("beacon_receipt")
    if not isinstance(reconstruction, dict) or not isinstance(seal, dict) or not isinstance(beacon, dict):
        errors.append("EXP-277 raw reconstruction/seal/beacon lineage missing")
        return errors
    reconstruction_errors = validate_exp277_reconstruction_authorization(reconstruction)
    if reconstruction_errors:
        errors.append("EXP-277 raw reconstruction authorization invalid: " + "; ".join(reconstruction_errors))
    seal_errors = validate_exp277_gate_a_seal(seal)
    if seal_errors:
        errors.append("EXP-277 raw Gate A seal invalid: " + "; ".join(seal_errors))

    reserved = payload.get("reserved_replicate_ids") or []
    rows = payload.get("per_replicate") or []
    if payload.get("confirmatory_n") != len(reserved) or len(rows) != len(reserved):
        errors.append("EXP-277 raw confirmatory row count mismatch")
    if [row.get("replicate") for row in rows if isinstance(row, dict)] != reserved:
        errors.append("EXP-277 raw replicate lineage must exactly match reserved IDs")

    expected_inputs = _arm_input_receipt()
    expected_flops = reconstruction.get("arm_accounted_flops_per_episode") or {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("EXP-277 raw row is not an object")
            continue
        if row.get("row_digest") != _row_digest(row):
            errors.append("EXP-277 raw row digest mismatch")
        if row.get("arm_input_receipt") != expected_inputs:
            errors.append("EXP-277 raw arm input receipt mismatch")
        for arm in ("arcs_branch", "oracle_cbrf"):
            arm_payload = row.get(arm) or {}
            expected_cost = _analytical_cost_receipt(int(expected_flops.get(arm, 0))) if expected_flops.get(arm) else None
            if arm_payload.get("analytical_cost_receipt") != expected_cost:
                errors.append(f"EXP-277 {arm} analytical cost receipt mismatch")

    if checkpoint_path is not None and checkpoint_receipt is not None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_errors = validate_exp277_raw_row_against_reconstruction(
                raw_row=row,
                reconstruction_authorization=reconstruction,
                seal=seal,
                beacon_receipt=beacon,
                checkpoint_path=checkpoint_path,
                checkpoint_receipt=checkpoint_receipt,
            )
            errors.extend(row_errors)
    return errors
