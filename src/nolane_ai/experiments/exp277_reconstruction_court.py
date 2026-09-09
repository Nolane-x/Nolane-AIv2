from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from .exp277_beacon import derive_exp277_challenge_seed, validate_exp277_beacon_receipt
from .exp277_challenge_worlds import build_exp277_challenge_batch
from .exp277_checkpoint import load_exp277_trained_checkpoint, validate_exp277_checkpoint_receipt
from .exp277_confirmatory_authorization import validate_exp277_gate_a_seal
from .exp277_paired_runner import _functional_state_digest


SCHEMA = "NLM-EXP-277-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
STATUS = "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
EXPERIMENT_ID = "EXP-277"
CHALLENGE_STREAM = "challenge"


def _authorization_digest(payload: dict[str, Any]) -> str:
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
            "checkpoint_seal_created_at_utc": payload.get("checkpoint_seal_created_at_utc"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "checkpoint_scientific_identity_digest": payload.get("checkpoint_scientific_identity_digest"),
            "checkpoint_receipt_digest": payload.get("checkpoint_receipt_digest"),
            "checkpoint_file_sha256": payload.get("checkpoint_file_sha256"),
            "arm_accounted_flops_per_episode": payload.get("arm_accounted_flops_per_episode"),
            "model_geometry": payload.get("model_geometry"),
            "oracle_information_separation": payload.get("oracle_information_separation"),
            "reconstruction_code_digest": payload.get("reconstruction_code_digest"),
        }
    )


def _row_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("row_digest", None)
    return canonical_sha256(clean)


def _valid_digest(value: Any, *, size: int = 64) -> bool:
    return isinstance(value, str) and len(value) == size


def _sealed_authorization(seal: dict[str, Any]) -> dict[str, Any]:
    errors = validate_exp277_gate_a_seal(seal)
    if errors:
        raise ValueError("invalid EXP-277 Gate A seal: " + "; ".join(errors))
    snapshot = seal.get("authorization_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("EXP-277 Gate A seal authorization snapshot missing")
    return snapshot


def build_exp277_reconstruction_authorization(
    *,
    seal: dict[str, Any],
    reconstruction_code_digest: str,
) -> dict[str, Any]:
    sealed = _sealed_authorization(seal)
    if not _valid_digest(reconstruction_code_digest):
        raise ValueError("EXP-277 reconstruction code digest must be a 64-character digest")
    machinery = sealed.get("machinery_digests") or {}
    if machinery.get("reconstruction_code_digest") != reconstruction_code_digest:
        raise ValueError("EXP-277 reconstruction code digest does not match Gate A machinery freeze")

    checkpoint = sealed.get("checkpoint") or {}
    resource = sealed.get("resource_court") or {}
    arm_flops = deepcopy(resource.get("arm_accounted_flops_per_episode") or {})
    if set(arm_flops) != {"arcs_branch", "oracle_cbrf"}:
        raise ValueError("EXP-277 reconstruction requires exact per-arm FLOP bindings")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in arm_flops.values()):
        raise ValueError("EXP-277 reconstruction per-arm FLOP binding invalid")

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
        "checkpoint_seal_created_at_utc": sealed.get("checkpoint_seal_created_at_utc"),
        "confirmatory_n": sealed.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(sealed.get("reserved_replicate_ids") or []),
        "checkpoint_scientific_identity_digest": checkpoint.get("scientific_identity_digest"),
        "checkpoint_receipt_digest": checkpoint.get("receipt_digest"),
        "checkpoint_file_sha256": checkpoint.get("checkpoint_file_sha256"),
        "arm_accounted_flops_per_episode": arm_flops,
        "model_geometry": deepcopy(sealed.get("model_geometry") or {}),
        "oracle_information_separation": deepcopy(sealed.get("oracle_information_separation") or {}),
        "reconstruction_code_digest": reconstruction_code_digest,
        "binding_digest": "",
        "reconstruction_digest": "",
    }
    payload["binding_digest"] = _binding_digest(payload)
    payload["reconstruction_digest"] = _authorization_digest(payload)
    errors = validate_exp277_reconstruction_authorization(payload)
    if errors:
        raise RuntimeError("invalid EXP-277 reconstruction authorization: " + "; ".join(errors))
    return payload


def validate_exp277_reconstruction_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-277 reconstruction authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-277 reconstruction authorization cannot promote evidence")
    if payload.get("status") != STATUS:
        errors.append("EXP-277 reconstruction authorization status drift")
    for flag in ("confirmatory_ready", "confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(flag) is not False:
            errors.append(f"EXP-277 reconstruction authorization forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-277 reconstruction authorization cannot materialize seeds")

    seal = payload.get("seal_snapshot")
    if not isinstance(seal, dict):
        errors.append("EXP-277 reconstruction seal snapshot missing")
        return errors
    seal_errors = validate_exp277_gate_a_seal(seal)
    if seal_errors:
        errors.append("invalid EXP-277 reconstruction seal snapshot: " + "; ".join(seal_errors))
        return errors
    if payload.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-277 reconstruction seal digest mismatch")
    if payload.get("seal_snapshot_digest") != canonical_sha256(seal):
        errors.append("EXP-277 reconstruction seal snapshot digest mismatch")

    sealed = seal.get("authorization_snapshot") or {}
    checkpoint = sealed.get("checkpoint") or {}
    resource = sealed.get("resource_court") or {}
    expected = {
        "protocol_digest": sealed.get("protocol_digest"),
        "source_tree_digest": sealed.get("source_tree_digest"),
        "freeze_commit_sha": sealed.get("freeze_commit_sha"),
        "freeze_commit_timestamp_utc": sealed.get("freeze_commit_timestamp_utc"),
        "checkpoint_seal_created_at_utc": sealed.get("checkpoint_seal_created_at_utc"),
        "confirmatory_n": sealed.get("confirmatory_n"),
        "reserved_replicate_ids": sealed.get("reserved_replicate_ids"),
        "checkpoint_scientific_identity_digest": checkpoint.get("scientific_identity_digest"),
        "checkpoint_receipt_digest": checkpoint.get("receipt_digest"),
        "checkpoint_file_sha256": checkpoint.get("checkpoint_file_sha256"),
        "arm_accounted_flops_per_episode": resource.get("arm_accounted_flops_per_episode"),
        "model_geometry": sealed.get("model_geometry"),
        "oracle_information_separation": sealed.get("oracle_information_separation"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            errors.append(f"EXP-277 reconstruction Gate A lineage mismatch: {field}")

    machinery = sealed.get("machinery_digests") or {}
    if payload.get("reconstruction_code_digest") != machinery.get("reconstruction_code_digest"):
        errors.append("EXP-277 reconstruction machinery digest mismatch")
    arm_flops = payload.get("arm_accounted_flops_per_episode")
    if not isinstance(arm_flops, dict) or set(arm_flops) != {"arcs_branch", "oracle_cbrf"}:
        errors.append("EXP-277 reconstruction per-arm FLOP binding missing")
    elif any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in arm_flops.values()):
        errors.append("EXP-277 reconstruction per-arm FLOP binding invalid")

    if payload.get("binding_digest") != _binding_digest(payload):
        errors.append("EXP-277 reconstruction semantic binding digest mismatch")
    if payload.get("reconstruction_digest") != _authorization_digest(payload):
        errors.append("EXP-277 reconstruction authorization digest mismatch")
    return errors


def _validate_checkpoint_against_seal(
    *,
    seal: dict[str, Any],
    checkpoint_receipt: dict[str, Any],
) -> None:
    receipt_errors = validate_exp277_checkpoint_receipt(checkpoint_receipt)
    if receipt_errors:
        raise ValueError("invalid EXP-277 checkpoint receipt: " + "; ".join(receipt_errors))
    sealed = (seal.get("authorization_snapshot") or {}).get("checkpoint") or {}
    for field in (
        "receipt_digest",
        "scientific_identity_digest",
        "execution_contract_digest",
        "arcs_branch_final_digest",
        "oracle_cbrf_final_digest",
        "checkpoint_file_sha256",
        "state_policy",
    ):
        if checkpoint_receipt.get(field) != sealed.get(field):
            raise ValueError(f"EXP-277 checkpoint Gate A binding mismatch: {field}")


def _arm_metrics(output: Any, targets: torch.Tensor, *, accounted_flops: int) -> dict[str, Any]:
    predictions = output.decision_logits.argmax(dim=-1)
    exact_per_episode = (predictions == targets).all(dim=-1)
    solution_rate = float(exact_per_episode.to(torch.float32).mean().item())
    decision_accuracy = float((predictions == targets).to(torch.float32).mean().item())
    return {
        "predictions": predictions.detach().cpu().tolist(),
        "verified_solution_rate": solution_rate,
        "verified_decision_accuracy": decision_accuracy,
        "mean_verifier_confidence": float(output.verifier_confidence.mean().item()),
        "accounted_flops_per_episode": int(accounted_flops),
        "verified_utility_per_accounted_flop": solution_rate / int(accounted_flops),
    }


def reconstruct_exp277_expected_row(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    replicate: int,
) -> dict[str, Any]:
    auth_errors = validate_exp277_reconstruction_authorization(reconstruction_authorization)
    if auth_errors:
        raise ValueError("invalid EXP-277 reconstruction authorization: " + "; ".join(auth_errors))
    seal_errors = validate_exp277_gate_a_seal(seal)
    if seal_errors:
        raise ValueError("invalid EXP-277 Gate A seal: " + "; ".join(seal_errors))
    if reconstruction_authorization.get("seal_digest") != seal.get("seal_digest"):
        raise ValueError("EXP-277 reconstruction/seal binding mismatch")
    if reconstruction_authorization.get("seal_snapshot_digest") != canonical_sha256(seal):
        raise ValueError("EXP-277 reconstruction seal snapshot mismatch")

    reserved = list(reconstruction_authorization.get("reserved_replicate_ids") or [])
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate not in reserved:
        raise ValueError("EXP-277 reconstruction replicate is outside frozen reserved lineage")

    beacon_errors = validate_exp277_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=reconstruction_authorization.get("freeze_commit_timestamp_utc"),
        checkpoint_seal_created_at_utc=reconstruction_authorization.get("checkpoint_seal_created_at_utc"),
    )
    if beacon_errors:
        raise ValueError("invalid EXP-277 beacon receipt: " + "; ".join(beacon_errors))

    _validate_checkpoint_against_seal(seal=seal, checkpoint_receipt=checkpoint_receipt)
    arcs, oracle = load_exp277_trained_checkpoint(
        checkpoint_path=checkpoint_path,
        receipt=checkpoint_receipt,
    )
    arcs.eval()
    oracle.eval()
    before = {
        "arcs_branch": _functional_state_digest(arcs),
        "oracle_cbrf": _functional_state_digest(oracle),
    }

    challenge_seed = derive_exp277_challenge_seed(
        protocol_digest=str(reconstruction_authorization.get("protocol_digest") or ""),
        beacon_receipt=beacon_receipt,
        stream=CHALLENGE_STREAM,
        replicate=replicate,
        freeze_commit_timestamp_utc=reconstruction_authorization.get("freeze_commit_timestamp_utc"),
        checkpoint_seal_created_at_utc=reconstruction_authorization.get("checkpoint_seal_created_at_utc"),
    )
    geometry = reconstruction_authorization.get("model_geometry") or {}
    world = geometry.get("world") or {}
    batch = build_exp277_challenge_batch(
        challenge_seed=challenge_seed,
        replicate=replicate,
        batch_size=int(world.get("batch_size", 0)),
        timesteps=int(world.get("timesteps", 0)),
        variables=int(world.get("variables", 0)),
        constraints=int(world.get("constraints", 0)),
        d_model=int(world.get("d_model", 0)),
        noise_std=float(world.get("noise_std", -1.0)),
        device="cpu",
    )
    arm_flops = reconstruction_authorization["arm_accounted_flops_per_episode"]
    with torch.no_grad():
        arcs_output = arcs(batch.surface_events, batch.variable_states)
        oracle_output = oracle(batch.surface_events, batch.variable_states, batch.oracle_incidence)
    after = {
        "arcs_branch": _functional_state_digest(arcs),
        "oracle_cbrf": _functional_state_digest(oracle),
    }
    if before != after:
        raise RuntimeError("EXP-277 reconstruction detected a forbidden parameter write")

    arcs_metrics = _arm_metrics(arcs_output, batch.targets, accounted_flops=int(arm_flops["arcs_branch"]))
    oracle_metrics = _arm_metrics(oracle_output, batch.targets, accounted_flops=int(arm_flops["oracle_cbrf"]))
    arcs_utility = float(arcs_metrics["verified_utility_per_accounted_flop"])
    oracle_utility = float(oracle_metrics["verified_utility_per_accounted_flop"])
    relative_gain = None if arcs_utility <= 0.0 else (oracle_utility - arcs_utility) / arcs_utility

    row: dict[str, Any] = {
        "schema": "NLM-EXP-277-CONFIRMATORY-RAW-ROW-V1",
        "experiment_id": EXPERIMENT_ID,
        "replicate": replicate,
        "replicate_ordinal": reserved.index(replicate),
        "challenge_seed": challenge_seed,
        "challenge_digest": batch.digest,
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "checkpoint_functional_state_before": before,
        "checkpoint_functional_state_after": after,
        "oracle_information_receipt": deepcopy(reconstruction_authorization.get("oracle_information_separation") or {}),
        "arcs_branch": arcs_metrics,
        "oracle_cbrf": oracle_metrics,
        "paired": {
            "world_pairing_closed": True,
            "solution_rate_difference": float(oracle_metrics["verified_solution_rate"]) - float(arcs_metrics["verified_solution_rate"]),
            "utility_difference": oracle_utility - arcs_utility,
            "relative_utility_gain": relative_gain,
            "baseline_denominator_positive": arcs_utility > 0.0,
        },
        "row_digest": "",
    }
    row["row_digest"] = _row_digest(row)
    return row


def _float_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return False
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return False
    left_f = float(left)
    right_f = float(right)
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return left_f == right_f
    return math.isclose(left_f, right_f, rel_tol=0.0, abs_tol=1e-15)


def validate_exp277_raw_row_against_reconstruction(
    *,
    raw_row: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw_row, dict):
        return ["EXP-277 raw row must be an object"]
    if raw_row.get("row_digest") != _row_digest(raw_row):
        errors.append("EXP-277 raw row digest mismatch")

    replicate = raw_row.get("replicate")
    reserved = list(reconstruction_authorization.get("reserved_replicate_ids") or [])
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate not in reserved:
        errors.append("EXP-277 raw row replicate is outside reserved lineage")
        return errors
    expected_ordinal = reserved.index(replicate)
    if raw_row.get("replicate_ordinal") != expected_ordinal:
        errors.append("EXP-277 raw row replicate lineage ordinal mismatch")

    try:
        expected = reconstruct_exp277_expected_row(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            checkpoint_path=checkpoint_path,
            checkpoint_receipt=checkpoint_receipt,
            replicate=replicate,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    if raw_row.get("challenge_seed") != expected.get("challenge_seed"):
        errors.append("EXP-277 raw row challenge seed mismatch")
    if raw_row.get("challenge_digest") != expected.get("challenge_digest"):
        errors.append("EXP-277 raw row challenge digest mismatch")
    if raw_row.get("checkpoint_scientific_identity_digest") != expected.get("checkpoint_scientific_identity_digest"):
        errors.append("EXP-277 raw row checkpoint identity mismatch")
    if raw_row.get("checkpoint_functional_state_before") != expected.get("checkpoint_functional_state_before") or raw_row.get("checkpoint_functional_state_after") != expected.get("checkpoint_functional_state_after"):
        errors.append("EXP-277 raw row checkpoint functional-state mismatch")
    if raw_row.get("oracle_information_receipt") != expected.get("oracle_information_receipt"):
        errors.append("EXP-277 raw row oracle-information separation mismatch")

    for arm in ("arcs_branch", "oracle_cbrf"):
        actual_arm = raw_row.get(arm) or {}
        expected_arm = expected.get(arm) or {}
        if actual_arm.get("predictions") != expected_arm.get("predictions"):
            errors.append(f"EXP-277 {arm} prediction mismatch")
        for field in ("verified_solution_rate", "verified_decision_accuracy", "mean_verifier_confidence"):
            if not _float_equal(actual_arm.get(field), expected_arm.get(field)):
                label = "solution" if field == "verified_solution_rate" else "prediction metric"
                errors.append(f"EXP-277 {arm} {label} mismatch: {field}")
        if actual_arm.get("accounted_flops_per_episode") != expected_arm.get("accounted_flops_per_episode"):
            errors.append(f"EXP-277 {arm} accounted FLOP mismatch")
        if not _float_equal(
            actual_arm.get("verified_utility_per_accounted_flop"),
            expected_arm.get("verified_utility_per_accounted_flop"),
        ):
            errors.append(f"EXP-277 {arm} utility mismatch")

    actual_paired = raw_row.get("paired") or {}
    expected_paired = expected.get("paired") or {}
    if actual_paired.get("world_pairing_closed") is not True:
        errors.append("EXP-277 paired world lineage is open")
    for field in ("solution_rate_difference", "utility_difference"):
        if not _float_equal(actual_paired.get(field), expected_paired.get(field)):
            errors.append(f"EXP-277 paired metric mismatch: {field}")
    if actual_paired.get("relative_utility_gain") is None or expected_paired.get("relative_utility_gain") is None:
        if actual_paired.get("relative_utility_gain") != expected_paired.get("relative_utility_gain"):
            errors.append("EXP-277 paired relative utility mismatch")
    elif not _float_equal(actual_paired.get("relative_utility_gain"), expected_paired.get("relative_utility_gain")):
        errors.append("EXP-277 paired relative utility mismatch")
    if actual_paired.get("baseline_denominator_positive") != expected_paired.get("baseline_denominator_positive"):
        errors.append("EXP-277 paired denominator status mismatch")
    return errors
