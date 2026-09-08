from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch

from nolane_ai.experiments.exp297_beacon import (
    derive_exp297_challenge_seed,
    validate_exp297_beacon_receipt,
)
from nolane_ai.experiments.exp297_challenge_worlds import (
    challenge_contract_digest,
    generate_exp297_challenge_world,
)
from nolane_ai.experiments.exp297_reconstruction_court import (
    validate_exp297_confirmatory_reconstruction,
)
from nolane_ai.experiments.matched_fidelity_arms import (
    audit_matched_fidelity_arms,
    build_matched_fidelity_arms,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.reasoning.fidelity import FidelityCourt, compile_valid

SCHEMA = "NLM-EXP-297-CONFIRMATORY-CHALLENGE-RAW-V1"
EXPERIMENT_ID = "EXP-297"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
STREAM = "challenge"
CANDIDATE_COUNT = 16

_EXPECTATION_CACHE: dict[str, dict[str, Any]] = {}


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _valid_freeze_sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_unit(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _decision_payload(decision: Any, *, semantic_ops: int) -> dict[str, Any]:
    return {
        "authority_granted": bool(decision.authority_granted),
        "fidelity_score": float(decision.fidelity_score),
        "authority_score": float(decision.authority_score),
        "verifier_score": float(decision.verifier_score),
        "receipt_semantics": decision.receipt_semantics,
        "arm_observable_digest": decision.arm_observable_digest,
        "neural_accounted_flops": int(decision.neural_accounted_flops),
        "compile_validation_operations": 1,
        "semantic_verification_operations": int(semantic_ops),
        "total_accounted_cost_proxy": int(
            decision.neural_accounted_flops + 1 + semantic_ops
        ),
        "hardware_profiler_flops_claimed": False,
    }


def _execution_identity(
    *,
    reconstruction: dict[str, Any],
    beacon_receipt: dict[str, Any],
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
    executor_code_digest: str,
) -> str:
    return canonical_sha256(
        {
            "reconstruction_digest": reconstruction.get("reconstruction_digest"),
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "freeze_commit_sha": freeze_commit_sha,
            "freeze_commit_timestamp_utc": freeze_commit_timestamp_utc,
            "executor_code_digest": executor_code_digest,
        }
    )


def _validate_inputs(
    *,
    reconstruction: dict[str, Any],
    beacon_receipt: dict[str, Any],
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
    executor_code_digest: str,
) -> None:
    reconstruction_errors = validate_exp297_confirmatory_reconstruction(reconstruction)
    if reconstruction_errors:
        raise ValueError(
            "invalid EXP-297 reconstruction authorization: "
            + "; ".join(reconstruction_errors)
        )
    if not _valid_freeze_sha(freeze_commit_sha):
        raise ValueError("EXP-297 freeze commit SHA must be a 40- or 64-hex digest")
    if not isinstance(executor_code_digest, str) or not executor_code_digest:
        raise ValueError("executor_code_digest is required")
    beacon_errors = validate_exp297_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
    )
    if beacon_errors:
        raise ValueError("invalid EXP-297 beacon receipt: " + "; ".join(beacon_errors))
    if reconstruction.get("challenge_contract_digest") != challenge_contract_digest():
        raise ValueError("EXP-297 reconstruction challenge contract drift")


def _build_expected(
    *,
    reconstruction: dict[str, Any],
    beacon_receipt: dict[str, Any],
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
    executor_code_digest: str,
) -> dict[str, Any]:
    cache_key = _execution_identity(
        reconstruction=reconstruction,
        beacon_receipt=beacon_receipt,
        freeze_commit_sha=freeze_commit_sha,
        freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
        executor_code_digest=executor_code_digest,
    )
    cached = _EXPECTATION_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    geometry = reconstruction.get("model_geometry") or {}
    try:
        d_model = int(geometry["d_model"])
        hidden_size = int(geometry["hidden_size"])
        target_parameters = int(geometry["target_parameters"])
        max_exact_assignments = int(geometry["max_exact_assignments"])
        model_init_seed = int((reconstruction.get("lineage") or {})["model_init_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EXP-297 frozen model reconstruction geometry invalid") from exc
    if min(d_model, hidden_size, target_parameters, max_exact_assignments) <= 0:
        raise ValueError("EXP-297 frozen model reconstruction geometry invalid")

    reserved = reconstruction.get("reserved_replicate_ids") or []
    confirmatory_n = reconstruction.get("confirmatory_n")
    if (
        not isinstance(confirmatory_n, int)
        or isinstance(confirmatory_n, bool)
        or not 32 <= confirmatory_n <= 128
        or not isinstance(reserved, list)
        or len(reserved) != confirmatory_n
        or len(set(reserved)) != len(reserved)
        or (reserved and reserved != list(range(reserved[0], reserved[0] + len(reserved))))
    ):
        raise ValueError("EXP-297 frozen confirmatory replicate lineage invalid")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_init_seed)
        compile_arm, fidelity_arm = build_matched_fidelity_arms(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
        )
    compile_arm.eval()
    fidelity_arm.eval()
    pair_audit = audit_matched_fidelity_arms(compile_arm, fidelity_arm)
    expected_pair_digest = (reconstruction.get("lineage") or {}).get("pair_audit_digest")
    if canonical_sha256(pair_audit) != expected_pair_digest:
        raise ValueError("EXP-297 frozen matched-pair audit reconstruction mismatch")

    protocol_digest = (reconstruction.get("lineage") or {}).get("protocol_digest")
    if not isinstance(protocol_digest, str) or not protocol_digest:
        raise ValueError("EXP-297 reconstruction protocol lineage missing")

    court = FidelityCourt(max_exact_assignments=max_exact_assignments)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for replicate in reserved:
            challenge_seed = derive_exp297_challenge_seed(
                protocol_digest=protocol_digest,
                freeze_commit_sha=freeze_commit_sha,
                freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
                beacon_receipt=beacon_receipt,
                stream=STREAM,
                replicate=replicate,
            )
            batch = generate_exp297_challenge_world(challenge_seed)
            if len(batch.candidates) != CANDIDATE_COUNT:
                raise ValueError("EXP-297 hidden challenge candidate count drift")
            for candidate_index, case in enumerate(batch.candidates):
                compiled = compile_valid(case.candidate)
                receipt = court.adjudicate(batch.source, case.candidate)
                compile_decision = compile_arm.decide(
                    batch.source,
                    case.candidate,
                    compile_valid=compiled,
                )
                fidelity_decision = fidelity_arm.decide(
                    batch.source,
                    case.candidate,
                    compile_valid=compiled,
                    court_receipt=receipt,
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "challenge_seed": challenge_seed,
                        "candidate_index": candidate_index,
                        "candidate_id": case.candidate_id,
                        "stratum": case.stratum,
                        "is_faithful": case.is_faithful,
                        "source_digest": batch.source_digest,
                        "candidate_digest": case.candidate_digest,
                        "compile_valid": compiled,
                        "court_receipt": receipt.as_dict(),
                        "arm_input_receipt": {
                            "candidate_set_frozen_before_arms": True,
                            "byte_identical_candidate_order": True,
                            "compile_only_candidate_digest": case.candidate_digest,
                            "fidelity_court_candidate_digest": case.candidate_digest,
                            "evaluator_truth_in_causal_path": False,
                            "trap_family_in_causal_path": False,
                        },
                        "arms": {
                            "compile_only": _decision_payload(
                                compile_decision,
                                semantic_ops=0,
                            ),
                            "fidelity_court": _decision_payload(
                                fidelity_decision,
                                semantic_ops=receipt.semantic_verification_operations,
                            ),
                        },
                    }
                )

    result = {
        "rows": rows,
        "pair_audit": pair_audit,
        "pair_audit_digest": canonical_sha256(pair_audit),
        "model_init_seed": model_init_seed,
        "model_geometry": {
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
            "max_exact_assignments": max_exact_assignments,
        },
    }
    _EXPECTATION_CACHE[cache_key] = deepcopy(result)
    return result


def _validate_arm(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    arm_id: str,
    errors: list[str],
) -> None:
    exact_fields = (
        "authority_granted",
        "receipt_semantics",
        "arm_observable_digest",
        "neural_accounted_flops",
        "compile_validation_operations",
        "semantic_verification_operations",
        "total_accounted_cost_proxy",
        "hardware_profiler_flops_claimed",
    )
    for key in exact_fields:
        if observed.get(key) != expected.get(key):
            errors.append(f"EXP-297 {arm_id} {key} reconstruction mismatch")
    for key in ("fidelity_score", "authority_score", "verifier_score"):
        value = observed.get(key)
        expected_value = expected.get(key)
        if not _finite_unit(value):
            errors.append(f"EXP-297 {arm_id} {key} invalid")
            continue
        if not math.isclose(
            float(value),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            errors.append(f"EXP-297 {arm_id} {key} reconstruction mismatch")


def execute_exp297_confirmatory_challenge(
    *,
    reconstruction_authorization: dict[str, Any],
    beacon_receipt: dict[str, Any],
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
    executor_code_digest: str,
) -> dict[str, Any]:
    _validate_inputs(
        reconstruction=reconstruction_authorization,
        beacon_receipt=beacon_receipt,
        freeze_commit_sha=freeze_commit_sha,
        freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
        executor_code_digest=executor_code_digest,
    )
    expected = _build_expected(
        reconstruction=reconstruction_authorization,
        beacon_receipt=beacon_receipt,
        freeze_commit_sha=freeze_commit_sha,
        freeze_commit_timestamp_utc=freeze_commit_timestamp_utc,
        executor_code_digest=executor_code_digest,
    )

    test_only = beacon_receipt.get("test_only") is True
    scientific_eligible = bool(
        beacon_receipt.get("scientific_evidence_eligible") is True and not test_only
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": scientific_eligible,
        "confirmatory_data_consumed": bool(not test_only),
        "synthetic_challenge_data_consumed": bool(test_only),
        "seed_materialization_status": (
            "TEST_ONLY_EXECUTED" if test_only else "EXECUTED"
        ),
        "challenge_materialized": True,
        "decision_rule_executed": False,
        "semantic_authority_promoted": False,
        "confirmatory_n": reconstruction_authorization.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(
            reconstruction_authorization.get("reserved_replicate_ids") or []
        ),
        "candidate_count_per_replicate": CANDIDATE_COUNT,
        "challenge_stream": STREAM,
        "freeze_commit_sha": freeze_commit_sha,
        "freeze_commit_timestamp_utc": freeze_commit_timestamp_utc,
        "beacon_receipt": deepcopy(beacon_receipt),
        "model_geometry": deepcopy(expected["model_geometry"]),
        "pair_audit": deepcopy(expected["pair_audit"]),
        "raw_candidates": deepcopy(expected["rows"]),
        "reconstruction_authorization": deepcopy(reconstruction_authorization),
        "lineage": {
            "protocol_digest": (reconstruction_authorization.get("lineage") or {}).get(
                "protocol_digest"
            ),
            "reconstruction_digest": reconstruction_authorization.get(
                "reconstruction_digest"
            ),
            "execution_authorization_digest": (
                reconstruction_authorization.get("lineage") or {}
            ).get("execution_authorization_digest"),
            "challenge_contract_digest": reconstruction_authorization.get(
                "challenge_contract_digest"
            ),
            "pair_audit_digest": expected["pair_audit_digest"],
            "model_init_seed": expected["model_init_seed"],
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "executor_code_digest": executor_code_digest,
        },
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp297_confirmatory_raw(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 confirmatory raw artifact: " + "; ".join(errors))
    return payload


def validate_exp297_confirmatory_raw(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-297 confirmatory raw identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 confirmatory raw cannot promote evidence")
    if payload.get("decision_rule_executed") is not False:
        errors.append("EXP-297 raw executor cannot execute the frozen decision rule")
    if payload.get("semantic_authority_promoted") is not False:
        errors.append("EXP-297 raw executor cannot promote semantic authority")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-297 raw executor must record challenge materialization")
    if payload.get("candidate_count_per_replicate") != CANDIDATE_COUNT:
        errors.append("EXP-297 raw candidate count contract drift")
    if payload.get("challenge_stream") != STREAM:
        errors.append("EXP-297 raw challenge stream drift")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-297 confirmatory raw artifact digest mismatch")

    test_only = payload.get("test_only")
    scientific_eligible = payload.get("scientific_evidence_eligible")
    if test_only is True:
        if payload.get("status") != TEST_ONLY_STATUS:
            errors.append("EXP-297 TEST-ONLY raw status drift")
        if scientific_eligible is not False:
            errors.append("EXP-297 TEST-ONLY raw cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-297 TEST-ONLY raw cannot consume confirmatory data")
        if payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("EXP-297 TEST-ONLY synthetic challenge consumption missing")
        if payload.get("seed_materialization_status") != "TEST_ONLY_EXECUTED":
            errors.append("EXP-297 TEST-ONLY seed materialization status drift")
    elif test_only is False:
        if payload.get("status") != SCIENTIFIC_STATUS:
            errors.append("EXP-297 confirmatory raw status drift")
        if scientific_eligible is not True:
            errors.append("EXP-297 external beacon raw must be scientific-evidence eligible")
        if payload.get("confirmatory_data_consumed") is not True:
            errors.append("EXP-297 confirmatory raw must record consumed data")
        if payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("EXP-297 scientific raw cannot claim synthetic challenge consumption")
        if payload.get("seed_materialization_status") != "EXECUTED":
            errors.append("EXP-297 confirmatory seed materialization status drift")
    else:
        errors.append("EXP-297 raw test_only classification missing")

    freeze_sha = payload.get("freeze_commit_sha")
    freeze_time = payload.get("freeze_commit_timestamp_utc")
    executor_code_digest = (payload.get("lineage") or {}).get("executor_code_digest")
    if not _valid_freeze_sha(freeze_sha):
        errors.append("EXP-297 raw freeze commit SHA invalid")
    if not isinstance(executor_code_digest, str) or not executor_code_digest:
        errors.append("EXP-297 raw executor code digest missing")

    beacon = payload.get("beacon_receipt")
    if not isinstance(beacon, dict):
        errors.append("EXP-297 raw beacon receipt missing")
        return errors
    beacon_errors = validate_exp297_beacon_receipt(
        beacon,
        freeze_commit_timestamp_utc=freeze_time,
    )
    if beacon_errors:
        errors.extend(beacon_errors)
        return errors
    if beacon.get("test_only") is not test_only:
        errors.append("EXP-297 raw/beacon TEST-ONLY classification mismatch")

    reconstruction = payload.get("reconstruction_authorization")
    if not isinstance(reconstruction, dict):
        errors.append("EXP-297 raw reconstruction authorization missing")
        return errors
    reconstruction_errors = validate_exp297_confirmatory_reconstruction(reconstruction)
    if reconstruction_errors:
        errors.extend(
            "EXP-297 raw reconstruction invalid: " + error
            for error in reconstruction_errors
        )
        return errors

    if payload.get("confirmatory_n") != reconstruction.get("confirmatory_n"):
        errors.append("EXP-297 raw confirmatory n mismatch")
    if payload.get("reserved_replicate_ids") != reconstruction.get(
        "reserved_replicate_ids"
    ):
        errors.append("EXP-297 raw reserved replicate lineage mismatch")
    if payload.get("model_geometry") != reconstruction.get("model_geometry"):
        errors.append("EXP-297 raw model geometry mismatch")

    lineage = payload.get("lineage") or {}
    expected_lineage_static = {
        "protocol_digest": (reconstruction.get("lineage") or {}).get("protocol_digest"),
        "reconstruction_digest": reconstruction.get("reconstruction_digest"),
        "execution_authorization_digest": (reconstruction.get("lineage") or {}).get(
            "execution_authorization_digest"
        ),
        "challenge_contract_digest": reconstruction.get("challenge_contract_digest"),
        "beacon_receipt_digest": beacon.get("receipt_digest"),
        "executor_code_digest": executor_code_digest,
    }
    for key, expected_value in expected_lineage_static.items():
        if lineage.get(key) != expected_value:
            errors.append(f"EXP-297 raw lineage {key} mismatch")
    if errors:
        return errors

    try:
        expected = _build_expected(
            reconstruction=reconstruction,
            beacon_receipt=beacon,
            freeze_commit_sha=freeze_sha,
            freeze_commit_timestamp_utc=freeze_time,
            executor_code_digest=executor_code_digest,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    if lineage.get("pair_audit_digest") != expected["pair_audit_digest"]:
        errors.append("EXP-297 raw pair-audit lineage mismatch")
    if lineage.get("model_init_seed") != expected["model_init_seed"]:
        errors.append("EXP-297 raw model-init lineage mismatch")
    if payload.get("pair_audit") != expected["pair_audit"]:
        errors.append("EXP-297 raw matched-pair audit mismatch")

    rows = payload.get("raw_candidates")
    expected_rows = expected["rows"]
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        errors.append("EXP-297 raw candidate cardinality mismatch")
        return errors

    structural_fields = (
        "replicate",
        "challenge_seed",
        "candidate_index",
        "candidate_id",
        "stratum",
        "is_faithful",
        "source_digest",
        "candidate_digest",
        "compile_valid",
        "court_receipt",
        "arm_input_receipt",
    )
    for row, expected_row in zip(rows, expected_rows, strict=True):
        for key in structural_fields:
            if row.get(key) != expected_row.get(key):
                errors.append(f"EXP-297 raw candidate {key} reconstruction mismatch")
        observed_arms = row.get("arms") or {}
        expected_arms = expected_row["arms"]
        for arm_id in ("compile_only", "fidelity_court"):
            _validate_arm(
                observed_arms.get(arm_id) or {},
                expected_arms[arm_id],
                arm_id=arm_id,
                errors=errors,
            )
        if errors:
            break
    return errors
