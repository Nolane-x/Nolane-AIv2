from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256, require_frozen_stage_a_v1_sha256
from .exp286_beacon import derive_exp286_challenge_seed, validate_exp286_beacon_receipt
from .exp286_challenge_worlds import challenge_contract_digest
from .exp286_checkpoint import load_exp286_trained_checkpoint, validate_exp286_checkpoint_receipt
from .exp286_confirmatory_ceremony import validate_exp286_confirmatory_gate_a_seal
from .exp286_conflict_worlds import Exp286ConflictGenerator
from .exp286_paired_runner import _search_batch
from .matched_conflict_arms import audit_matched_exp286_arm_pair

SCHEMA = "NLM-EXP-286-CONFIRMATORY-CHALLENGE-RAW-V1"
EXPERIMENT_ID = "EXP-286"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
TEST_ONLY_SCOPE = "synthetic-exp286-post-freeze-challenge-test-only"
SCIENTIFIC_SCOPE = "exp286-post-freeze-confirmatory-challenge"
STREAM = "challenge"


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _row_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("row_digest", None)
    return canonical_sha256(clean)


def _is_hex(value: Any, *, lengths: set[int] = {64}) -> bool:
    if not isinstance(value, str) or len(value) not in lengths:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _mode(beacon_receipt: dict[str, Any]) -> bool:
    test_only = beacon_receipt.get("test_only")
    scientific = beacon_receipt.get("scientific_evidence_eligible")
    if test_only is True and scientific is False:
        return True
    if test_only is False and scientific is True:
        return False
    raise ValueError("EXP-286 beacon evidence mode is invalid")


def _validate_inputs(
    *,
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    seal_errors = validate_exp286_confirmatory_gate_a_seal(seal)
    if seal_errors:
        raise ValueError("invalid EXP-286 Gate-A seal: " + "; ".join(seal_errors))
    if not _is_hex(current_source_tree_digest) or current_source_tree_digest != seal.get("code_tree_digest"):
        raise ValueError("EXP-286 challenge source-tree digest does not match sealed code tree")
    lineage = seal.get("lineage") or {}
    if not _is_hex(executor_code_digest) or executor_code_digest != lineage.get("execution_code_digest"):
        raise ValueError("EXP-286 challenge executor code digest does not match execution authorization")
    if seal.get("challenge_contract_digest") != challenge_contract_digest():
        raise ValueError("EXP-286 challenge contract drift after Gate-A freeze")

    beacon_errors = validate_exp286_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=str(seal.get("freeze_commit_timestamp_utc") or ""),
    )
    if beacon_errors:
        raise ValueError("invalid EXP-286 beacon receipt: " + "; ".join(beacon_errors))
    _mode(beacon_receipt)

    checkpoint_errors = validate_exp286_checkpoint_receipt(checkpoint_receipt)
    if checkpoint_errors:
        raise ValueError("invalid EXP-286 checkpoint receipt: " + "; ".join(checkpoint_errors))
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"EXP-286 checkpoint file is missing: {path}")
    if file_sha256(path) != checkpoint_receipt.get("checkpoint_file_sha256"):
        raise ValueError("EXP-286 checkpoint file SHA256 mismatch before challenge execution")

    protocol_digest = lineage.get("protocol_digest")
    require_frozen_stage_a_v1_sha256(protocol_digest)
    contract = checkpoint_receipt.get("execution_contract") or {}
    if contract.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-286 checkpoint protocol lineage does not match Gate-A seal")

    chronological, oracle = load_exp286_trained_checkpoint(
        checkpoint_path=path,
        receipt=checkpoint_receipt,
    )
    config = contract.get("execution_config") or {}
    resource = contract.get("resource_binding") or {}
    pair_audit = audit_matched_exp286_arm_pair(
        chronological,
        oracle,
        timesteps=int(config["timesteps"]),
        variables=int(config["variables"]),
        max_search_steps=int(config["max_search_steps"]),
        max_accounted_flops_per_episode=int(resource["declared_max_accounted_flops_per_episode"]),
    )
    if canonical_sha256(pair_audit) != resource.get("pair_audit_digest"):
        raise ValueError("EXP-286 challenge matched-pair audit reconstruction mismatch")
    if pair_audit.get("compute_budget_closed") is not True or pair_audit.get("oracle_information_separation") is not True:
        raise ValueError("EXP-286 challenge resource/oracle-information court is not closed")
    return config, pair_audit, chronological, oracle


def execute_exp286_confirmatory_challenge(
    *,
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> dict[str, Any]:
    config, pair_audit, chronological, oracle = _validate_inputs(
        seal=seal,
        beacon_receipt=beacon_receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source_tree_digest,
        executor_code_digest=executor_code_digest,
    )
    test_only = _mode(beacon_receipt)
    confirmatory_n = seal.get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not 32 <= confirmatory_n <= 128:
        raise ValueError("EXP-286 confirmatory_n is outside frozen bounds")

    resource = (checkpoint_receipt.get("execution_contract") or {}).get("resource_binding") or {}
    ceiling = int(resource["declared_max_accounted_flops_per_episode"])
    compute = pair_audit.get("compute_ledger") or {}
    chronological_flops = int(compute["chronological_failure"]["accounted_neural_flops_per_search_step"])
    oracle_flops = int(compute["oracle_conflict_core"]["accounted_neural_flops_per_search_step"])
    scope = TEST_ONLY_SCOPE if test_only else SCIENTIFIC_SCOPE
    rows: list[dict[str, Any]] = []
    lineage = seal.get("lineage") or {}

    for replicate in range(confirmatory_n):
        challenge_seed = derive_exp286_challenge_seed(
            protocol_digest=str(lineage["protocol_digest"]),
            freeze_commit_sha=str(seal["freeze_commit_sha"]),
            freeze_commit_timestamp_utc=str(seal["freeze_commit_timestamp_utc"]),
            beacon_receipt=beacon_receipt,
            stream=STREAM,
            replicate=replicate,
        )
        batch = Exp286ConflictGenerator(
            root_seed=f"NLM|EXP-286|POST_FREEZE_CHALLENGE|{challenge_seed}"
        ).make_batch(
            replicate=0,
            batch_size=int(config["batch_size"]),
            timesteps=int(config["timesteps"]),
            variables=int(config["variables"]),
            decoys=int(config["decoys"]),
            d_model=int(config["d_model"]),
            noise_std=float(config["noise_std"]),
            rng_stream="evaluation",
            scope=scope,
            device="cpu",
        )
        row: dict[str, Any] = {
            "replicate": replicate,
            "challenge_seed": int(challenge_seed),
            "challenge_batch_digest": batch.digest,
            "challenge_scope": scope,
            "chronological_failure": _search_batch(
                arm_id="chronological_failure",
                arm=chronological,
                batch=batch,
                per_step_flops=chronological_flops,
                ceiling=ceiling,
                max_search_steps=int(config["max_search_steps"]),
            ),
            "oracle_conflict_core": _search_batch(
                arm_id="oracle_conflict_core",
                arm=oracle,
                batch=batch,
                per_step_flops=oracle_flops,
                ceiling=ceiling,
                max_search_steps=int(config["max_search_steps"]),
            ),
            "row_digest": "",
        }
        row["row_digest"] = _row_digest(row)
        rows.append(row)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": not test_only,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_n": confirmatory_n,
        "confirmatory_data_consumed": not test_only,
        "synthetic_challenge_data_consumed": test_only,
        "challenge_materialized": True,
        "seed_materialization_status": "TEST_ONLY_EXECUTED" if test_only else "EXECUTED",
        "decision_rule_executed": False,
        "seal_digest": seal.get("seal_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "beacon_receipt": deepcopy(beacon_receipt),
        "lineage": {
            "protocol_digest": lineage.get("protocol_digest"),
            "gate_a_seal_digest": seal.get("seal_digest"),
            "challenge_contract_digest": seal.get("challenge_contract_digest"),
            "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
            "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
            "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "source_tree_digest": current_source_tree_digest,
            "executor_code_digest": executor_code_digest,
        },
        "per_replicate": rows,
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp286_confirmatory_raw(payload)
    if errors:
        raise RuntimeError("invalid EXP-286 confirmatory raw artifact: " + "; ".join(errors))
    return payload


def validate_exp286_confirmatory_raw(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-286 raw artifact must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-286 raw artifact identity")
    test_only = payload.get("test_only")
    if not isinstance(test_only, bool):
        errors.append("EXP-286 raw evidence mode is invalid")
    elif test_only:
        if payload.get("status") != TEST_ONLY_STATUS or payload.get("scientific_evidence_eligible") is not False:
            errors.append("TEST-ONLY EXP-286 raw classification drift")
        if payload.get("confirmatory_data_consumed") is not False or payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("TEST-ONLY EXP-286 raw consumption boundary drift")
    else:
        if payload.get("status") != SCIENTIFIC_STATUS or payload.get("scientific_evidence_eligible") is not True:
            errors.append("scientific EXP-286 raw classification drift")
        if payload.get("confirmatory_data_consumed") is not True or payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("scientific EXP-286 raw consumption boundary drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED" or payload.get("decision_rule_executed") is not False:
        errors.append("EXP-286 raw stage cannot promote or execute the decision rule")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-286 raw challenge must be materialized")
    confirmatory_n = payload.get("confirmatory_n")
    rows = payload.get("per_replicate")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not 32 <= confirmatory_n <= 128:
        errors.append("EXP-286 raw confirmatory_n is outside frozen bounds")
    if not isinstance(rows, list) or not isinstance(confirmatory_n, int) or len(rows) != confirmatory_n:
        errors.append("EXP-286 raw replicate count mismatch")
    elif [row.get("replicate") for row in rows] != list(range(confirmatory_n)):
        errors.append("EXP-286 raw replicate lineage must be contiguous and ordered")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or row.get("row_digest") != _row_digest(row):
                errors.append("EXP-286 raw row digest mismatch")
                break
    beacon = payload.get("beacon_receipt") or {}
    if payload.get("lineage", {}).get("beacon_receipt_digest") != beacon.get("receipt_digest"):
        errors.append("EXP-286 raw beacon lineage mismatch")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-286 raw artifact digest mismatch")
    return errors
