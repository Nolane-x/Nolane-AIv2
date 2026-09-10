from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256, require_frozen_stage_a_v1_sha256
from .exp289_beacon import derive_exp289_challenge_seed, validate_exp289_beacon_receipt
from .exp289_challenge_worlds import challenge_contract_digest
from .exp289_checkpoint import (
    load_exp289_trained_checkpoint,
    validate_exp289_checkpoint_receipt,
)
from .exp289_confirmatory_ceremony import validate_exp289_confirmatory_gate_a_seal
from .exp289_nogood_worlds import Exp289NogoodGenerator
from .exp289_paired_runner import _json_digest, _run_episode, _summarize_arm
from .matched_nogood_arms import audit_matched_exp289_arm_pair


SCHEMA = "NLM-EXP-289-CONFIRMATORY-CHALLENGE-RAW-V1"
EXPERIMENT_ID = "EXP-289"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
STREAM = "challenge"
SEARCH_PATH_PAIRING_POLICY = (
    "same_initial_world_restart_lineage_with_causal_post_memory_divergence_preserved"
)
TEST_ONLY_SCOPE = "synthetic-exp289-post-freeze-challenge-test-only"
SCIENTIFIC_SCOPE = "exp289-post-freeze-confirmatory-challenge"

_EXPECTED_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}


def _row_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("row_digest", None)
    return canonical_sha256(clean)


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _is_hex(value: Any, lengths: set[int] = {64}) -> bool:
    if not isinstance(value, str) or len(value) not in lengths:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _reserved_errors(value: Any, confirmatory_n: Any) -> list[str]:
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not 32 <= confirmatory_n <= 128:
        return ["EXP-289 raw confirmatory_n is outside frozen bounds"]
    if not isinstance(value, list) or len(value) != confirmatory_n:
        return ["EXP-289 raw reserved replicate count mismatch"]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        return ["EXP-289 raw reserved replicate IDs must be non-negative integers"]
    if len(set(value)) != len(value):
        return ["EXP-289 raw reserved replicate IDs must be unique"]
    if value and value != list(range(value[0], value[0] + len(value))):
        return ["EXP-289 raw reserved replicate IDs must be contiguous and ordered"]
    return []


def _beacon_mode(beacon_receipt: dict[str, Any]) -> bool:
    test_only = beacon_receipt.get("test_only")
    scientific = beacon_receipt.get("scientific_evidence_eligible")
    if test_only is True:
        if scientific is not False:
            raise ValueError("EXP-289 TEST-ONLY beacon cannot be scientific evidence")
        return True
    if test_only is False:
        if scientific is not True:
            raise ValueError(
                "EXP-289 non-test beacon must be explicitly marked scientific_evidence_eligible"
            )
        return False
    raise ValueError("EXP-289 beacon evidence mode is invalid")


def _validate_execution_inputs(
    *,
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], list[int], Any, Any, dict[str, Any]]:
    seal_errors = validate_exp289_confirmatory_gate_a_seal(seal)
    if seal_errors:
        raise ValueError("invalid EXP-289 Gate-A seal: " + "; ".join(seal_errors))
    if not _is_hex(current_source_tree_digest) or current_source_tree_digest != seal.get("code_tree_digest"):
        raise ValueError("EXP-289 challenge source-tree digest does not match sealed code tree")
    if not _is_hex(executor_code_digest) or executor_code_digest != seal.get("code_tree_digest"):
        raise ValueError("EXP-289 challenge executor code digest does not match sealed code tree")
    if seal.get("challenge_contract_digest") != challenge_contract_digest():
        raise ValueError("EXP-289 challenge contract drift after Gate-A freeze")

    beacon_errors = validate_exp289_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=str(seal.get("freeze_commit_timestamp_utc") or ""),
    )
    if beacon_errors:
        raise ValueError("invalid EXP-289 beacon receipt: " + "; ".join(beacon_errors))
    _beacon_mode(beacon_receipt)

    checkpoint_errors = validate_exp289_checkpoint_receipt(checkpoint_receipt)
    if checkpoint_errors:
        raise ValueError("invalid EXP-289 checkpoint receipt: " + "; ".join(checkpoint_errors))
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"EXP-289 checkpoint file is missing: {path}")
    if file_sha256(path) != checkpoint_receipt.get("checkpoint_file_sha256"):
        raise ValueError("EXP-289 checkpoint file SHA256 mismatch before challenge execution")

    lineage = seal.get("lineage") or {}
    require_frozen_stage_a_v1_sha256(lineage.get("protocol_digest"))
    expected_checkpoint = {
        "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
        "checkpoint_execution_contract_digest": checkpoint_receipt.get("execution_contract_digest"),
    }
    for key, expected in expected_checkpoint.items():
        if lineage.get(key) != expected:
            raise ValueError(f"EXP-289 sealed checkpoint lineage mismatch: {key}")

    confirmatory_n = seal.get("confirmatory_n")
    reserved = deepcopy(seal.get("reserved_replicate_ids"))
    reserved_errors = _reserved_errors(reserved, confirmatory_n)
    if reserved_errors:
        raise ValueError("; ".join(reserved_errors))
    authorization = seal.get("execution_authorization") or {}
    if authorization.get("reserved_replicate_ids") != reserved:
        raise ValueError("EXP-289 reserved replicate IDs drifted after execution authorization")

    no_nogood, local_nogood = load_exp289_trained_checkpoint(
        checkpoint_path=path,
        receipt=checkpoint_receipt,
    )
    contract = checkpoint_receipt.get("execution_contract") or {}
    config = contract.get("configuration") or {}
    resource = contract.get("resource_binding") or {}
    try:
        pair_audit = audit_matched_exp289_arm_pair(
            no_nogood,
            local_nogood,
            restarts=int(config["restarts"]),
            variables=int(config["variables"]),
            max_search_steps=int(config["max_search_steps"]),
            max_accounted_cost_per_episode=int(resource["declared_max_accounted_cost_per_episode"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EXP-289 challenge checkpoint execution geometry is invalid") from exc
    if canonical_sha256(pair_audit) != resource.get("pair_audit_digest"):
        raise ValueError("EXP-289 challenge matched-pair audit reconstruction mismatch")
    return config, pair_audit, reserved, no_nogood, local_nogood, lineage


def _expected_rows(
    *,
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> list[dict[str, Any]]:
    config, pair_audit, reserved, no_nogood, local_nogood, lineage = _validate_execution_inputs(
        seal=seal,
        beacon_receipt=beacon_receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source_tree_digest,
        executor_code_digest=executor_code_digest,
    )
    test_only = beacon_receipt.get("test_only") is True
    challenge_scope = TEST_ONLY_SCOPE if test_only else SCIENTIFIC_SCOPE
    cache_key = canonical_sha256(
        {
            "seal_digest": seal.get("seal_digest"),
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
            "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
            "source_tree_digest": current_source_tree_digest,
            "executor_code_digest": executor_code_digest,
            "challenge_scope": challenge_scope,
        }
    )
    cached = _EXPECTED_ROWS_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    common_ceiling = int(pair_audit["declared_max_accounted_cost_per_episode"])
    no_flops = int(
        pair_audit["compute_ledger"]["no_nogood"]["accounted_neural_flops_per_search_step"]
    )
    local_flops = int(
        pair_audit["compute_ledger"]["local_nogood"]["accounted_neural_flops_per_search_step"]
    )
    rows: list[dict[str, Any]] = []
    for replicate in reserved:
        challenge_seed = derive_exp289_challenge_seed(
            protocol_digest=str(lineage["protocol_digest"]),
            freeze_commit_sha=str(seal["freeze_commit_sha"]),
            freeze_commit_timestamp_utc=str(seal["freeze_commit_timestamp_utc"]),
            beacon_receipt=beacon_receipt,
            stream=STREAM,
            replicate=replicate,
        )
        generator = Exp289NogoodGenerator(
            root_seed=f"NLM|EXP-289|POST_FREEZE_CHALLENGE|{challenge_seed}"
        )
        batch = generator.make_batch(
            replicate=0,
            batch_size=int(config["batch_size"]),
            timesteps=int(config["timesteps"]),
            restarts=int(config["restarts"]),
            variables=int(config["variables"]),
            decoys=int(config["decoys"]),
            d_model=int(config["d_model"]),
            noise_std=float(config["noise_std"]),
            rng_stream="evaluation",
            scope=challenge_scope,
            device="cpu",
        )
        no_episodes: list[dict[str, Any]] = []
        local_episodes: list[dict[str, Any]] = []
        world_receipts: list[dict[str, Any]] = []
        for episode_index, episode_metadata in enumerate(batch.metadata["episodes"]):
            no_episodes.append(
                _run_episode(
                    arm_name="no_nogood",
                    model=no_nogood,
                    batch=batch,
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    max_search_steps=int(config["max_search_steps"]),
                    neural_flops_per_step=no_flops,
                    common_ceiling=common_ceiling,
                )
            )
            local_episodes.append(
                _run_episode(
                    arm_name="local_nogood",
                    model=local_nogood,
                    batch=batch,
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    max_search_steps=int(config["max_search_steps"]),
                    neural_flops_per_step=local_flops,
                    common_ceiling=common_ceiling,
                )
            )
            manifest = deepcopy(episode_metadata["repeat_opportunities"])
            world_receipts.append(
                {
                    "episode_index": int(episode_index),
                    "episode_digest": episode_metadata["episode_digest"],
                    "problem_digest": episode_metadata["problem_digest"],
                    "valid_solutions": deepcopy(episode_metadata["valid_solutions"]),
                    "repeat_opportunities": manifest,
                    "predeclared_repeat_opportunities": int(
                        episode_metadata["predeclared_repeat_opportunities"]
                    ),
                    "opportunity_manifest_digest": _json_digest(manifest),
                    "evaluator_metadata_delivered_to_arm": False,
                }
            )
        row: dict[str, Any] = {
            "replicate": int(replicate),
            "challenge_seed": int(challenge_seed),
            "challenge_batch_digest": batch.digest,
            "challenge_seed_domain": "beacon_derived_root_then_frozen_exp289_generator",
            "challenge_scope": challenge_scope,
            "initial_world_pairing_closed": True,
            "opportunity_manifest_pairing_closed": True,
            "evaluator_metadata_delivered_to_arm": False,
            "search_path_pairing_policy": SEARCH_PATH_PAIRING_POLICY,
            "world_receipts": world_receipts,
            "no_nogood": _summarize_arm(
                arm_name="no_nogood",
                episodes=no_episodes,
                common_ceiling=common_ceiling,
            ),
            "local_nogood": _summarize_arm(
                arm_name="local_nogood",
                episodes=local_episodes,
                common_ceiling=common_ceiling,
            ),
            "row_digest": "",
        }
        row["row_digest"] = _row_digest(row)
        rows.append(row)

    _EXPECTED_ROWS_CACHE[cache_key] = deepcopy(rows)
    return rows


def execute_exp289_confirmatory_challenge(
    *,
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> dict[str, Any]:
    rows = _expected_rows(
        seal=seal,
        beacon_receipt=beacon_receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source_tree_digest,
        executor_code_digest=executor_code_digest,
    )
    test_only = beacon_receipt.get("test_only") is True
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": not test_only,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_data_consumed": not test_only,
        "synthetic_challenge_data_consumed": test_only,
        "challenge_materialized": True,
        "seed_materialization_status": "TEST_ONLY_EXECUTED" if test_only else "EXECUTED",
        "decision_rule_executed": False,
        "confirmatory_n": int(seal["confirmatory_n"]),
        "reserved_replicate_ids": deepcopy(seal["reserved_replicate_ids"]),
        "seal_digest": seal["seal_digest"],
        "checkpoint_scientific_identity_digest": checkpoint_receipt["scientific_identity_digest"],
        "beacon_receipt": deepcopy(beacon_receipt),
        "lineage": {
            "protocol_digest": (seal.get("lineage") or {}).get("protocol_digest"),
            "gate_a_seal_digest": seal.get("seal_digest"),
            "pre_beacon_binding_digest": seal.get("pre_beacon_binding_digest"),
            "challenge_contract_digest": seal.get("challenge_contract_digest"),
            "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
            "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
            "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
            "checkpoint_execution_contract_digest": checkpoint_receipt.get("execution_contract_digest"),
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "source_tree_digest": current_source_tree_digest,
            "executor_code_digest": executor_code_digest,
        },
        "per_replicate": rows,
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    errors = validate_exp289_confirmatory_raw(
        payload,
        seal=seal,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if errors:
        raise RuntimeError("invalid EXP-289 raw challenge artifact: " + "; ".join(errors))
    return payload


def validate_exp289_confirmatory_raw(
    payload: dict[str, Any],
    *,
    seal: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-289 raw challenge artifact must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-289 raw challenge identity")

    test_only = payload.get("test_only")
    scientific = payload.get("scientific_evidence_eligible")
    if test_only is True:
        expected_status = TEST_ONLY_STATUS
        expected_boundary = {
            "test_only": True,
            "scientific_evidence_eligible": False,
            "evidence_level": "EV-E2",
            "decision": "UNVERIFIED",
            "confirmatory_data_consumed": False,
            "synthetic_challenge_data_consumed": True,
            "challenge_materialized": True,
            "seed_materialization_status": "TEST_ONLY_EXECUTED",
            "decision_rule_executed": False,
        }
    elif test_only is False:
        expected_status = SCIENTIFIC_STATUS
        expected_boundary = {
            "test_only": False,
            "scientific_evidence_eligible": True,
            "evidence_level": "EV-E2",
            "decision": "UNVERIFIED",
            "confirmatory_data_consumed": True,
            "synthetic_challenge_data_consumed": False,
            "challenge_materialized": True,
            "seed_materialization_status": "EXECUTED",
            "decision_rule_executed": False,
        }
    else:
        expected_status = None
        expected_boundary = {}
        errors.append("EXP-289 raw evidence mode invalid")
    if payload.get("status") != expected_status:
        errors.append("EXP-289 raw challenge status/evidence-mode mismatch")
    for key, expected in expected_boundary.items():
        if payload.get(key) != expected:
            errors.append(f"EXP-289 raw evidence boundary drift: {key}")
    if not isinstance(scientific, bool):
        errors.append("EXP-289 raw scientific evidence flag invalid")

    seal_errors = validate_exp289_confirmatory_gate_a_seal(seal)
    if seal_errors:
        errors.append("EXP-289 raw Gate-A seal invalid: " + "; ".join(seal_errors))
        return errors
    if payload.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-289 raw/seal digest lineage mismatch")

    confirmatory_n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids")
    errors.extend(_reserved_errors(reserved, confirmatory_n))
    if reserved != seal.get("reserved_replicate_ids") or confirmatory_n != seal.get("confirmatory_n"):
        errors.append("EXP-289 raw reserved replicate lineage mismatch")

    beacon = payload.get("beacon_receipt")
    if not isinstance(beacon, dict):
        errors.append("EXP-289 raw beacon receipt missing")
        return errors
    beacon_errors = validate_exp289_beacon_receipt(
        beacon,
        freeze_commit_timestamp_utc=str(seal.get("freeze_commit_timestamp_utc") or ""),
    )
    if beacon_errors:
        errors.append("EXP-289 raw beacon invalid: " + "; ".join(beacon_errors))
    try:
        beacon_test_only = _beacon_mode(beacon)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if beacon_test_only is not test_only:
            errors.append("EXP-289 raw/beacon evidence mode mismatch")

    checkpoint_errors = validate_exp289_checkpoint_receipt(checkpoint_receipt)
    if checkpoint_errors:
        errors.append("EXP-289 raw checkpoint receipt invalid: " + "; ".join(checkpoint_errors))
        return errors
    path = Path(checkpoint_path)
    if not path.is_file() or file_sha256(path) != checkpoint_receipt.get("checkpoint_file_sha256"):
        errors.append("EXP-289 raw checkpoint file/hash mismatch")
        return errors
    if payload.get("checkpoint_scientific_identity_digest") != checkpoint_receipt.get("scientific_identity_digest"):
        errors.append("EXP-289 raw checkpoint scientific identity mismatch")

    lineage = payload.get("lineage") or {}
    sealed_lineage = seal.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": sealed_lineage.get("protocol_digest"),
        "gate_a_seal_digest": seal.get("seal_digest"),
        "pre_beacon_binding_digest": seal.get("pre_beacon_binding_digest"),
        "challenge_contract_digest": seal.get("challenge_contract_digest"),
        "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
        "checkpoint_execution_contract_digest": checkpoint_receipt.get("execution_contract_digest"),
        "beacon_receipt_digest": beacon.get("receipt_digest"),
        "source_tree_digest": seal.get("code_tree_digest"),
        "executor_code_digest": seal.get("code_tree_digest"),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            errors.append(f"EXP-289 raw lineage mismatch: {key}")
    try:
        require_frozen_stage_a_v1_sha256(lineage.get("protocol_digest"))
    except ValueError as exc:
        errors.append(str(exc))

    rows = payload.get("per_replicate")
    if not isinstance(rows, list) or len(rows) != confirmatory_n:
        errors.append("EXP-289 raw per-replicate row count mismatch")
    else:
        if [row.get("replicate") for row in rows if isinstance(row, dict)] != reserved:
            errors.append("EXP-289 raw replicate IDs are not the exact sealed reserved IDs")
        expected_scope = TEST_ONLY_SCOPE if test_only is True else SCIENTIFIC_SCOPE
        for row in rows:
            if not isinstance(row, dict) or row.get("row_digest") != _row_digest(row):
                errors.append("EXP-289 raw row digest mismatch")
                break
            if not isinstance(row.get("challenge_seed"), int) or isinstance(row.get("challenge_seed"), bool) or row.get("challenge_seed") < 0:
                errors.append("EXP-289 raw challenge seed invalid")
                break
            if not _is_hex(row.get("challenge_batch_digest")):
                errors.append("EXP-289 raw challenge batch digest invalid")
                break
            if row.get("challenge_scope") != expected_scope:
                errors.append("EXP-289 raw challenge scope/evidence-mode mismatch")
                break

    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-289 raw artifact digest mismatch")

    if not errors:
        try:
            expected_rows = _expected_rows(
                seal=seal,
                beacon_receipt=beacon,
                checkpoint_path=path,
                checkpoint_receipt=checkpoint_receipt,
                current_source_tree_digest=str(lineage.get("source_tree_digest") or ""),
                executor_code_digest=str(lineage.get("executor_code_digest") or ""),
            )
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"EXP-289 raw semantic reconstruction failed: {exc}")
        else:
            if rows != expected_rows:
                errors.append("EXP-289 raw semantic reconstruction mismatch")
    return errors
