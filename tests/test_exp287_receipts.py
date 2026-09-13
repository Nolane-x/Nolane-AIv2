from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp287_learned_conflict_localization import (
    ESTABLISHED,
    NOT_ESTABLISHED,
    ORACLE_NOT_REPLICATED,
    classify_exp287_root_metrics,
)
from nolane_ai.experiments.exp287_receipts import (
    CROSS_SCHEMA,
    ROOT_SCHEMA,
    build_exp287_root_receipt,
    canonical_json_bytes,
    receipt_sha256,
    reduce_exp287_cross_root,
    validate_exp287_cross_receipt,
    validate_exp287_root_receipt,
)


PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
ROOT_PREFIX = "20260913-exp287-learned-conflict-localization-v1-dev"
CODE_DIGEST = "a" * 64
GEOMETRY = {
    "root_prefix": ROOT_PREFIX,
    "canonical_indices": [0, 1, 2, 3],
    "train_replicates": 64,
    "eval_replicates": 32,
    "eval_start_replicate": 10000,
    "batch_size": 8,
    "d_model": 64,
    "hidden_size": 48,
    "target_parameters": 500000,
    "timesteps": 4,
    "variables": 8,
    "decoys": 3,
    "max_search_steps": 16,
    "noise_std": 0.05,
    "lr": 0.002,
    "weight_decay": 0.0,
    "top_k": 2,
    "capture_threshold": 0.5,
    "precision_threshold": 0.5,
    "solution_rate_floor_delta": -0.005,
}


def _batch_receipt(index: int, replicate: int, *, model_digest: str) -> dict:
    mode = lambda cost, solution, oracle, learned_oracle=False: {
        "mean_accounted_cost": float(cost),
        "verified_solution_rate": float(solution),
        "oracle_information_delivered": bool(oracle),
        "learned_oracle_information_delivered": bool(learned_oracle),
        "censored_episode_count": 0,
        "episodes": [],
    }
    return {
        "schema": "NLM-EXP-287-SAME-WEIGHTS-BATCH-EVAL-V1",
        "experiment_id": "EXP-287",
        "replicate": replicate,
        "rng_stream": "evaluation",
        "batch_digest": hashlib.sha256(f"batch-{index}-{replicate}".encode()).hexdigest(),
        "model_state_digest": model_digest,
        "modes": {
            "NULL_CORE_CONTROL": mode(100.0, 0.90, False),
            "LEARNED_CORE_TOP2": mode(70.0, 0.90, False),
            "ORACLE_CORE_UPPER_BOUND": mode(50.0, 0.90, True),
        },
        "localization": {
            "evaluated_variables": 64,
            "episodes": 8,
            "selected_true_core_members": 12,
            "selected_core_members": 16,
            "top2_core_precision": 0.75,
            "top2_core_recall": 0.75,
            "exact_core_recovery_rate": 0.50,
            "off_core_selection_rate": 0.25,
            "variable_average_precision": 0.80,
            "raw_core_prevalence": 0.25,
        },
        "precontradiction_conflict_delivery": False,
    }


def _primitive_root(index: int, *, established: bool = True) -> dict:
    model_digest = hashlib.sha256(f"model-{index}".encode()).hexdigest()
    if established:
        metrics = {
            "mean_control_cost": 100.0,
            "mean_learned_cost": 70.0,
            "mean_oracle_cost": 50.0,
            "control_solution_rate": 0.90,
            "learned_solution_rate": 0.90,
            "oracle_solution_rate": 0.91,
            "top2_core_precision": 0.75,
            "learned_oracle_information_delivered": False,
            "precontradiction_conflict_delivery": False,
        }
    else:
        metrics = {
            "mean_control_cost": 100.0,
            "mean_learned_cost": 95.0,
            "mean_oracle_cost": 50.0,
            "control_solution_rate": 0.90,
            "learned_solution_rate": 0.90,
            "oracle_solution_rate": 0.91,
            "top2_core_precision": 0.50,
            "learned_oracle_information_delivered": False,
            "precontradiction_conflict_delivery": False,
        }
    return {
        "schema": "NLM-EXP-287-LEARNED-CONFLICT-LOCALIZATION-ROOT-V1",
        "experiment_id": "EXP-287",
        "canonical_index": index,
        "root_seed": f"{ROOT_PREFIX}::{index}",
        "geometry": deepcopy(GEOMETRY),
        "geometry_digest": hashlib.sha256(b"frozen-exp287-geometry").hexdigest(),
        "code_digest": CODE_DIGEST,
        "protocol_digest": PROTOCOL_DIGEST,
        "model_init_seed": 1000 + index,
        "model_state_digest": model_digest,
        "training": {
            "rng_stream": "augmentation",
            "replicates": 64,
            "loss_formula": "rollback_ce + verifier_bce + localizer_bce",
            "loss_coefficients": [1.0, 1.0, 1.0],
            "first_loss": 2.0,
            "last_loss": 0.5,
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "replicates": 32,
            "start_replicate": 10000,
            "batch_receipts": [
                _batch_receipt(index, 10000 + offset, model_digest=model_digest)
                for offset in range(32)
            ],
        },
        "mode_audit": {
            "modes": [
                "NULL_CORE_CONTROL",
                "LEARNED_CORE_TOP2",
                "ORACLE_CORE_UPPER_BOUND",
            ],
            "parameter_inventory_shared": True,
            "active_functional_parameters_shared": True,
            "optimizer_visible_parameters_shared": True,
            "accounted_flops_per_search_step_shared": True,
            "localizer_flops_charged_in_all_modes": True,
            "oracle_mode_deployable": False,
        },
        "localization": {
            "evaluated_variables": 2048,
            "episodes": 256,
            "selected_true_core_members": int(metrics["top2_core_precision"] * 512),
            "selected_core_members": 512,
            "top2_core_precision": metrics["top2_core_precision"],
            "top2_core_recall": metrics["top2_core_precision"],
            "exact_core_recovery_rate": 0.5,
            "off_core_selection_rate": 1.0 - metrics["top2_core_precision"],
            "variable_average_precision": 0.8,
            "raw_core_prevalence": 0.25,
        },
        "primitive_metrics": metrics,
        "decision": classify_exp287_root_metrics(metrics),
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "evaluation_labels_used_for_training": False,
        "evaluation_core_used_by_learned_mode": False,
        "oracle_mode_deployable": False,
    }


def _root(index: int, *, established: bool = True) -> dict:
    return build_exp287_root_receipt(_primitive_root(index, established=established))


def test_canonical_json_roundtrip_and_sha_are_byte_stable() -> None:
    receipt = _root(0)
    payload = canonical_json_bytes(receipt)
    reparsed = json.loads(payload)
    assert canonical_json_bytes(reparsed) == payload
    assert receipt_sha256(receipt) == hashlib.sha256(payload).hexdigest()


def test_root_builder_recomputes_classification_and_digest() -> None:
    primitive = _primitive_root(0)
    primitive["decision"] = {"classification": "FORGED_FAVORABLE_RESULT"}
    receipt = build_exp287_root_receipt(primitive)
    assert receipt["schema"] == ROOT_SCHEMA
    assert receipt["decision"]["classification"] == ESTABLISHED
    assert len(receipt["artifact_digest"]) == 64
    validate_exp287_root_receipt(receipt)


def test_root_validator_rejects_forged_classification_even_after_rehash() -> None:
    receipt = _root(0)
    receipt["decision"]["classification"] = NOT_ESTABLISHED
    receipt["artifact_digest"] = receipt_sha256({k: v for k, v in receipt.items() if k != "artifact_digest"})
    with pytest.raises(ValueError, match="classification|decision"):
        validate_exp287_root_receipt(receipt)


def test_root_validator_rejects_wrong_root_identity_or_geometry() -> None:
    receipt = _root(0)
    receipt["root_seed"] = f"{ROOT_PREFIX}::3"
    with pytest.raises(ValueError, match="root"):
        validate_exp287_root_receipt(receipt)

    receipt = _root(0)
    receipt["geometry"]["top_k"] = 3
    with pytest.raises(ValueError, match="geometry"):
        validate_exp287_root_receipt(receipt)


def test_root_validator_rejects_training_evaluation_overlap() -> None:
    receipt = _root(0)
    receipt["evaluation"]["start_replicate"] = 32
    with pytest.raises(ValueError, match="lineage|overlap"):
        validate_exp287_root_receipt(receipt)


def test_root_validator_rejects_model_digest_drift_inside_heldout_receipts() -> None:
    receipt = _root(0)
    receipt["evaluation"]["batch_receipts"][7]["model_state_digest"] = "0" * 64
    with pytest.raises(ValueError, match="model.*digest|digest.*model"):
        validate_exp287_root_receipt(receipt)


def test_root_validator_rejects_learned_oracle_leakage_or_precontradiction_delivery() -> None:
    receipt = _root(0)
    receipt["evaluation"]["batch_receipts"][3]["modes"]["LEARNED_CORE_TOP2"][
        "learned_oracle_information_delivered"
    ] = True
    with pytest.raises(ValueError, match="oracle|leak"):
        validate_exp287_root_receipt(receipt)

    receipt = _root(0)
    receipt["evaluation"]["batch_receipts"][4]["precontradiction_conflict_delivery"] = True
    with pytest.raises(ValueError, match="precontradiction|before.*contradiction"):
        validate_exp287_root_receipt(receipt)


def test_root_validator_rejects_evidence_boundary_overclaim() -> None:
    receipt = _root(0)
    receipt["scientific_evidence_eligible"] = True
    with pytest.raises(ValueError, match="scientific|boundary"):
        validate_exp287_root_receipt(receipt)


def test_cross_reducer_requires_exactly_four_unique_canonical_roots() -> None:
    roots = [_root(0), _root(1), _root(2), _root(2)]
    with pytest.raises(ValueError, match="duplicate|canonical"):
        reduce_exp287_cross_root(roots)

    with pytest.raises(ValueError, match="four|4"):
        reduce_exp287_cross_root([_root(0), _root(1), _root(2)])


def test_cross_recurrent_is_only_authorizing_disposition() -> None:
    cross = reduce_exp287_cross_root([_root(0), _root(1), _root(2), _root(3)])
    assert cross["schema"] == CROSS_SCHEMA
    assert cross["decision"] == "LOCALIZATION_VALUE_RECURRENT"
    assert cross["successor_design_authorized"] is True
    assert cross["authorization_scope"] == "DESIGN_EXP288_BACKJUMP_COURT_ONLY"
    assert cross["mechanism_successor_authorized"] is False
    assert cross["scientific_evidence_eligible"] is False
    assert cross["confirmatory_data_consumed"] is False
    assert cross["challenge_materialized"] is False
    assert cross["promotion_claimed"] is False
    validate_exp287_cross_receipt(cross)


def test_cross_intermittent_is_fail_closed() -> None:
    cross = reduce_exp287_cross_root(
        [_root(0), _root(1), _root(2), _root(3, established=False)]
    )
    assert cross["decision"] == "LOCALIZATION_VALUE_INTERMITTENT"
    assert cross["successor_design_authorized"] is False
    assert cross["authorization_scope"] == "NONE"
    validate_exp287_cross_receipt(cross)


def test_cross_oracle_replication_incomplete_outranks_learned_negative() -> None:
    primitive = _primitive_root(3)
    primitive["primitive_metrics"]["mean_oracle_cost"] = 101.0
    primitive["decision"] = classify_exp287_root_metrics(primitive["primitive_metrics"])
    assert primitive["decision"]["classification"] == ORACLE_NOT_REPLICATED
    cross = reduce_exp287_cross_root([_root(0), _root(1), _root(2), build_exp287_root_receipt(primitive)])
    assert cross["decision"] == "ORACLE_REPLICATION_INCOMPLETE"
    assert cross["successor_design_authorized"] is False


def test_cross_validator_rejects_forged_cross_decision_even_after_rehash() -> None:
    cross = reduce_exp287_cross_root(
        [_root(0), _root(1), _root(2), _root(3, established=False)]
    )
    cross["decision"] = "LOCALIZATION_VALUE_RECURRENT"
    cross["successor_design_authorized"] = True
    cross["authorization_scope"] = "DESIGN_EXP288_BACKJUMP_COURT_ONLY"
    cross["artifact_digest"] = receipt_sha256({k: v for k, v in cross.items() if k != "artifact_digest"})
    with pytest.raises(ValueError, match="decision|authorization"):
        validate_exp287_cross_receipt(cross)
