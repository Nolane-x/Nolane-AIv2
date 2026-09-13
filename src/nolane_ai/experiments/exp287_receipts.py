from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from .exp287_learned_conflict_localization import (
    ESTABLISHED,
    NOT_ESTABLISHED,
    ORACLE_NOT_REPLICATED,
    SCHEMA as ROOT_SCHEMA,
    classify_exp287_root_metrics,
)
from .matched_conflict_localizer_arms import LEARNED_MODE, NULL_MODE, ORACLE_MODE


CROSS_SCHEMA = "NLM-EXP-287-LEARNED-CONFLICT-LOCALIZATION-CROSS-V1"
PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
ROOT_PREFIX = "20260913-exp287-learned-conflict-localization-v1-dev"
FROZEN_GEOMETRY: dict[str, Any] = {
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
_MODES = {NULL_MODE, LEARNED_MODE, ORACLE_MODE}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_BOUNDARY_FALSE_KEYS = (
    "scientific_evidence_eligible",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
    "stage_a_protocol_modified",
    "evaluation_labels_used_for_training",
    "evaluation_core_used_by_learned_mode",
    "oracle_mode_deployable",
)


def canonical_json_bytes(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def receipt_sha256(receipt_or_bytes: dict[str, Any] | bytes) -> str:
    payload = (
        receipt_or_bytes
        if isinstance(receipt_or_bytes, bytes)
        else canonical_json_bytes(receipt_or_bytes)
    )
    return hashlib.sha256(payload).hexdigest()


def _artifact_digest(receipt: dict[str, Any]) -> str:
    clean = deepcopy(receipt)
    clean.pop("artifact_digest", None)
    return receipt_sha256(clean)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_hex64(value: Any, field: str) -> None:
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"EXP-287 {field} must be lowercase sha256",
    )


def _validate_boundaries(receipt: dict[str, Any], *, context: str) -> None:
    for key in _BOUNDARY_FALSE_KEYS:
        _require(
            receipt.get(key) is False,
            f"EXP-287 {context} scientific boundary {key} must remain false",
        )


def _validate_root_identity(receipt: dict[str, Any]) -> None:
    _require(receipt.get("schema") == ROOT_SCHEMA, "EXP-287 root schema mismatch")
    _require(
        receipt.get("experiment_id") == "EXP-287",
        "EXP-287 root experiment identity mismatch",
    )
    index = receipt.get("canonical_index")
    _require(index in {0, 1, 2, 3}, "EXP-287 root canonical index must be 0..3")
    _require(
        receipt.get("root_seed") == f"{ROOT_PREFIX}::{index}",
        "EXP-287 root identity mismatch",
    )
    _require(receipt.get("geometry") == FROZEN_GEOMETRY, "EXP-287 frozen geometry mismatch")
    _require_hex64(receipt.get("geometry_digest"), "geometry_digest")
    _require_hex64(receipt.get("code_digest"), "code_digest")
    _require(
        receipt.get("protocol_digest") == PROTOCOL_DIGEST,
        "EXP-287 protocol digest mismatch",
    )
    _require(
        isinstance(receipt.get("model_init_seed"), int),
        "EXP-287 model init seed must be integer",
    )
    _require_hex64(receipt.get("model_state_digest"), "model_state_digest")


def _validate_training_evaluation_contract(receipt: dict[str, Any]) -> None:
    training = receipt.get("training")
    evaluation = receipt.get("evaluation")
    _require(isinstance(training, dict), "EXP-287 training receipt missing")
    _require(isinstance(evaluation, dict), "EXP-287 evaluation receipt missing")
    _require(
        training.get("rng_stream") == "augmentation",
        "EXP-287 training lineage must be augmentation",
    )
    _require(training.get("replicates") == 64, "EXP-287 training replicate count mismatch")
    _require(
        training.get("loss_formula") == "rollback_ce + verifier_bce + localizer_bce",
        "EXP-287 training loss formula drift",
    )
    _require(
        training.get("loss_coefficients") == [1.0, 1.0, 1.0],
        "EXP-287 training loss weights drift",
    )
    _require(
        evaluation.get("rng_stream") == "evaluation",
        "EXP-287 heldout lineage must be evaluation",
    )
    _require(evaluation.get("replicates") == 32, "EXP-287 evaluation replicate count mismatch")
    eval_start = evaluation.get("start_replicate")
    _require(isinstance(eval_start, int), "EXP-287 evaluation start replicate must be integer")
    _require(eval_start >= 64, "EXP-287 training/evaluation lineage overlap")
    _require(eval_start == 10000, "EXP-287 evaluation start replicate drift")

    batches = evaluation.get("batch_receipts")
    _require(
        isinstance(batches, list) and len(batches) == 32,
        "EXP-287 heldout batch receipt count mismatch",
    )
    expected_model_digest = receipt["model_state_digest"]
    expected_replicates = list(range(10000, 10032))
    observed_replicates: list[int] = []
    for row in batches:
        _require(isinstance(row, dict), "EXP-287 heldout batch receipt must be object")
        _require(
            row.get("schema") == "NLM-EXP-287-SAME-WEIGHTS-BATCH-EVAL-V1",
            "EXP-287 heldout batch schema mismatch",
        )
        _require(
            row.get("experiment_id") == "EXP-287",
            "EXP-287 heldout batch identity mismatch",
        )
        _require(
            row.get("rng_stream") == "evaluation",
            "EXP-287 heldout batch stream mismatch",
        )
        observed_replicates.append(int(row.get("replicate", -1)))
        _require_hex64(row.get("batch_digest"), "heldout batch digest")
        _require(
            row.get("model_state_digest") == expected_model_digest,
            "EXP-287 model digest drift inside heldout receipts",
        )
        modes = row.get("modes")
        _require(
            isinstance(modes, dict) and set(modes) == _MODES,
            "EXP-287 heldout information-mode set mismatch",
        )
        learned = modes[LEARNED_MODE]
        null = modes[NULL_MODE]
        oracle = modes[ORACLE_MODE]
        _require(
            learned.get("learned_oracle_information_delivered") is False
            and learned.get("oracle_information_delivered") is False,
            "EXP-287 learned mode oracle leakage detected",
        )
        _require(
            null.get("oracle_information_delivered") is False,
            "EXP-287 null control cannot receive oracle information",
        )
        _require(
            oracle.get("oracle_information_delivered") is True,
            "EXP-287 oracle mode did not record oracle delivery",
        )
        _require(
            row.get("precontradiction_conflict_delivery") is False,
            "EXP-287 precontradiction conflict delivery detected",
        )
    _require(
        observed_replicates == expected_replicates,
        "EXP-287 heldout replicate lineage drift",
    )


def _validate_mode_audit(receipt: dict[str, Any]) -> None:
    audit = receipt.get("mode_audit")
    _require(isinstance(audit, dict), "EXP-287 matched mode audit missing")
    _require(
        set(audit.get("modes", [])) == _MODES,
        "EXP-287 matched mode audit mode set mismatch",
    )
    for key in (
        "parameter_inventory_shared",
        "active_functional_parameters_shared",
        "optimizer_visible_parameters_shared",
        "accounted_flops_per_search_step_shared",
        "localizer_flops_charged_in_all_modes",
    ):
        _require(
            audit.get(key) is True,
            f"EXP-287 matched resource audit {key} did not close",
        )
    _require(
        audit.get("oracle_mode_deployable") is False,
        "EXP-287 oracle mode must remain non-deployable",
    )


def _validate_metrics(receipt: dict[str, Any]) -> dict[str, Any]:
    metrics = receipt.get("primitive_metrics")
    _require(isinstance(metrics, dict), "EXP-287 primitive metrics missing")
    required = {
        "mean_control_cost",
        "mean_learned_cost",
        "mean_oracle_cost",
        "control_solution_rate",
        "learned_solution_rate",
        "oracle_solution_rate",
        "top2_core_precision",
        "learned_oracle_information_delivered",
        "precontradiction_conflict_delivery",
    }
    _require(required.issubset(metrics), "EXP-287 primitive metrics incomplete")
    for key in ("mean_control_cost", "mean_learned_cost", "mean_oracle_cost"):
        _require(float(metrics[key]) >= 0.0, f"EXP-287 {key} must be non-negative")
    for key in (
        "control_solution_rate",
        "learned_solution_rate",
        "oracle_solution_rate",
        "top2_core_precision",
    ):
        value = float(metrics[key])
        _require(0.0 <= value <= 1.0, f"EXP-287 {key} must be in [0,1]")
    _require(
        metrics["learned_oracle_information_delivered"] is False,
        "EXP-287 primitive metrics report learned oracle leakage",
    )
    _require(
        metrics["precontradiction_conflict_delivery"] is False,
        "EXP-287 primitive metrics report precontradiction delivery",
    )

    localization = receipt.get("localization")
    _require(isinstance(localization, dict), "EXP-287 localization diagnostics missing")
    _require(localization.get("episodes") == 256, "EXP-287 localization episode count mismatch")
    _require(
        localization.get("evaluated_variables") == 2048,
        "EXP-287 localization variable count mismatch",
    )
    _require(
        localization.get("selected_core_members") == 512,
        "EXP-287 top-2 selection count mismatch",
    )
    _require(
        float(localization.get("top2_core_precision"))
        == float(metrics["top2_core_precision"]),
        "EXP-287 localization precision and primitive metric disagree",
    )
    _require(
        float(localization.get("raw_core_prevalence")) == 0.25,
        "EXP-287 raw core prevalence drift",
    )
    return metrics


def _validate_semantic_decision(receipt: dict[str, Any]) -> None:
    metrics = _validate_metrics(receipt)
    expected = classify_exp287_root_metrics(metrics)
    _require(
        receipt.get("decision") == expected,
        "EXP-287 forged root decision/classification",
    )


def validate_exp287_root_receipt(receipt: dict[str, Any]) -> None:
    _validate_root_identity(receipt)
    _validate_training_evaluation_contract(receipt)
    _validate_mode_audit(receipt)
    _validate_boundaries(receipt, context="root")
    _validate_semantic_decision(receipt)
    _require_hex64(receipt.get("artifact_digest"), "artifact_digest")
    _require(
        receipt["artifact_digest"] == _artifact_digest(receipt),
        "EXP-287 root artifact digest mismatch",
    )


def build_exp287_root_receipt(primitive_payload: dict[str, Any]) -> dict[str, Any]:
    receipt = deepcopy(primitive_payload)
    receipt["schema"] = ROOT_SCHEMA
    metrics = receipt.get("primitive_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("EXP-287 primitive metrics missing")
    receipt["decision"] = classify_exp287_root_metrics(metrics)
    receipt.pop("artifact_digest", None)
    receipt["artifact_digest"] = _artifact_digest(receipt)
    validate_exp287_root_receipt(receipt)
    return receipt


def _cross_decision(root_classes: list[str]) -> str:
    if any(value == ORACLE_NOT_REPLICATED for value in root_classes):
        return "ORACLE_REPLICATION_INCOMPLETE"
    established = sum(value == ESTABLISHED for value in root_classes)
    if established == 4:
        return "LOCALIZATION_VALUE_RECURRENT"
    if established > 0:
        return "LOCALIZATION_VALUE_INTERMITTENT"
    return "LOCALIZATION_VALUE_NOT_ESTABLISHED"


def _validate_cross_identity(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _require(
        len(roots) == 4,
        "EXP-287 cross reducer requires exactly four canonical roots",
    )
    for root in roots:
        validate_exp287_root_receipt(root)
    ordered = sorted(roots, key=lambda row: int(row["canonical_index"]))
    indices = [int(row["canonical_index"]) for row in ordered]
    _require(
        indices == [0, 1, 2, 3],
        "EXP-287 cross reducer has duplicate or missing canonical root",
    )
    _require(
        all(row["protocol_digest"] == PROTOCOL_DIGEST for row in ordered),
        "EXP-287 cross protocol identity mismatch",
    )
    code_digests = {str(row["code_digest"]) for row in ordered}
    _require(len(code_digests) == 1, "EXP-287 cross code digest identity mismatch")
    geometry_digests = {row["geometry_digest"] for row in ordered}
    _require(len(geometry_digests) == 1, "EXP-287 cross geometry digest mismatch")
    model_digests = [row["model_state_digest"] for row in ordered]
    _require(
        len(set(model_digests)) == 4,
        "EXP-287 canonical root model digests must be independently rooted",
    )
    return ordered


def reduce_exp287_cross_root(roots: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = _validate_cross_identity(roots)
    classes = [str(row["decision"]["classification"]) for row in ordered]
    decision = _cross_decision(classes)
    authorized = decision == "LOCALIZATION_VALUE_RECURRENT"
    receipt: dict[str, Any] = {
        "schema": CROSS_SCHEMA,
        "experiment_id": "EXP-287",
        "evidence_level": "EV-E2",
        "analysis_scope": "development_learned_conflict_localization_same_weights_oracle_value_capture",
        "protocol_digest": PROTOCOL_DIGEST,
        "code_digest": ordered[0]["code_digest"],
        "geometry": deepcopy(FROZEN_GEOMETRY),
        "geometry_digest": ordered[0]["geometry_digest"],
        "canonical_indices": [0, 1, 2, 3],
        "root_artifact_digests": {
            str(row["canonical_index"]): row["artifact_digest"] for row in ordered
        },
        "root_model_digests": {
            str(row["canonical_index"]): row["model_state_digest"] for row in ordered
        },
        "root_classifications": {
            str(row["canonical_index"]): row["decision"]["classification"]
            for row in ordered
        },
        "root_decisions": {
            str(row["canonical_index"]): deepcopy(row["decision"])
            for row in ordered
        },
        "decision": decision,
        "successor_design_authorized": authorized,
        "authorization_scope": (
            "DESIGN_EXP288_BACKJUMP_COURT_ONLY" if authorized else "NONE"
        ),
        "mechanism_successor_authorized": False,
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "evaluation_labels_used_for_training": False,
        "evaluation_core_used_by_learned_mode": False,
        "oracle_mode_deployable": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    validate_exp287_cross_receipt(receipt)
    return receipt


def validate_exp287_cross_receipt(receipt: dict[str, Any]) -> None:
    _require(receipt.get("schema") == CROSS_SCHEMA, "EXP-287 cross schema mismatch")
    _require(
        receipt.get("experiment_id") == "EXP-287",
        "EXP-287 cross experiment identity mismatch",
    )
    _require(
        receipt.get("evidence_level") == "EV-E2",
        "EXP-287 cross evidence level must remain EV-E2",
    )
    _require(
        receipt.get("analysis_scope")
        == "development_learned_conflict_localization_same_weights_oracle_value_capture",
        "EXP-287 cross analysis scope mismatch",
    )
    _require(
        receipt.get("protocol_digest") == PROTOCOL_DIGEST,
        "EXP-287 cross protocol digest mismatch",
    )
    _require_hex64(receipt.get("code_digest"), "cross code_digest")
    _require(
        receipt.get("geometry") == FROZEN_GEOMETRY,
        "EXP-287 cross frozen geometry mismatch",
    )
    _require_hex64(receipt.get("geometry_digest"), "cross geometry_digest")
    _require(
        receipt.get("canonical_indices") == [0, 1, 2, 3],
        "EXP-287 cross canonical index set mismatch",
    )

    classes = receipt.get("root_classifications")
    decisions = receipt.get("root_decisions")
    artifact_digests = receipt.get("root_artifact_digests")
    model_digests = receipt.get("root_model_digests")
    expected_keys = {"0", "1", "2", "3"}
    _require(
        isinstance(classes, dict) and set(classes) == expected_keys,
        "EXP-287 cross root classification keys mismatch",
    )
    _require(
        isinstance(decisions, dict) and set(decisions) == expected_keys,
        "EXP-287 cross root decision keys mismatch",
    )
    _require(
        isinstance(artifact_digests, dict) and set(artifact_digests) == expected_keys,
        "EXP-287 cross root artifact keys mismatch",
    )
    _require(
        isinstance(model_digests, dict) and set(model_digests) == expected_keys,
        "EXP-287 cross root model keys mismatch",
    )
    for key in expected_keys:
        _require_hex64(artifact_digests[key], f"root artifact digest {key}")
        _require_hex64(model_digests[key], f"root model digest {key}")
        root_decision = decisions[key]
        _require(
            isinstance(root_decision, dict),
            f"EXP-287 cross root decision {key} missing",
        )
        _require(
            root_decision.get("classification") == classes[key],
            f"EXP-287 cross root decision/classification mismatch {key}",
        )
        _require(
            classes[key] in {ESTABLISHED, NOT_ESTABLISHED, ORACLE_NOT_REPLICATED},
            f"EXP-287 unknown root classification {key}",
        )

    expected_decision = _cross_decision(
        [str(classes[str(index)]) for index in range(4)]
    )
    _require(
        receipt.get("decision") == expected_decision,
        "EXP-287 forged cross decision",
    )
    expected_authorized = expected_decision == "LOCALIZATION_VALUE_RECURRENT"
    _require(
        receipt.get("successor_design_authorized") is expected_authorized,
        "EXP-287 cross authorization inconsistent with decision",
    )
    _require(
        receipt.get("authorization_scope")
        == ("DESIGN_EXP288_BACKJUMP_COURT_ONLY" if expected_authorized else "NONE"),
        "EXP-287 cross authorization scope inconsistent with decision",
    )
    _require(
        receipt.get("mechanism_successor_authorized") is False,
        "EXP-287 cannot authorize a mechanism successor",
    )
    _validate_boundaries(receipt, context="cross")
    _require_hex64(receipt.get("artifact_digest"), "cross artifact_digest")
    _require(
        receipt["artifact_digest"] == _artifact_digest(receipt),
        "EXP-287 cross artifact digest mismatch",
    )
