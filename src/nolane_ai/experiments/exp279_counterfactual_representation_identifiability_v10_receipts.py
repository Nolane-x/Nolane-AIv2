from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256

from .exp279_counterfactual_representation_identifiability_v10 import (
    REPRESENTATION_DIMENSIONS,
    REPRESENTATION_VIEWS,
    classify_cross_budget,
    classify_representation_budget,
    classify_representation_root,
)
from .exp279_counterfactual_representation_identifiability_v10_runner import (
    FIT_REPLICATES_PER_ROOT,
    FROZEN_GEOMETRY,
    FROZEN_NN,
    FROZEN_OPTIMIZER,
    HELDOUT_REPLICATES_PER_ROOT,
    PROTOCOL_DIGEST,
    SCHEMA as SHARD_SCHEMA,
    build_root_schedule,
)

BUDGET_SCHEMA = "NLM-EXP-279-V10-CRIC-BUDGET-V1"
CROSS_SCHEMA = "NLM-EXP-279-V10-CRIC-CROSS-V1"
ANALYSIS_SCOPE = "development_counterfactual_representation_identifiability_upper_bound"
_MATCHED_AUDIT_KEYS = (
    "parameter_match",
    "functional_parameter_match",
    "active_functional_parameter_match",
    "optimizer_visible_parameter_match",
    "reclaimed_parameter_assignment_closed",
    "compute_budget_closed",
)
_BOUNDARY_FALSE_KEYS = (
    "scientific_evidence_eligible",
    "evaluation_rng_used",
    "evaluation_lineage_consumed",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
    "mechanism_successor_authorized",
    "raw_examples_exported",
    "raw_model_outputs_exported",
)
_OUTCOME_KEYS = {"RESCUE", "HARM", "BOTH_SUCCESS", "BOTH_FAILURE"}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")


def canonical_receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def receipt_sha256(receipt_or_bytes: dict[str, Any] | bytes) -> str:
    payload = (
        receipt_or_bytes
        if isinstance(receipt_or_bytes, bytes)
        else canonical_receipt_bytes(receipt_or_bytes)
    )
    return hashlib.sha256(payload).hexdigest()


def _artifact_digest(receipt: dict[str, Any]) -> str:
    clean = deepcopy(receipt)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_hex64(value: Any, field: str) -> None:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None, f"V10 {field} must be lowercase sha256")


def _require_hex40(value: Any, field: str) -> None:
    _require(isinstance(value, str) and _HEX40.fullmatch(value) is not None, f"V10 {field} must be 40-hex Git SHA")


def _validate_common_identity(receipt: dict[str, Any]) -> None:
    _require(receipt.get("evidence_level") == "EV-E2", "V10 evidence level must be EV-E2")
    _require(receipt.get("protocol_id") == "NLM-REASONING-STAGE-A-CONFIRMATORY-V1", "V10 protocol id mismatch")
    _require(receipt.get("protocol_digest") == PROTOCOL_DIGEST, "V10 protocol digest mismatch")
    _require_hex64(receipt.get("code_digest"), "code_digest")
    _require_hex40(receipt.get("scientific_branch_head"), "scientific_branch_head")
    _require_hex40(receipt.get("executed_commit"), "executed_commit")
    for key in _BOUNDARY_FALSE_KEYS:
        _require(receipt.get(key) is False, f"V10 boundary {key} must remain false")


def _validate_metric_shape(metrics: dict[str, Any]) -> None:
    required = {
        "episodes",
        "route_count",
        "policy_solutions",
        "policy_flops",
        "stop_successes",
        "branch_successes",
        "stop_flops",
        "branch_flops",
        "selected_rescues",
        "selected_harms",
        "selected_both_success",
        "selected_both_failure",
        "raw_rescues",
        "provenance_closed",
    }
    _require(required.issubset(metrics), "V10 heldout metrics are incomplete")
    episodes = int(metrics["episodes"])
    route_count = int(metrics["route_count"])
    _require(episodes == 2 * HELDOUT_REPLICATES_PER_ROOT * int(FROZEN_GEOMETRY["batch_size"]), "V10 heldout episode count mismatch")
    _require(0 <= route_count <= episodes, "V10 route count is invalid")
    _require(metrics.get("provenance_closed") is True, "V10 metric provenance must close")
    _require(
        int(metrics["selected_rescues"])
        + int(metrics["selected_harms"])
        + int(metrics["selected_both_success"])
        + int(metrics["selected_both_failure"])
        == route_count,
        "V10 selected outcome partition did not close",
    )
    _require(0 <= int(metrics["raw_rescues"]) <= episodes, "V10 raw rescue count invalid")
    _require(float(metrics["policy_flops"]) > 0.0, "V10 policy FLOPs must be positive")
    _require(float(metrics["stop_flops"]) > 0.0, "V10 stop FLOPs must be positive")
    _require(float(metrics["branch_flops"]) > 0.0, "V10 branch FLOPs must be positive")


def validate_shard_receipt(receipt: dict[str, Any]) -> None:
    _require(receipt.get("schema") == SHARD_SCHEMA, "V10 shard schema mismatch")
    _validate_common_identity(receipt)
    _require(receipt.get("analysis_scope") == ANALYSIS_SCOPE, "V10 shard analysis scope mismatch")

    budget = receipt.get("train_replicates")
    canonical = receipt.get("canonical_index")
    _require(budget in {60, 120}, "V10 shard train budget must be 60 or 120")
    _require(canonical in {0, 1, 2, 3}, "V10 shard canonical index must be 0..3")
    expected_roots = build_root_schedule(int(budget), int(canonical))
    _require(receipt.get("roots") == expected_roots, "V10 shard root schedule mismatch")
    _require(receipt.get("world_model_geometry") == FROZEN_GEOMETRY, "V10 frozen geometry mismatch")
    _require(receipt.get("optimizer") == FROZEN_OPTIMIZER, "V10 frozen optimizer mismatch")
    _require(receipt.get("nearest_neighbor_contract") == FROZEN_NN, "V10 frozen 1-NN contract mismatch")
    _require(receipt.get("fit_replicates_per_root") == FIT_REPLICATES_PER_ROOT, "V10 fit replicate count mismatch")
    _require(receipt.get("heldout_replicates_per_root") == HELDOUT_REPLICATES_PER_ROOT, "V10 heldout replicate count mismatch")

    _require_hex64(receipt.get("training_batches_digest"), "training_batches_digest")
    fit_digests = receipt.get("fit_root_batch_digests")
    heldout_digests = receipt.get("heldout_root_batch_digests")
    _require(isinstance(fit_digests, dict) and set(fit_digests) == set(expected_roots["fit_roots"]), "V10 fit root digest keys mismatch")
    _require(isinstance(heldout_digests, dict) and set(heldout_digests) == set(expected_roots["heldout_roots"]), "V10 heldout root digest keys mismatch")
    for root, digest in {**fit_digests, **heldout_digests}.items():
        _require_hex64(digest, f"root digest {root}")

    loss = receipt.get("training_loss_summary")
    _require(isinstance(loss, dict) and loss.get("count") == budget, "V10 training loss count mismatch")
    _require(isinstance(receipt.get("model_init_seed"), int), "V10 model init seed must be integer")
    _require_hex64(receipt.get("canonical_model_digest"), "canonical_model_digest")
    _require_hex64(receipt.get("canonical_model_digest_after_court"), "canonical_model_digest_after_court")
    _require(
        receipt["canonical_model_digest"] == receipt["canonical_model_digest_after_court"],
        "V10 canonical model digest drifted during court",
    )

    audit = receipt.get("matched_resource_audit")
    _require(isinstance(audit, dict), "V10 matched resource audit missing")
    _require(set(audit) == set(_MATCHED_AUDIT_KEYS), "V10 matched resource audit keys mismatch")
    _require(all(audit[key] is True for key in _MATCHED_AUDIT_KEYS), "V10 matched resource audit did not close")

    ledger = receipt.get("compute_ledger")
    _require(isinstance(ledger, dict), "V10 compute ledger missing")
    stop_cost = float(ledger.get("stop_accounted_flops_per_episode", 0.0))
    branch_cost = float(ledger.get("branch_accounted_flops_per_episode", 0.0))
    _require(stop_cost > 0.0 and branch_cost >= stop_cost, "V10 compute ledger costs invalid")

    fit_outcomes = receipt.get("fit_outcome_counts")
    _require(isinstance(fit_outcomes, dict) and set(fit_outcomes) == _OUTCOME_KEYS, "V10 fit outcome partition keys mismatch")
    expected_fit = 2 * FIT_REPLICATES_PER_ROOT * int(FROZEN_GEOMETRY["batch_size"])
    _require(sum(int(fit_outcomes[key]) for key in _OUTCOME_KEYS) == expected_fit, "V10 fit outcome partition count mismatch")

    representations = receipt.get("representations")
    _require(isinstance(representations, dict) and set(representations) == set(REPRESENTATION_VIEWS), "V10 representation set mismatch")
    for view in REPRESENTATION_VIEWS:
        row = representations[view]
        _require(row.get("dimension") == REPRESENTATION_DIMENSIONS[view], f"V10 dimension mismatch for {view}")
        fit_branch = int(row.get("fit_branch_labels", -1))
        fit_stop = int(row.get("fit_stop_labels", -1))
        _require(fit_branch >= 0 and fit_stop >= 0 and fit_branch + fit_stop == expected_fit, f"V10 fit label partition mismatch for {view}")
        metrics = row.get("heldout_metrics")
        _require(isinstance(metrics, dict), f"V10 heldout metrics missing for {view}")
        _validate_metric_shape(metrics)
        expected_class = classify_representation_root(metrics)
        _require(row.get("root_classification") == expected_class, f"V10 forged root classification for {view}")

    _require_hex64(receipt.get("artifact_digest"), "artifact_digest")
    _require(receipt["artifact_digest"] == _artifact_digest(receipt), "V10 shard artifact digest mismatch")


def _require_same_identity(receipts: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    identity = (
        receipts[0]["protocol_digest"],
        receipts[0]["code_digest"],
        receipts[0]["scientific_branch_head"],
        receipts[0]["executed_commit"],
    )
    for receipt in receipts[1:]:
        other = (
            receipt["protocol_digest"],
            receipt["code_digest"],
            receipt["scientific_branch_head"],
            receipt["executed_commit"],
        )
        _require(other == identity, "V10 reducer provenance identity mismatch")
    return identity


def build_budget_receipt(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if len(shards) != 4:
        raise ValueError("V10 budget receipt requires exactly four canonical shards")
    for shard in shards:
        validate_shard_receipt(shard)
    ordered = sorted(shards, key=lambda row: int(row["canonical_index"]))
    _require([row["canonical_index"] for row in ordered] == [0, 1, 2, 3], "V10 budget canonical indices must be exactly 0..3")
    budgets = {int(row["train_replicates"]) for row in ordered}
    _require(len(budgets) == 1, "V10 budget reducer received mixed train budgets")
    budget = budgets.pop()
    protocol_digest, code_digest, branch_head, executed_commit = _require_same_identity(ordered)

    representations: dict[str, Any] = {}
    classes: dict[str, str] = {}
    for view in REPRESENTATION_VIEWS:
        root_metrics = [deepcopy(row["representations"][view]["heldout_metrics"]) for row in ordered]
        root_classes = [classify_representation_root(metrics) for metrics in root_metrics]
        budget_class, pooled = classify_representation_budget(root_metrics)
        classes[view] = budget_class
        representations[view] = {
            "root_metrics": root_metrics,
            "root_classifications": root_classes,
            "budget_classification": budget_class,
            "pooled_metrics": pooled,
        }

    receipt: dict[str, Any] = {
        "schema": BUDGET_SCHEMA,
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "scientific_branch_head": branch_head,
        "executed_commit": executed_commit,
        "train_replicates": budget,
        "canonical_indices": [0, 1, 2, 3],
        "source_shard_receipt_sha256": [receipt_sha256(row) for row in ordered],
        "representation_budget_classifications": classes,
        "representations": representations,
        "evaluation_rng_used": False,
        "evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "mechanism_successor_authorized": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    validate_budget_receipt(receipt)
    return receipt


def validate_budget_receipt(receipt: dict[str, Any]) -> None:
    _require(receipt.get("schema") == BUDGET_SCHEMA, "V10 budget schema mismatch")
    _validate_common_identity(receipt)
    budget = receipt.get("train_replicates")
    _require(budget in {60, 120}, "V10 budget train budget must be 60 or 120")
    _require(receipt.get("canonical_indices") == [0, 1, 2, 3], "V10 budget canonical indices mismatch")
    source = receipt.get("source_shard_receipt_sha256")
    _require(isinstance(source, list) and len(source) == 4, "V10 budget requires four source shard digests")
    for digest in source:
        _require_hex64(digest, "source shard receipt sha256")

    representations = receipt.get("representations")
    classes = receipt.get("representation_budget_classifications")
    _require(isinstance(representations, dict) and set(representations) == set(REPRESENTATION_VIEWS), "V10 budget representation set mismatch")
    _require(isinstance(classes, dict) and set(classes) == set(REPRESENTATION_VIEWS), "V10 budget classification set mismatch")
    for view in REPRESENTATION_VIEWS:
        row = representations[view]
        root_metrics = row.get("root_metrics")
        _require(isinstance(root_metrics, list) and len(root_metrics) == 4, f"V10 budget {view} requires four root metrics")
        for metrics in root_metrics:
            _validate_metric_shape(metrics)
        expected_root_classes = [classify_representation_root(metrics) for metrics in root_metrics]
        expected_budget_class, expected_pooled = classify_representation_budget(root_metrics)
        _require(row.get("root_classifications") == expected_root_classes, f"V10 budget forged root classes for {view}")
        _require(row.get("budget_classification") == expected_budget_class, f"V10 budget forged classification for {view}")
        _require(classes.get(view) == expected_budget_class, f"V10 budget classification map mismatch for {view}")
        _require(row.get("pooled_metrics") == expected_pooled, f"V10 budget pooled metrics mismatch for {view}")

    _require_hex64(receipt.get("artifact_digest"), "artifact_digest")
    _require(receipt["artifact_digest"] == _artifact_digest(receipt), "V10 budget artifact digest mismatch")


def build_cross_receipt(train60: dict[str, Any], train120: dict[str, Any]) -> dict[str, Any]:
    validate_budget_receipt(train60)
    validate_budget_receipt(train120)
    _require(train60["train_replicates"] == 60, "V10 cross train60 input has wrong budget")
    _require(train120["train_replicates"] == 120, "V10 cross train120 input has wrong budget")
    protocol_digest, code_digest, branch_head, executed_commit = _require_same_identity([train60, train120])

    train_budget_classifications = {
        "60": deepcopy(train60["representation_budget_classifications"]),
        "120": deepcopy(train120["representation_budget_classifications"]),
    }
    disposition = classify_cross_budget(train_budget_classifications)
    receipt: dict[str, Any] = {
        "schema": CROSS_SCHEMA,
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "scientific_branch_head": branch_head,
        "executed_commit": executed_commit,
        "train_budget_classifications": train_budget_classifications,
        "source_budget_receipt_sha256": {
            "60": receipt_sha256(train60),
            "120": receipt_sha256(train120),
        },
        **disposition,
        "evaluation_rng_used": False,
        "evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    validate_cross_receipt(receipt)
    return receipt


def validate_cross_receipt(receipt: dict[str, Any]) -> None:
    _require(receipt.get("schema") == CROSS_SCHEMA, "V10 cross schema mismatch")
    _validate_common_identity(receipt)
    source = receipt.get("source_budget_receipt_sha256")
    _require(isinstance(source, dict) and set(source) == {"60", "120"}, "V10 cross source budget keys must be strings 60/120")
    for key in ("60", "120"):
        _require_hex64(source[key], f"source budget {key} receipt sha256")

    classes = receipt.get("train_budget_classifications")
    _require(isinstance(classes, dict) and set(classes) == {"60", "120"}, "V10 cross train budget keys must be strings 60/120")
    for budget in ("60", "120"):
        _require(isinstance(classes[budget], dict) and set(classes[budget]) == set(REPRESENTATION_VIEWS), f"V10 cross representation set mismatch for budget {budget}")
    expected = classify_cross_budget(classes)
    for key in (
        "representation_cross_classifications",
        "decision",
        "successor_design_authorized",
        "authorized_representation_view",
        "authorization_scope",
        "mechanism_successor_authorized",
    ):
        _require(receipt.get(key) == expected[key], f"V10 cross forged {key}")

    _require_hex64(receipt.get("artifact_digest"), "artifact_digest")
    _require(receipt["artifact_digest"] == _artifact_digest(receipt), "V10 cross artifact digest mismatch")
