from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from .exp279_counterfactual_outcome_quartet_v9 import (
    CANONICAL_INDICES,
    DECISION_REPLICATES,
    FIT_REPLICATES,
    FROZEN_PROTOCOL_DIGEST,
    PRIMARY_FAMILY,
    SCHEMA_BUDGET,
    SCHEMA_CROSS,
    SCHEMA_SHARD,
    STUDENT_GEOMETRY,
    STUDENT_OPTIMIZER,
    TRAIN_BUDGETS,
    canonical_root,
    classify_quartet_budget,
    classify_quartet_cross_budget,
    decision_root,
    fit_root,
)

PROTOCOL_ID = "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FALSE_BOUNDARY_KEYS = (
    "scientific_evidence_eligible",
    "fresh_evaluation_lineage_may_be_reserved",
    "fresh_evaluation_lineage_consumed",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
)


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()


def _append(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _valid_commit(value: Any) -> bool:
    return isinstance(value, str) and _HEX40.fullmatch(value) is not None


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _closed_boundaries(receipt: Mapping[str, Any], errors: list[str], *, prefix: str) -> None:
    for key in _FALSE_BOUNDARY_KEYS:
        _append(errors, receipt.get(key) is False, f"{prefix} boundary {key} must be false")


def _identity(receipt: Mapping[str, Any], errors: list[str], *, prefix: str) -> None:
    _append(errors, receipt.get("protocol_digest") == FROZEN_PROTOCOL_DIGEST, f"{prefix} protocol digest mismatch")
    _append(errors, _valid_digest(receipt.get("code_digest")), f"{prefix} code digest must be lowercase 64-hex")
    _append(errors, _valid_commit(receipt.get("scientific_branch_head")), f"{prefix} scientific branch head must be lowercase 40-hex")
    _append(errors, _valid_commit(receipt.get("executed_commit")), f"{prefix} executed commit must be lowercase 40-hex")


def validate_shard_receipt(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    _append(errors, receipt.get("schema") == SCHEMA_SHARD, "V9 shard schema mismatch")
    _append(errors, receipt.get("evidence_level") == "EV-E2", "V9 shard evidence level mismatch")
    _append(errors, receipt.get("decision") == "UNVERIFIED", "V9 shard decision must remain UNVERIFIED")
    _append(errors, receipt.get("protocol_id") == PROTOCOL_ID, "V9 shard protocol id mismatch")
    _identity(receipt, errors, prefix="V9 shard")
    _closed_boundaries(receipt, errors, prefix="V9 shard")

    try:
        train = int(receipt.get("train_replicates"))
    except (TypeError, ValueError):
        train = -1
    try:
        canonical = int(receipt.get("canonical_index"))
    except (TypeError, ValueError):
        canonical = -1
    _append(errors, train in TRAIN_BUDGETS, "V9 shard train budget mismatch")
    _append(errors, canonical in CANONICAL_INDICES, "V9 shard canonical index mismatch")
    if train in TRAIN_BUDGETS and canonical in CANONICAL_INDICES:
        _append(errors, receipt.get("canonical_root") == canonical_root(train, canonical), "V9 shard canonical root mismatch")
        _append(
            errors,
            receipt.get("fit_roots") == [fit_root(train, canonical, 0), fit_root(train, canonical, 1)],
            "V9 shard fit roots mismatch",
        )
        _append(
            errors,
            receipt.get("decision_roots") == [decision_root(train, canonical, 0), decision_root(train, canonical, 1)],
            "V9 shard decision roots mismatch",
        )

    _append(
        errors,
        receipt.get("rng_streams") == {
            "canonical_training": "augmentation",
            "fit": "augmentation",
            "decision": "augmentation",
        },
        "V9 shard must use augmentation RNG only",
    )
    _append(errors, receipt.get("student_family") == PRIMARY_FAMILY, "V9 shard student family mismatch")
    _append(errors, receipt.get("student_geometry") == STUDENT_GEOMETRY, "V9 shard student geometry mismatch")
    _append(errors, receipt.get("student_optimizer") == STUDENT_OPTIMIZER, "V9 shard student optimizer mismatch")
    _append(errors, receipt.get("fit_replicates_per_root") == FIT_REPLICATES, "V9 shard fit replicate geometry mismatch")
    _append(errors, receipt.get("decision_replicates_per_root") == DECISION_REPLICATES, "V9 shard decision replicate geometry mismatch")
    metrics = receipt.get("root_metrics")
    _append(errors, isinstance(metrics, Mapping), "V9 shard root metrics missing")
    if isinstance(metrics, Mapping):
        _append(errors, metrics.get("evidence_boundary_closed") is True, "V9 shard root evidence boundary must be closed")
    return errors


def validate_budget_receipt(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    _append(errors, receipt.get("schema") == SCHEMA_BUDGET, "V9 budget schema mismatch")
    _append(errors, receipt.get("evidence_level") == "EV-E2", "V9 budget evidence level mismatch")
    _append(errors, receipt.get("decision") == "UNVERIFIED", "V9 budget decision must remain UNVERIFIED")
    _identity(receipt, errors, prefix="V9 budget")
    _closed_boundaries(receipt, errors, prefix="V9 budget")

    try:
        train = int(receipt.get("train_replicates"))
    except (TypeError, ValueError):
        train = -1
    _append(errors, train in TRAIN_BUDGETS, "V9 budget train budget mismatch")

    shards = receipt.get("root_receipts")
    if not isinstance(shards, list) or len(shards) != len(CANONICAL_INDICES):
        errors.append("V9 budget requires exactly four root receipts")
        return errors

    canonical_indices: list[int] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            errors.append(f"V9 budget shard {index} is not a mapping")
            continue
        errors.extend(f"root[{index}]: {error}" for error in validate_shard_receipt(shard))
        try:
            canonical_indices.append(int(shard.get("canonical_index")))
        except (TypeError, ValueError):
            canonical_indices.append(-1)
        _append(errors, shard.get("train_replicates") == train, f"V9 budget shard {index} train mismatch")
        for key in ("protocol_digest", "code_digest", "scientific_branch_head", "executed_commit"):
            _append(errors, shard.get(key) == receipt.get(key), f"V9 budget shard {index} {key} mismatch")

    _append(errors, sorted(canonical_indices) == list(CANONICAL_INDICES), "V9 budget canonical roots must be exactly 0..3")

    expected_digests = [receipt_sha256(shard) for shard in shards if isinstance(shard, Mapping)]
    _append(errors, receipt.get("source_shard_digests") == expected_digests, "V9 budget source shard digests mismatch")

    root_metrics = receipt.get("root_metrics")
    expected_metrics = [shard.get("root_metrics") for shard in shards if isinstance(shard, Mapping)]
    _append(errors, root_metrics == expected_metrics, "V9 budget root metrics must match source shards")
    if isinstance(root_metrics, list) and len(root_metrics) == len(CANONICAL_INDICES):
        try:
            recomputed = classify_quartet_budget(root_metrics)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"V9 budget metrics could not be recomputed: {exc}")
        else:
            _append(errors, receipt.get("classification") == recomputed.get("classification"), "V9 budget classification mismatch")
    return errors


def validate_cross_receipt(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    _append(errors, receipt.get("schema") == SCHEMA_CROSS, "V9 cross schema mismatch")
    _append(errors, receipt.get("evidence_level") == "EV-E2", "V9 cross evidence level mismatch")
    _identity(receipt, errors, prefix="V9 cross")
    _closed_boundaries(receipt, errors, prefix="V9 cross")

    budgets = receipt.get("budget_receipts")
    if not isinstance(budgets, Mapping) or set(budgets) != {"60", "120"}:
        errors.append("V9 cross requires exactly train60 and train120 budget receipts")
        return errors
    train60 = budgets["60"]
    train120 = budgets["120"]
    if not isinstance(train60, Mapping) or not isinstance(train120, Mapping):
        errors.append("V9 cross budget receipts must be mappings")
        return errors

    errors.extend(f"train60: {error}" for error in validate_budget_receipt(train60))
    errors.extend(f"train120: {error}" for error in validate_budget_receipt(train120))
    _append(errors, train60.get("train_replicates") == 60, "V9 cross train60 receipt identity mismatch")
    _append(errors, train120.get("train_replicates") == 120, "V9 cross train120 receipt identity mismatch")
    for label, budget in (("train60", train60), ("train120", train120)):
        for key in ("protocol_digest", "code_digest", "scientific_branch_head", "executed_commit"):
            _append(errors, budget.get(key) == receipt.get(key), f"V9 cross {label} {key} mismatch")

    expected_budget_digests = {"60": receipt_sha256(train60), "120": receipt_sha256(train120)}
    _append(errors, receipt.get("source_budget_digests") == expected_budget_digests, "V9 cross source budget digests mismatch")

    try:
        recomputed = classify_quartet_cross_budget(dict(train60), dict(train120))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"V9 cross decision could not be recomputed: {exc}")
    else:
        for key in (
            "decision",
            "authorization_scope",
            "successor_design_authorized",
            "mechanism_successor_authorized",
            "scientific_evidence_eligible",
            "fresh_evaluation_lineage_may_be_reserved",
            "fresh_evaluation_lineage_consumed",
            "confirmatory_data_consumed",
            "challenge_materialized",
            "promotion_claimed",
        ):
            _append(errors, receipt.get(key) == recomputed.get(key), f"V9 cross {key} mismatch")
    return errors


__all__ = [
    "canonical_receipt_bytes",
    "receipt_sha256",
    "validate_shard_receipt",
    "validate_budget_receipt",
    "validate_cross_receipt",
]
