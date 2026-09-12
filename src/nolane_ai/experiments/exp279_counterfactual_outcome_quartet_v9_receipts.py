from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from nolane_ai.protocol.evidence import canonical_sha256

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


def _artifact_digest(receipt: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in receipt.items() if key != "artifact_digest"})


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
    artifact = receipt.get("artifact_digest")
    if artifact is not None:
        _append(errors, artifact == _artifact_digest(receipt), "V9 shard artifact digest mismatch")
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

    _append(errors, canonical_indices == list(CANONICAL_INDICES), "V9 budget canonical roots must be ordered exactly 0..3")

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
            _append(errors, receipt.get("root_classifications") == recomputed.get("root_classifications"), "V9 budget root classifications mismatch")
            _append(errors, receipt.get("budget_summary") == {key: value for key, value in recomputed.items() if key not in {"classification", "root_classifications"}}, "V9 budget summary mismatch")
    artifact = receipt.get("artifact_digest")
    if artifact is not None:
        _append(errors, artifact == _artifact_digest(receipt), "V9 budget artifact digest mismatch")
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
        _append(errors, receipt.get("train_budget_classifications") == recomputed.get("train_budget_classifications"), "V9 cross budget classifications mismatch")
        _append(errors, receipt.get("recomputed_budgets") == recomputed.get("recomputed_budgets"), "V9 cross recomputed budget summary mismatch")
    artifact = receipt.get("artifact_digest")
    if artifact is not None:
        _append(errors, artifact == _artifact_digest(receipt), "V9 cross artifact digest mismatch")
    return errors


def _raise_validation(prefix: str, errors: list[str]) -> None:
    if errors:
        raise ValueError(prefix + ": " + "; ".join(errors))


def build_budget_receipt(shards: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(shards) != len(CANONICAL_INDICES):
        raise ValueError("V9 budget reducer requires exactly four shard receipts")
    checked = [dict(shard) for shard in shards]
    for index, shard in enumerate(checked):
        _raise_validation(f"V9 source shard {index} invalid", validate_shard_receipt(shard))
    try:
        checked.sort(key=lambda shard: int(shard["canonical_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V9 source shard canonical identity is invalid") from exc
    if [int(shard["canonical_index"]) for shard in checked] != list(CANONICAL_INDICES):
        raise ValueError("V9 budget reducer requires unique canonical roots 0..3")

    first = checked[0]
    train = int(first["train_replicates"])
    if train not in TRAIN_BUDGETS:
        raise ValueError("V9 budget reducer source train budget is not frozen")
    for shard in checked:
        if int(shard["train_replicates"]) != train:
            raise ValueError("V9 budget reducer cannot mix train budgets")
        for key in ("protocol_digest", "code_digest", "scientific_branch_head", "executed_commit"):
            if shard.get(key) != first.get(key):
                raise ValueError(f"V9 budget reducer source {key} mismatch")

    metrics = [dict(shard["root_metrics"]) for shard in checked]
    reduced = classify_quartet_budget(metrics)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_BUDGET,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_digest": first["protocol_digest"],
        "code_digest": first["code_digest"],
        "scientific_branch_head": first["scientific_branch_head"],
        "executed_commit": first["executed_commit"],
        "train_replicates": train,
        "source_shard_digests": [receipt_sha256(shard) for shard in checked],
        "root_receipts": checked,
        "root_metrics": metrics,
        "root_classifications": reduced["root_classifications"],
        "budget_summary": {key: value for key, value in reduced.items() if key not in {"classification", "root_classifications"}},
        "classification": reduced["classification"],
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    _raise_validation("V9 built budget receipt invalid", validate_budget_receipt(receipt))
    return receipt


def build_cross_receipt(train60: Mapping[str, Any], train120: Mapping[str, Any]) -> dict[str, Any]:
    left = dict(train60)
    right = dict(train120)
    _raise_validation("V9 train60 budget invalid", validate_budget_receipt(left))
    _raise_validation("V9 train120 budget invalid", validate_budget_receipt(right))
    if int(left.get("train_replicates", -1)) != 60 or int(right.get("train_replicates", -1)) != 120:
        raise ValueError("V9 cross reducer requires train60 then train120 receipts")
    for key in ("protocol_digest", "code_digest", "scientific_branch_head", "executed_commit"):
        if left.get(key) != right.get(key):
            raise ValueError(f"V9 cross reducer budget {key} mismatch")

    reduced = classify_quartet_cross_budget(left, right)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_CROSS,
        "evidence_level": "EV-E2",
        "protocol_digest": left["protocol_digest"],
        "code_digest": left["code_digest"],
        "scientific_branch_head": left["scientific_branch_head"],
        "executed_commit": left["executed_commit"],
        "source_budget_digests": {"60": receipt_sha256(left), "120": receipt_sha256(right)},
        "budget_receipts": {"60": left, "120": right},
        **reduced,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    _raise_validation("V9 built cross receipt invalid", validate_cross_receipt(receipt))
    return receipt


__all__ = [
    "build_budget_receipt",
    "build_cross_receipt",
    "canonical_receipt_bytes",
    "receipt_sha256",
    "validate_shard_receipt",
    "validate_budget_receipt",
    "validate_cross_receipt",
]
