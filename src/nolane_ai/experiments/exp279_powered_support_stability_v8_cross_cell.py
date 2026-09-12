from __future__ import annotations

import math
import re
from typing import Any, Mapping

from nolane_ai.protocol.evidence import canonical_sha256

from .exp279_powered_support_stability_v8 import (
    CANONICAL_INDICES,
    FROZEN_ARM_GEOMETRY,
    FROZEN_FAMILY_ALPHA,
    FROZEN_GLOBAL_FOLDS,
    FROZEN_OPTIMIZER,
    FROZEN_PROTOCOL_DIGEST,
    FROZEN_ROUTE_THRESHOLD,
    FROZEN_V7_FLOOR,
    FROZEN_WORLD_GEOMETRY,
    PROBE_FOLDS,
    PROBE_INDICES,
    PROBE_REPLICATES,
    PROTOCOL_ID,
    ROOT_PREFIX,
    SCHEMA_BUDGET,
    classify_train_cell,
    expected_root_map,
    required_fold_episodes,
    wilson_lower_bound,
)

SCHEMA_CROSS_CELL = "NLM-EXP-279-POWERED-SUPPORT-CROSS-CELL-V8"
_DECISION_BUDGETS = (60, 120)
_COUNT_KEYS = ("branch_rescues", "branch_harms", "both_success", "both_failure")
_FALSE_BOUNDARY_KEYS = (
    "scientific_evidence_eligible",
    "evaluation_rng_stream_used",
    "evaluation_targets_used",
    "raw_examples_exported",
    "raw_model_outputs_exported",
    "selector_trained",
    "cdd_distiller_trained",
    "branch_preview_selector_trained",
    "threshold_tuned",
    "fresh_evaluation_lineage_may_be_reserved",
    "fresh_evaluation_lineage_consumed",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _artifact_digest(receipt: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in receipt.items() if key != "artifact_digest"})


def _same_float(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _require_hex(value: Any, *, bits: int, label: str) -> str:
    text = str(value)
    pattern = _HEX40 if bits == 160 else _HEX64
    if pattern.fullmatch(text) is None:
        raise ValueError(f"V8 {label} must be exact lowercase {bits // 4}-hex")
    return text


def _validate_fold_record(
    fold: Mapping[str, Any],
    *,
    canonical_index: int,
    probe_index: int,
    fold_index: int,
    canonical_root: str,
    probe_root: str,
) -> dict[str, int]:
    if int(fold.get("canonical_index", -1)) != canonical_index:
        raise ValueError("V8 budget fold canonical index mismatch")
    if int(fold.get("probe_index", -1)) != probe_index:
        raise ValueError("V8 budget fold probe index mismatch")
    if int(fold.get("fold", -1)) != fold_index:
        raise ValueError("V8 budget fold index mismatch")
    if fold.get("canonical_root") != canonical_root or fold.get("probe_root") != probe_root:
        raise ValueError("V8 budget fold root mismatch")

    episodes = int(fold.get("episodes", -1))
    if episodes != required_fold_episodes():
        raise ValueError("V8 budget frozen fold episode count mismatch")
    counts = {key: int(fold.get(key, -1)) for key in _COUNT_KEYS}
    if any(value < 0 for value in counts.values()) or sum(counts.values()) != episodes:
        raise ValueError("V8 budget fold outcome partition did not close")
    rescues = counts["branch_rescues"]
    nonrescue = episodes - rescues
    if int(fold.get("nonrescue_count", -1)) != nonrescue:
        raise ValueError("V8 budget fold nonrescue mismatch")
    if not _same_float(fold.get("rescue_prevalence"), rescues / episodes):
        raise ValueError("V8 budget fold prevalence mismatch")
    if fold.get("fold_rescue_supported") is not (rescues > 0):
        raise ValueError("V8 budget fold rescue support mismatch")
    if fold.get("fold_nonrescue_supported") is not (nonrescue > 0):
        raise ValueError("V8 budget fold nonrescue support mismatch")
    return {"episodes": episodes, **counts}


def _validate_budget_receipt(receipt: Mapping[str, Any], *, expected_train: int) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA_BUDGET:
        raise ValueError("V8 budget schema mismatch")
    if receipt.get("evidence_level") != "EV-E2" or receipt.get("decision") != "UNVERIFIED":
        raise ValueError("V8 budget evidence state mismatch")
    if receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("protocol_digest") != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("V8 budget frozen protocol identity mismatch")
    if receipt.get("root_prefix") != ROOT_PREFIX:
        raise ValueError("V8 budget root prefix mismatch")
    if int(receipt.get("train_replicates", -1)) != expected_train:
        raise ValueError("V8 budget train identity mismatch")

    branch_head = _require_hex(receipt.get("scientific_branch_head"), bits=160, label="scientific branch head")
    executed = _require_hex(receipt.get("executed_commit"), bits=160, label="executed commit")
    code_digest = _require_hex(receipt.get("code_digest"), bits=256, label="code digest")

    if receipt.get("world_geometry") != FROZEN_WORLD_GEOMETRY:
        raise ValueError("V8 budget frozen world geometry mismatch")
    if receipt.get("arm_geometry") != FROZEN_ARM_GEOMETRY:
        raise ValueError("V8 budget frozen arm geometry mismatch")
    if receipt.get("optimizer") != FROZEN_OPTIMIZER:
        raise ValueError("V8 budget frozen optimizer mismatch")
    if receipt.get("route_config") != {
        "canonical_threshold": FROZEN_ROUTE_THRESHOLD,
        "threshold_tuned": False,
    }:
        raise ValueError("V8 budget frozen route threshold mismatch")
    if receipt.get("probe_config") != {
        "replicates": PROBE_REPLICATES,
        "folds": PROBE_FOLDS,
        "probe_roots": len(PROBE_INDICES),
    }:
        raise ValueError("V8 budget frozen probe configuration mismatch")
    if receipt.get("power_contract") != {
        "v7_prevalence_floor": FROZEN_V7_FLOOR,
        "global_folds": FROZEN_GLOBAL_FOLDS,
        "family_alpha": FROZEN_FAMILY_ALPHA,
        "required_fold_episodes": required_fold_episodes(),
    }:
        raise ValueError("V8 budget frozen power contract mismatch")
    if receipt.get("training_rng_stream") != "augmentation" or receipt.get("probe_rng_stream") != "augmentation":
        raise ValueError("V8 budget RNG stream mismatch")
    for key in _FALSE_BOUNDARY_KEYS:
        if receipt.get(key) is not False:
            raise ValueError(f"V8 budget boundary mismatch for {key}")

    expected_roots = expected_root_map(expected_train)
    root_map = receipt.get("root_map")
    if root_map != expected_roots:
        raise ValueError("V8 budget root map mismatch")
    if receipt.get("root_map_digest") != canonical_sha256(expected_roots):
        raise ValueError("V8 budget root map digest mismatch")

    source_digests = receipt.get("source_shard_artifact_digests")
    if not isinstance(source_digests, list) or len(source_digests) != len(CANONICAL_INDICES):
        raise ValueError("V8 budget source shard digest cardinality mismatch")
    source_digests = [_require_hex(value, bits=256, label="source shard artifact digest") for value in source_digests]

    canonical_records = receipt.get("canonical_records")
    pair_records = receipt.get("pair_records")
    fold_records = receipt.get("fold_records")
    if not isinstance(canonical_records, list) or len(canonical_records) != 4:
        raise ValueError("V8 budget canonical record cardinality mismatch")
    if not isinstance(pair_records, list) or len(pair_records) != 16:
        raise ValueError("V8 budget pair record cardinality mismatch")
    if not isinstance(fold_records, list) or len(fold_records) != 48:
        raise ValueError("V8 budget fold record cardinality mismatch")

    expected_pair_episodes = required_fold_episodes() * PROBE_FOLDS
    rebuilt_canonical: list[dict[str, Any]] = []
    pair_cursor = 0
    fold_cursor = 0
    for canonical_index in CANONICAL_INDICES:
        canonical_root = expected_roots["canonical_roots"][canonical_index]
        canonical_counts = {"episodes": 0, **{key: 0 for key in _COUNT_KEYS}}
        all_probe_support = True
        all_fold_support = True
        for probe_index in PROBE_INDICES:
            probe_root = expected_roots["probe_roots"][canonical_index * len(PROBE_INDICES) + probe_index]
            pair = pair_records[pair_cursor]
            pair_cursor += 1
            if int(pair.get("canonical_index", -1)) != canonical_index or int(pair.get("probe_index", -1)) != probe_index:
                raise ValueError("V8 budget pair index mismatch")
            if pair.get("canonical_root") != canonical_root or pair.get("probe_root") != probe_root:
                raise ValueError("V8 budget pair root mismatch")

            pair_counts = {"episodes": 0, **{key: 0 for key in _COUNT_KEYS}}
            pair_fold_support = True
            for fold_index in range(PROBE_FOLDS):
                fold = fold_records[fold_cursor]
                fold_cursor += 1
                normalized = _validate_fold_record(
                    fold,
                    canonical_index=canonical_index,
                    probe_index=probe_index,
                    fold_index=fold_index,
                    canonical_root=canonical_root,
                    probe_root=probe_root,
                )
                for key, value in normalized.items():
                    pair_counts[key] += int(value)
                pair_fold_support = pair_fold_support and bool(
                    fold["fold_rescue_supported"] and fold["fold_nonrescue_supported"]
                )

            if pair_counts["episodes"] != expected_pair_episodes:
                raise ValueError("V8 budget pair episode count mismatch")
            for key in ("episodes", *_COUNT_KEYS):
                if int(pair.get(key, -1)) != pair_counts[key]:
                    raise ValueError(f"V8 budget pair sufficient statistic mismatch: {key}")
            rescues = pair_counts["branch_rescues"]
            nonrescue = pair_counts["episodes"] - rescues
            expected_pair = {
                "nonrescue_count": nonrescue,
                "probe_rescue_supported": rescues > 0,
                "probe_nonrescue_supported": nonrescue > 0,
                "fold_support_closed": pair_fold_support,
            }
            for key, expected in expected_pair.items():
                if pair.get(key) != expected:
                    raise ValueError(f"V8 budget pair support mismatch: {key}")
            if not _same_float(pair.get("rescue_prevalence"), rescues / pair_counts["episodes"]):
                raise ValueError("V8 budget pair prevalence mismatch")
            if not _same_float(pair.get("wilson_lower_bound"), wilson_lower_bound(rescues, pair_counts["episodes"])):
                raise ValueError("V8 budget pair Wilson mismatch")
            all_probe_support = all_probe_support and rescues > 0 and nonrescue > 0
            all_fold_support = all_fold_support and pair_fold_support
            for key, value in pair_counts.items():
                canonical_counts[key] += int(value)

        canonical = canonical_records[canonical_index]
        if int(canonical.get("canonical_index", -1)) != canonical_index or canonical.get("canonical_root") != canonical_root:
            raise ValueError("V8 budget canonical identity mismatch")
        _require_hex(canonical.get("final_canonical_digest"), bits=256, label="canonical state digest")
        source_digest = _require_hex(
            canonical.get("source_shard_artifact_digest"),
            bits=256,
            label="canonical source shard artifact digest",
        )
        if source_digest != source_digests[canonical_index]:
            raise ValueError("V8 budget canonical/source shard artifact digest mismatch")

        rescues = canonical_counts["branch_rescues"]
        nonrescue = canonical_counts["episodes"] - rescues
        expected_fields = {
            "canonical_total_episodes": canonical_counts["episodes"],
            "canonical_total_rescues": rescues,
            "canonical_total_harms": canonical_counts["branch_harms"],
            "canonical_total_both_success": canonical_counts["both_success"],
            "canonical_total_both_failure": canonical_counts["both_failure"],
            "canonical_total_nonrescue": nonrescue,
            "canonical_has_any_rescue": rescues > 0,
            "canonical_all_probe_roots_supported": all_probe_support,
            "canonical_all_folds_supported": all_fold_support,
        }
        for key, expected in expected_fields.items():
            if canonical.get(key) != expected:
                raise ValueError(f"V8 budget canonical sufficient statistic mismatch: {key}")
        prevalence = rescues / canonical_counts["episodes"]
        lower = wilson_lower_bound(rescues, canonical_counts["episodes"])
        if not _same_float(canonical.get("pooled_rescue_prevalence"), prevalence):
            raise ValueError("V8 budget canonical prevalence mismatch")
        if not _same_float(canonical.get("pooled_wilson_lower_bound"), lower):
            raise ValueError("V8 budget canonical Wilson mismatch")
        rebuilt_canonical.append(
            {
                **expected_fields,
                "pooled_rescue_prevalence": prevalence,
                "pooled_wilson_lower_bound": lower,
            }
        )

    classification = classify_train_cell(rebuilt_canonical)
    if receipt.get("train_cell_classification") != classification:
        raise ValueError("V8 budget train-cell classification mismatch")
    prevalence_floor = min(float(item["pooled_wilson_lower_bound"]) for item in rebuilt_canonical)
    if not _same_float(receipt.get("canonical_prevalence_floor"), prevalence_floor):
        raise ValueError("V8 budget canonical prevalence floor mismatch")
    expected_total = expected_pair_episodes * len(CANONICAL_INDICES) * len(PROBE_INDICES)
    if int(receipt.get("total_episodes", -1)) != expected_total:
        raise ValueError("V8 budget total episode count mismatch")
    if receipt.get("artifact_digest") != _artifact_digest(receipt):
        raise ValueError("V8 budget artifact digest mismatch")

    return {
        "classification": classification,
        "canonical_prevalence_floor": prevalence_floor,
        "scientific_branch_head": branch_head,
        "executed_commit": executed,
        "code_digest": code_digest,
        "artifact_digest": str(receipt["artifact_digest"]),
        "roots": set(expected_roots["canonical_roots"] + expected_roots["probe_roots"]),
    }


def classify_exp279_powered_support_stability_v8_cross_cell(
    cells: Mapping[int, Mapping[str, Any]],
    *,
    expected_scientific_branch_head: str,
    expected_executed_commit: str,
    expected_code_digest: str,
) -> dict[str, Any]:
    expected_branch_head = _require_hex(
        expected_scientific_branch_head,
        bits=160,
        label="expected scientific branch head",
    )
    expected_executed = _require_hex(
        expected_executed_commit,
        bits=160,
        label="expected executed commit",
    )
    expected_code = _require_hex(
        expected_code_digest,
        bits=256,
        label="expected code digest",
    )
    if set(cells.keys()) != set(_DECISION_BUDGETS):
        raise ValueError("V8 cross-cell decision budgets must be exactly train60 and train120")
    validated = {
        train: _validate_budget_receipt(cells[train], expected_train=train)
        for train in _DECISION_BUDGETS
    }

    expected_provenance = {
        "scientific_branch_head": expected_branch_head,
        "executed_commit": expected_executed,
        "code_digest": expected_code,
    }
    for train in _DECISION_BUDGETS:
        for field, expected in expected_provenance.items():
            if validated[train][field] != expected:
                raise ValueError(f"V8 cross-cell provenance trust-anchor mismatch: {field}")
    for field in expected_provenance:
        if validated[60][field] != validated[120][field]:
            raise ValueError(f"V8 cross-cell provenance mismatch: {field}")
    if validated[60]["roots"] & validated[120]["roots"]:
        raise ValueError("V8 cross-cell train60/train120 root sets must be disjoint")

    classifications = {train: str(validated[train]["classification"]) for train in _DECISION_BUDGETS}
    if "MODEL_ROOT_SUPPORT_COLLAPSE" in classifications.values():
        decision = "MODEL_ROOT_SUPPORT_COLLAPSE"
        successor = False
        scope = "CANONICAL_BRANCH_SEMANTICS_RESEARCH_ONLY"
    elif "PROBE_SUPPORT_INTERMITTENT" in classifications.values():
        decision = "PROBE_SUPPORT_INTERMITTENT"
        successor = False
        scope = "CANONICAL_BRANCH_SEMANTICS_RESEARCH_ONLY"
    else:
        decision = "SUPPORT_RECURRENT"
        successor = True
        scope = "DESIGN_NEW_MECHANISM_COURT_ONLY"

    receipt: dict[str, Any] = {
        "schema": SCHEMA_CROSS_CELL,
        "evidence_level": "EV-E2",
        "decision": decision,
        "scientific_evidence_eligible": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": FROZEN_PROTOCOL_DIGEST,
        "code_digest": validated[60]["code_digest"],
        "scientific_branch_head": validated[60]["scientific_branch_head"],
        "executed_commit": validated[60]["executed_commit"],
        "train_cell_classifications": {str(train): classifications[train] for train in _DECISION_BUDGETS},
        "canonical_prevalence_floors": {
            str(train): float(validated[train]["canonical_prevalence_floor"])
            for train in _DECISION_BUDGETS
        },
        "input_artifact_digests": {
            str(train): validated[train]["artifact_digest"]
            for train in _DECISION_BUDGETS
        },
        "successor_design_authorized": successor,
        "mechanism_successor_authorized": False,
        "authorization_scope": scope,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    return receipt
