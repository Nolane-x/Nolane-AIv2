from __future__ import annotations

import math
from typing import Any, Mapping

from nolane_ai.protocol.evidence import canonical_sha256

from .exp279_multiroot_support_stability_v7 import (
    CANONICAL_INDICES,
    FROZEN_ARM_GEOMETRY,
    FROZEN_OPTIMIZER,
    FROZEN_PROTOCOL_DIGEST,
    FROZEN_ROUTE_THRESHOLD,
    FROZEN_WORLD_GEOMETRY,
    PROBE_FOLDS,
    PROBE_INDICES,
    PROBE_REPLICATES,
    PROTOCOL_ID,
    ROOT_PREFIX,
    SCHEMA_BUDGET,
    classify_budget_support,
    expected_root_map,
    wilson_lower_bound,
)

SCHEMA_CROSS_CELL = "NLM-EXP-279-MULTIROOT-SUPPORT-CROSS-CELL-V7"
_DECISION_BUDGETS = (60, 120)
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
_COUNT_KEYS = ("branch_rescues", "branch_harms", "both_success", "both_failure")


def _artifact_digest(receipt: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in receipt.items() if key != "artifact_digest"})


def _same_float(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def prospective_probe_episodes(prevalence_floor: float) -> int | None:
    p = float(prevalence_floor)
    if not math.isfinite(p):
        raise ValueError("V7 prevalence floor must be finite")
    if p <= 0.0:
        return None
    if p >= 1.0:
        return 1
    return math.ceil(math.log(1.0 - 0.99) / math.log(1.0 - p))


def _validate_fold_record(
    fold: Mapping[str, Any],
    *,
    train: int,
    canonical_index: int,
    probe_index: int,
    fold_index: int,
    canonical_root: str,
    probe_root: str,
) -> dict[str, int]:
    if int(fold.get("canonical_index", -1)) != canonical_index:
        raise ValueError("V7 budget fold canonical index mismatch")
    if int(fold.get("probe_index", -1)) != probe_index:
        raise ValueError("V7 budget fold probe index mismatch")
    if int(fold.get("fold", -1)) != fold_index:
        raise ValueError("V7 budget fold index mismatch")
    if fold.get("canonical_root") != canonical_root or fold.get("probe_root") != probe_root:
        raise ValueError("V7 budget fold root mismatch")
    expected_episodes = PROBE_REPLICATES * FROZEN_WORLD_GEOMETRY["batch_size"] // PROBE_FOLDS
    episodes = int(fold.get("episodes", -1))
    counts = {key: int(fold.get(key, -1)) for key in _COUNT_KEYS}
    if episodes != expected_episodes or any(value < 0 for value in counts.values()):
        raise ValueError("V7 budget fold episode/count mismatch")
    if sum(counts.values()) != episodes:
        raise ValueError("V7 budget fold outcome partition did not close")
    rescues = counts["branch_rescues"]
    nonrescue = episodes - rescues
    if int(fold.get("nonrescue_count", -1)) != nonrescue:
        raise ValueError("V7 budget fold nonrescue mismatch")
    if not _same_float(fold.get("rescue_prevalence"), rescues / episodes):
        raise ValueError("V7 budget fold prevalence mismatch")
    if fold.get("fold_rescue_supported") is not (rescues > 0):
        raise ValueError("V7 budget fold rescue support mismatch")
    if fold.get("fold_nonrescue_supported") is not (nonrescue > 0):
        raise ValueError("V7 budget fold nonrescue support mismatch")
    return {"episodes": episodes, **counts}


def _validate_budget_receipt(receipt: Mapping[str, Any], *, expected_train: int) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA_BUDGET:
        raise ValueError("V7 budget schema mismatch")
    if receipt.get("evidence_level") != "EV-E2" or receipt.get("decision") != "UNVERIFIED":
        raise ValueError("V7 budget evidence state mismatch")
    if receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("protocol_digest") != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("V7 budget frozen protocol identity mismatch")
    if receipt.get("root_prefix") != ROOT_PREFIX:
        raise ValueError("V7 budget root prefix mismatch")
    if int(receipt.get("train_replicates", -1)) != expected_train:
        raise ValueError("V7 budget train identity mismatch")
    if receipt.get("world_geometry") != FROZEN_WORLD_GEOMETRY:
        raise ValueError("V7 budget frozen world geometry mismatch")
    if receipt.get("arm_geometry") != FROZEN_ARM_GEOMETRY:
        raise ValueError("V7 budget frozen arm geometry mismatch")
    if receipt.get("optimizer") != FROZEN_OPTIMIZER:
        raise ValueError("V7 budget frozen optimizer mismatch")
    route = receipt.get("route_config")
    if route != {"canonical_threshold": FROZEN_ROUTE_THRESHOLD, "threshold_tuned": False}:
        raise ValueError("V7 budget frozen route threshold mismatch")
    if receipt.get("probe_config") != {
        "replicates": PROBE_REPLICATES,
        "folds": PROBE_FOLDS,
        "probe_roots_per_canonical": len(PROBE_INDICES),
    }:
        raise ValueError("V7 budget frozen probe configuration mismatch")
    if receipt.get("training_rng_stream") != "augmentation" or receipt.get("probe_rng_stream") != "augmentation":
        raise ValueError("V7 budget RNG stream mismatch")
    for key in _FALSE_BOUNDARY_KEYS:
        if receipt.get(key) is not False:
            raise ValueError(f"V7 budget boundary mismatch for {key}")

    root_map = receipt.get("root_map")
    expected_roots = expected_root_map(expected_train)
    if root_map != expected_roots:
        raise ValueError("V7 budget root map mismatch")
    if receipt.get("root_map_digest") != canonical_sha256(expected_roots):
        raise ValueError("V7 budget root map digest mismatch")

    canonical_records = receipt.get("canonical_records")
    pair_records = receipt.get("pair_records")
    fold_records = receipt.get("fold_records")
    if not isinstance(canonical_records, list) or len(canonical_records) != 4:
        raise ValueError("V7 budget canonical record cardinality mismatch")
    if not isinstance(pair_records, list) or len(pair_records) != 16:
        raise ValueError("V7 budget pair record cardinality mismatch")
    if not isinstance(fold_records, list) or len(fold_records) != 48:
        raise ValueError("V7 budget fold record cardinality mismatch")

    rebuilt_canonical: list[dict[str, Any]] = []
    pair_cursor = 0
    fold_cursor = 0
    expected_pair_episodes = PROBE_REPLICATES * FROZEN_WORLD_GEOMETRY["batch_size"]
    for canonical_index in CANONICAL_INDICES:
        canonical_root = expected_roots["canonical_roots"][canonical_index]
        canonical_counts = {"episodes": 0, **{key: 0 for key in _COUNT_KEYS}}
        all_probe_support = True
        all_fold_support = True
        for probe_index in PROBE_INDICES:
            probe_root = expected_roots["probe_roots"][canonical_index * 4 + probe_index]
            pair = pair_records[pair_cursor]
            pair_cursor += 1
            if int(pair.get("canonical_index", -1)) != canonical_index or int(pair.get("probe_index", -1)) != probe_index:
                raise ValueError("V7 budget pair index mismatch")
            if pair.get("canonical_root") != canonical_root or pair.get("probe_root") != probe_root:
                raise ValueError("V7 budget pair root mismatch")

            pair_counts = {"episodes": 0, **{key: 0 for key in _COUNT_KEYS}}
            pair_fold_support = True
            for fold_index in range(PROBE_FOLDS):
                fold = fold_records[fold_cursor]
                fold_cursor += 1
                normalized = _validate_fold_record(
                    fold,
                    train=expected_train,
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
                raise ValueError("V7 budget pair episode count mismatch")
            for key in ("episodes", *_COUNT_KEYS):
                if int(pair.get(key, -1)) != pair_counts[key]:
                    raise ValueError(f"V7 budget pair sufficient statistic mismatch: {key}")
            pair_rescues = pair_counts["branch_rescues"]
            pair_nonrescue = pair_counts["episodes"] - pair_rescues
            if int(pair.get("nonrescue_count", -1)) != pair_nonrescue:
                raise ValueError("V7 budget pair nonrescue mismatch")
            if not _same_float(pair.get("rescue_prevalence"), pair_rescues / pair_counts["episodes"]):
                raise ValueError("V7 budget pair prevalence mismatch")
            if pair.get("probe_rescue_supported") is not (pair_rescues > 0):
                raise ValueError("V7 budget pair rescue support mismatch")
            if pair.get("probe_nonrescue_supported") is not (pair_nonrescue > 0):
                raise ValueError("V7 budget pair nonrescue support mismatch")
            if pair.get("fold_support_closed") is not pair_fold_support:
                raise ValueError("V7 budget pair fold support mismatch")
            if not _same_float(
                pair.get("wilson_lower_bound"),
                wilson_lower_bound(pair_rescues, pair_counts["episodes"]),
            ):
                raise ValueError("V7 budget pair Wilson mismatch")
            all_probe_support = all_probe_support and pair_rescues > 0 and pair_nonrescue > 0
            all_fold_support = all_fold_support and pair_fold_support
            for key, value in pair_counts.items():
                canonical_counts[key] += int(value)

        canonical = canonical_records[canonical_index]
        if int(canonical.get("canonical_index", -1)) != canonical_index or canonical.get("canonical_root") != canonical_root:
            raise ValueError("V7 budget canonical identity mismatch")
        if not isinstance(canonical.get("final_canonical_digest"), str) or len(canonical["final_canonical_digest"]) != 64:
            raise ValueError("V7 budget canonical state digest mismatch")
        if not isinstance(canonical.get("artifact_digest"), str) or len(canonical["artifact_digest"]) != 64:
            raise ValueError("V7 budget shard artifact digest mismatch")
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
                raise ValueError(f"V7 budget canonical sufficient statistic mismatch: {key}")
        prevalence = rescues / canonical_counts["episodes"]
        lower = wilson_lower_bound(rescues, canonical_counts["episodes"])
        if not _same_float(canonical.get("pooled_rescue_prevalence"), prevalence):
            raise ValueError("V7 budget canonical prevalence mismatch")
        if not _same_float(canonical.get("pooled_wilson_lower_bound"), lower):
            raise ValueError("V7 budget canonical Wilson mismatch")
        rebuilt_canonical.append(
            {
                **expected_fields,
                "pooled_rescue_prevalence": prevalence,
                "pooled_wilson_lower_bound": lower,
            }
        )

    classification = classify_budget_support(rebuilt_canonical)
    if receipt.get("train_cell_classification") != classification:
        raise ValueError("V7 budget train-cell classification mismatch")
    prevalence_floor = min(float(item["pooled_wilson_lower_bound"]) for item in rebuilt_canonical)
    if not _same_float(receipt.get("canonical_prevalence_floor"), prevalence_floor):
        raise ValueError("V7 budget canonical prevalence floor mismatch")
    expected_total = expected_pair_episodes * 16
    if int(receipt.get("total_episodes", -1)) != expected_total:
        raise ValueError("V7 budget total episode count mismatch")
    if receipt.get("artifact_digest") != _artifact_digest(receipt):
        raise ValueError("V7 budget artifact digest mismatch")
    code_digest = receipt.get("code_digest")
    if not isinstance(code_digest, str) or len(code_digest) != 64:
        raise ValueError("V7 budget code digest mismatch")
    return {
        "classification": classification,
        "canonical_prevalence_floor": prevalence_floor,
        "code_digest": code_digest,
        "artifact_digest": str(receipt["artifact_digest"]),
        "roots": set(expected_roots["canonical_roots"] + expected_roots["probe_roots"]),
    }


def classify_exp279_multiroot_support_stability_v7_cross_cell(
    cells: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(cells.keys()) != set(_DECISION_BUDGETS):
        raise ValueError("V7 cross-cell decision budgets must be exactly train60 and train120")
    validated = {
        train: _validate_budget_receipt(cells[train], expected_train=train)
        for train in _DECISION_BUDGETS
    }
    if validated[60]["code_digest"] != validated[120]["code_digest"]:
        raise ValueError("V7 cross-cell code provenance mismatch")
    if validated[60]["roots"] & validated[120]["roots"]:
        raise ValueError("V7 cross-cell train60/train120 root sets must be disjoint")

    classifications = {train: str(validated[train]["classification"]) for train in _DECISION_BUDGETS}
    if "MODEL_ROOT_SUPPORT_COLLAPSE" in classifications.values():
        decision = "MODEL_ROOT_SUPPORT_COLLAPSE"
        successor_design_authorized = False
        authorization_scope = "NO_SUCCESSOR_AUTHORIZED"
        future_probe_episodes = None
    elif "PROBE_SUPPORT_INTERMITTENT" in classifications.values():
        decision = "PROBE_SUPPORT_INTERMITTENT"
        successor_design_authorized = False
        authorization_scope = "SUPPORT_COURT_PLANNING_ONLY"
        planned = [
            prospective_probe_episodes(float(validated[train]["canonical_prevalence_floor"]))
            for train in _DECISION_BUDGETS
        ]
        future_probe_episodes = None if any(value is None for value in planned) else max(int(value) for value in planned if value is not None)
    else:
        decision = "SUPPORT_RECURRENT"
        successor_design_authorized = True
        authorization_scope = "DESIGN_NEW_MECHANISM_COURT_ONLY"
        planned = [
            prospective_probe_episodes(float(validated[train]["canonical_prevalence_floor"]))
            for train in _DECISION_BUDGETS
        ]
        future_probe_episodes = None if any(value is None for value in planned) else max(int(value) for value in planned if value is not None)

    sample_sizes = {
        str(train): prospective_probe_episodes(float(validated[train]["canonical_prevalence_floor"]))
        for train in _DECISION_BUDGETS
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_CROSS_CELL,
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": FROZEN_PROTOCOL_DIGEST,
        "code_digest": validated[60]["code_digest"],
        "decision": decision,
        "train_cell_classifications": {str(train): classifications[train] for train in _DECISION_BUDGETS},
        "train_receipt_digests": {str(train): validated[train]["artifact_digest"] for train in _DECISION_BUDGETS},
        "canonical_prevalence_floors": {
            str(train): float(validated[train]["canonical_prevalence_floor"])
            for train in _DECISION_BUDGETS
        },
        "sample_size_recommendations": sample_sizes,
        "future_probe_episodes_per_canonical_root": future_probe_episodes,
        "successor_design_authorized": successor_design_authorized,
        "authorization_scope": authorization_scope,
        "mechanism_successor_authorized": False,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    return receipt
