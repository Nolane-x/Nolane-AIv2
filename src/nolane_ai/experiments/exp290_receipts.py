from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


ROOT_SCHEMA = "NLM-EXP-290-STRUCTURAL-CLAUSE-TRANSFER-ROOT-V1"
CROSS_SCHEMA = "NLM-EXP-290-STRUCTURAL-CLAUSE-TRANSFER-CROSS-V1"
ROOT_ESTABLISHED = "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"
ROOT_NOT_ESTABLISHED = "LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED"
ORACLE_NOT_REPLICATED = "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"
CROSS_RECURRENT = "STRUCTURAL_CLAUSE_TRANSFER_RECURRENT"
CROSS_INTERMITTENT = "STRUCTURAL_CLAUSE_TRANSFER_INTERMITTENT"
CROSS_NOT_ESTABLISHED = "STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED"
CROSS_ORACLE_INCOMPLETE = "ORACLE_TRANSFER_REPLICATION_INCOMPLETE"
EXP291_SCOPE = "DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY"
TARGET_MODES = (
    "LOCAL_ONLY_CONTROL",
    "LEARNED_STRUCTURAL_TRANSFER",
    "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
)


def canonical_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical_without_artifact_digest(value: Mapping[str, Any]) -> bytes:
    payload = deepcopy(dict(value))
    payload.pop("artifact_digest", None)
    return canonical_receipt_bytes(payload)


def _artifact_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_without_artifact_digest(value)).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _require_git_sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{name} must be a 40-character git SHA")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _require_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise ValueError(f"{key} must be false")


def _require_nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _classify_root(metrics: Mapping[str, Any]) -> str:
    oracle_headroom = float(metrics["oracle_headroom"])
    if oracle_headroom <= 0.0:
        return ORACLE_NOT_REPLICATED
    learned_solution_floor = float(metrics["control_verified_solution_rate"]) - 0.01
    success = (
        float(metrics["learned_headroom"]) > 0.0
        and float(metrics["learned_oracle_value_capture"]) >= 0.50
        and float(metrics["learned_valid_state_overprune_rate"]) <= 0.005
        and float(metrics["learned_verified_solution_rate"]) >= learned_solution_floor
        and int(metrics["learned_transferred_prune_count"]) > 0
        and float(metrics["heldout_nonidentity_permutation_rate"]) == 1.0
        and metrics["evaluation_mapping_used_by_learned_mode"] is False
        and metrics["oracle_mapping_used_by_learned_mode"] is False
    )
    return ROOT_ESTABLISHED if success else ROOT_NOT_ESTABLISHED


def _source_clause_set_digest(payload: Mapping[str, Any]) -> str:
    phases = payload["evaluation"]["common_source_phases"]
    clause_payload: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(phases):
        for episode in phase["episodes"]:
            clause_payload.append(
                {
                    "phase_index": int(phase_index),
                    "episode_index": int(episode.get("episode_index", len(clause_payload))),
                    "source_clauses": episode["source_clauses"],
                }
            )
    return hashlib.sha256(canonical_receipt_bytes({"source_clause_sets": clause_payload})).hexdigest()


def seal_root_receipt(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(raw))
    payload["source_clause_set_digest"] = _source_clause_set_digest(payload)
    payload.pop("artifact_digest", None)
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def _assert_close(name: str, observed: float, expected: float, *, tolerance: float = 1e-12) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise ValueError(f"{name} must be finite")
    if abs(observed - expected) > tolerance:
        raise ValueError(f"{name} is inconsistent with primitive rates")


def _recompute_mode_aggregate(mode_name: str, mode: Mapping[str, Any]) -> dict[str, Any]:
    if mode.get("mode") != mode_name:
        raise ValueError("mode aggregate name is inconsistent with target mode identity")
    rows = mode.get("per_episode")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("mode aggregate must contain non-empty per-episode primitives")

    slot_capacity = _require_nonnegative_int(
        "transferred_slot_capacity", mode.get("transferred_slot_capacity")
    )
    if slot_capacity <= 0:
        raise ValueError("transferred_slot_capacity must be positive")

    opportunities = 0
    dead_ends = 0
    transferred_prunes = 0
    valid_state_prunes = 0
    solutions = 0
    charged_comparisons = 0
    expected_oracle_delivery = mode_name == "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND"

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("per-episode primitive must be an object")
        if row.get("mode") != mode_name:
            raise ValueError("per-episode primitive mode mismatch")
        row_capacity = _require_nonnegative_int(
            "per-episode transferred_slot_capacity", row.get("transferred_slot_capacity")
        )
        if row_capacity != slot_capacity:
            raise ValueError("per-episode primitive transferred-slot capacity mismatch")
        row_opportunities = _require_nonnegative_int(
            "per-episode source_equivalent_opportunities",
            row.get("source_equivalent_opportunities"),
        )
        row_dead_ends = _require_nonnegative_int(
            "per-episode source_equivalent_target_dead_ends",
            row.get("source_equivalent_target_dead_ends"),
        )
        row_transferred = _require_nonnegative_int(
            "per-episode transferred_prune_count", row.get("transferred_prune_count")
        )
        row_valid = _require_nonnegative_int(
            "per-episode transferred_valid_state_prune_count",
            row.get("transferred_valid_state_prune_count"),
        )
        row_steps = _require_nonnegative_int("per-episode search_steps", row.get("search_steps"))
        row_comparisons = _require_nonnegative_int(
            "per-episode transferred_slot_comparisons_charged",
            row.get("transferred_slot_comparisons_charged"),
        )
        if row_dead_ends > row_opportunities:
            raise ValueError("per-episode primitive dead ends exceed transfer opportunities")
        if row_valid > row_transferred:
            raise ValueError("per-episode primitive valid-state prunes exceed transferred prunes")
        if row_comparisons != row_steps * slot_capacity:
            raise ValueError("per-episode primitive transferred-slot comparison charge mismatch")
        if row.get("target_local_memory_enabled") is not True:
            raise ValueError("per-episode primitive must keep target-local memory enabled")
        if row.get("target_local_insertion_requires_observed_contradiction") is not True:
            raise ValueError("per-episode primitive must require contradiction before local insertion")
        if row.get("transfer_scorer_executed") is not True:
            raise ValueError("per-episode primitive must execute transfer scorer")
        if row.get("fixed_transferred_slot_accounting") is not True:
            raise ValueError("per-episode primitive must use fixed transferred-slot accounting")
        if row.get("evaluator_truth_gates_target_action") is not False:
            raise ValueError("per-episode primitive cannot gate action on evaluator truth")
        if row.get("oracle_mapping_delivered") is not expected_oracle_delivery:
            raise ValueError("per-episode primitive oracle mapping delivery mismatch")
        if row.get("evaluation_mapping_delivered") is not expected_oracle_delivery:
            raise ValueError("per-episode primitive evaluation mapping delivery mismatch")

        opportunities += row_opportunities
        dead_ends += row_dead_ends
        transferred_prunes += row_transferred
        valid_state_prunes += row_valid
        charged_comparisons += row_comparisons
        solutions += row.get("verified_solution") is True

    episodes = len(rows)
    dead_end_rate = float(dead_ends / opportunities) if opportunities else 0.0
    overprune_rate = float(valid_state_prunes / transferred_prunes) if transferred_prunes else 0.0
    solution_rate = float(solutions / episodes)

    exact_fields = {
        "episodes": episodes,
        "source_equivalent_opportunities": opportunities,
        "source_equivalent_target_dead_ends": dead_ends,
        "transferred_prune_count": transferred_prunes,
        "transferred_valid_state_prune_count": valid_state_prunes,
        "transferred_slot_comparisons_charged": charged_comparisons,
    }
    for key, expected in exact_fields.items():
        if mode.get(key) != expected:
            raise ValueError(
                f"mode aggregate {key} is inconsistent with per-episode primitives"
            )
    _assert_close(
        "mode aggregate source_equivalent_target_dead_end_rate",
        float(mode["source_equivalent_target_dead_end_rate"]),
        dead_end_rate,
    )
    _assert_close(
        "mode aggregate valid_state_overprune_rate",
        float(mode["valid_state_overprune_rate"]),
        overprune_rate,
    )
    _assert_close(
        "mode aggregate verified_solution_rate",
        float(mode["verified_solution_rate"]),
        solution_rate,
    )
    if mode.get("target_local_memory_enabled") is not True:
        raise ValueError("mode aggregate must keep target-local memory enabled")
    if mode.get("target_local_insertion_requires_observed_contradiction") is not True:
        raise ValueError("mode aggregate must require contradiction before local insertion")
    if mode.get("transfer_scorer_executed") is not True:
        raise ValueError("mode aggregate must execute transfer scorer")
    if mode.get("fixed_transferred_slot_accounting") is not True:
        raise ValueError("mode aggregate must use fixed transferred-slot accounting")
    if mode.get("evaluator_truth_gates_target_action") is not False:
        raise ValueError("mode aggregate cannot gate action on evaluator truth")
    if mode.get("oracle_mapping_delivered") is not expected_oracle_delivery:
        raise ValueError("mode aggregate oracle mapping delivery mismatch")
    if mode.get("evaluation_mapping_delivered") is not expected_oracle_delivery:
        raise ValueError("mode aggregate evaluation mapping delivery mismatch")
    if mode_name == "LOCAL_ONLY_CONTROL" and transferred_prunes != 0:
        raise ValueError("mode aggregate control unexpectedly activated transferred clauses")

    return {
        "episodes": episodes,
        "source_equivalent_opportunities": opportunities,
        "source_equivalent_target_dead_ends": dead_ends,
        "source_equivalent_target_dead_end_rate": dead_end_rate,
        "transferred_prune_count": transferred_prunes,
        "transferred_valid_state_prune_count": valid_state_prunes,
        "valid_state_overprune_rate": overprune_rate,
        "verified_solution_rate": solution_rate,
        "transferred_slot_capacity": slot_capacity,
        "transferred_slot_comparisons_charged": charged_comparisons,
    }


def validate_root_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != ROOT_SCHEMA:
        raise ValueError("invalid EXP-290 root schema")
    if receipt.get("experiment_id") != "EXP-290":
        raise ValueError("root receipt must bind EXP-290")
    canonical_index = receipt.get("canonical_index")
    if type(canonical_index) is not int or canonical_index not in {0, 1, 2, 3}:
        raise ValueError("canonical_index must be one of 0,1,2,3")
    expected_root = f"20260913-exp290-structural-clause-transfer-v1-dev-root-{canonical_index}"
    if receipt.get("root_seed") != expected_root:
        raise ValueError("root_seed does not match frozen canonical identity")

    _require_git_sha("repository_head", receipt.get("repository_head"))
    for key in ("protocol_digest", "geometry_digest", "code_digest"):
        _require_sha256(key, receipt.get(key))
    _require_sha256("post_training_model_digest", receipt.get("post_training_model_digest"))
    _require_sha256("post_evaluation_model_digest", receipt.get("post_evaluation_model_digest"))
    _require_sha256("source_clause_set_digest", receipt.get("source_clause_set_digest"))

    if receipt.get("evidence_level") != "EV-E2":
        raise ValueError("EXP-290 V1 root evidence level must remain EV-E2")
    for key in (
        "scientific_evidence_eligible",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
        "stage_a_protocol_modified",
        "evaluation_mapping_used_for_training",
        "evaluation_mapping_used_by_learned_mode",
        "oracle_mapping_used_by_learned_mode",
        "oracle_transfer_mode_deployable",
        "arbitrary_value_symbol_remapping_claimed",
        "cross_domain_transfer_claimed",
        "lemma_generation_claimed",
    ):
        _require_false(receipt, key)
    if receipt.get("successor_design_authorized") is not False:
        raise ValueError("root receipts never authorize a successor design")
    if receipt.get("authorization_scope") != "NONE":
        raise ValueError("root authorization_scope must be NONE")
    if receipt.get("mechanism_successor_authorized") is not False:
        raise ValueError("root mechanism successor authorization must be false")

    if receipt.get("model_state_unchanged_during_evaluation") is not True:
        raise ValueError("model state must remain frozen during evaluation")
    if receipt["post_training_model_digest"] != receipt["post_evaluation_model_digest"]:
        raise ValueError("model digest drifted during target evaluation")

    training = receipt.get("training")
    evaluation = receipt.get("evaluation")
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("root receipt must contain training and evaluation mappings")
    if training.get("rng_stream") != "augmentation":
        raise ValueError("training must use augmentation lineage only")
    if training.get("evaluation_mapping_used_for_training") is not False:
        raise ValueError("evaluation mapping cannot train the transfer mechanism")
    if training.get("evaluation_lineage_consumed") is not False:
        raise ValueError("evaluation lineage cannot be consumed during training")
    if evaluation.get("rng_stream") != "evaluation":
        raise ValueError("held-out root execution must use evaluation lineage")
    train_pairs = list(training.get("pair_digests") or [])
    eval_pairs = list(evaluation.get("pair_digests") or [])
    if not train_pairs or not eval_pairs:
        raise ValueError("training and evaluation pair digests must be non-empty")
    for digest in train_pairs + eval_pairs:
        _require_sha256("pair_digest", digest)
    if set(train_pairs) & set(eval_pairs):
        raise ValueError("training/evaluation pair lineage overlap")

    modes = evaluation.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != set(TARGET_MODES):
        raise ValueError("root evaluation must contain exactly the three frozen target modes")
    model_digest = receipt["post_evaluation_model_digest"]
    recomputed_modes: dict[str, dict[str, Any]] = {}
    for mode_name in TARGET_MODES:
        mode = modes[mode_name]
        if not isinstance(mode, Mapping):
            raise ValueError("target mode receipt must be an object")
        if mode.get("model_digest") != model_digest:
            raise ValueError("target modes must share the exact frozen model digest")
        recomputed_modes[mode_name] = _recompute_mode_aggregate(mode_name, mode)

    evaluation_episode_count = _require_nonnegative_int(
        "evaluation_episode_count", evaluation.get("evaluation_episode_count")
    )
    heldout_nonidentity_count = _require_nonnegative_int(
        "heldout_nonidentity_episode_count",
        evaluation.get("heldout_nonidentity_episode_count"),
    )
    if evaluation_episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    if heldout_nonidentity_count > evaluation_episode_count:
        raise ValueError("heldout nonidentity episode count exceeds evaluation episode count")
    for mode_name, aggregate in recomputed_modes.items():
        if aggregate["episodes"] != evaluation_episode_count:
            raise ValueError(
                f"mode aggregate {mode_name} episode count does not match evaluation_episode_count"
            )

    learned = modes["LEARNED_STRUCTURAL_TRANSFER"]
    if learned.get("oracle_mapping_delivered") is not False:
        raise ValueError("learned mode received oracle mapping")
    if learned.get("evaluation_mapping_delivered") is not False:
        raise ValueError("learned mode received evaluation mapping")
    oracle = modes["ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND"]
    if oracle.get("oracle_mapping_delivered") is not True:
        raise ValueError("oracle upper bound must explicitly record mapping delivery")
    if oracle.get("evaluation_mapping_delivered") is not True:
        raise ValueError("oracle upper bound must explicitly record evaluator mapping delivery")

    phases = evaluation.get("common_source_phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)) or not phases:
        raise ValueError("common source phases must be non-empty")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("common source phase must be an object")
        if phase.get("source_clause_set_sealed_before_target") is not True:
            raise ValueError("source clause set must be sealed before target execution")
        if phase.get("evaluator_truth_gated_insertion") is not False:
            raise ValueError("evaluator truth cannot gate source insertion")
        if phase.get("oracle_mapping_used") is not False:
            raise ValueError("oracle mapping cannot enter the common source phase")
    if receipt["source_clause_set_digest"] != _source_clause_set_digest(receipt):
        raise ValueError("source_clause_set_digest mismatch")

    metrics = receipt.get("root_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("root_metrics must be an object")
    if metrics.get("primary_endpoint") != "source_equivalent_target_dead_end_rate":
        raise ValueError("unexpected EXP-290 primary endpoint")

    control_aggregate = recomputed_modes["LOCAL_ONLY_CONTROL"]
    learned_aggregate = recomputed_modes["LEARNED_STRUCTURAL_TRANSFER"]
    oracle_aggregate = recomputed_modes["ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND"]
    expected_metric_values = {
        "control_source_equivalent_target_dead_end_rate": control_aggregate[
            "source_equivalent_target_dead_end_rate"
        ],
        "learned_source_equivalent_target_dead_end_rate": learned_aggregate[
            "source_equivalent_target_dead_end_rate"
        ],
        "oracle_source_equivalent_target_dead_end_rate": oracle_aggregate[
            "source_equivalent_target_dead_end_rate"
        ],
        "learned_valid_state_overprune_rate": learned_aggregate[
            "valid_state_overprune_rate"
        ],
        "control_verified_solution_rate": control_aggregate["verified_solution_rate"],
        "learned_verified_solution_rate": learned_aggregate["verified_solution_rate"],
        "oracle_verified_solution_rate": oracle_aggregate["verified_solution_rate"],
        "heldout_nonidentity_permutation_rate": float(
            heldout_nonidentity_count / evaluation_episode_count
        ),
    }
    for key, expected in expected_metric_values.items():
        _assert_close(
            f"root_metrics {key} inconsistent with mode aggregate",
            float(metrics[key]),
            float(expected),
        )
    if int(metrics["learned_transferred_prune_count"]) != int(
        learned_aggregate["transferred_prune_count"]
    ):
        raise ValueError(
            "root_metrics learned_transferred_prune_count inconsistent with mode aggregate"
        )
    if metrics.get("evaluation_mapping_used_by_learned_mode") is not False:
        raise ValueError("root_metrics learned mapping leakage flag must be false")
    if metrics.get("oracle_mapping_used_by_learned_mode") is not False:
        raise ValueError("root_metrics learned oracle leakage flag must be false")
    if metrics.get("local_only_transfer_activated") is not False:
        raise ValueError("root_metrics local-only transfer activation must be false")
    if metrics.get("oracle_transfer_mode_deployable") is not False:
        raise ValueError("root_metrics oracle transfer mode deployability must be false")
    if metrics.get("epsilon_denominator_rescue_used") is not False:
        raise ValueError("root_metrics epsilon denominator rescue flag must be false")

    r0 = float(expected_metric_values["control_source_equivalent_target_dead_end_rate"])
    rl = float(expected_metric_values["learned_source_equivalent_target_dead_end_rate"])
    ro = float(expected_metric_values["oracle_source_equivalent_target_dead_end_rate"])
    oracle_headroom = r0 - ro
    learned_headroom = r0 - rl
    _assert_close("oracle_headroom", float(metrics["oracle_headroom"]), oracle_headroom)
    _assert_close("learned_headroom", float(metrics["learned_headroom"]), learned_headroom)
    expected_capture = learned_headroom / oracle_headroom if oracle_headroom > 0.0 else 0.0
    _assert_close(
        "learned_oracle_value_capture",
        float(metrics["learned_oracle_value_capture"]),
        expected_capture,
    )
    expected_nonidentity_rate = float(heldout_nonidentity_count / evaluation_episode_count)
    _assert_close(
        "heldout nonidentity permutation rate",
        float(metrics["heldout_nonidentity_permutation_rate"]),
        expected_nonidentity_rate,
    )
    expected_decision = _classify_root(metrics)
    if receipt.get("decision") != expected_decision:
        raise ValueError("root decision does not match recomputed frozen predicates")

    expected_artifact_digest = _artifact_digest(receipt)
    if receipt.get("artifact_digest") != expected_artifact_digest:
        raise ValueError("root artifact_digest mismatch")


def _cross_decision(decisions: Sequence[str]) -> str:
    if any(decision == ORACLE_NOT_REPLICATED for decision in decisions):
        return CROSS_ORACLE_INCOMPLETE
    established = sum(decision == ROOT_ESTABLISHED for decision in decisions)
    if established == 4:
        return CROSS_RECURRENT
    if established == 0:
        return CROSS_NOT_ESTABLISHED
    return CROSS_INTERMITTENT


def reduce_cross_roots(roots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roots = [deepcopy(dict(root)) for root in roots]
    if len(roots) != 4 or {root.get("canonical_index") for root in roots} != {0, 1, 2, 3}:
        raise ValueError("EXP-290 cross reducer requires exact canonical roots {0,1,2,3}")
    for root in roots:
        validate_root_receipt(root)
    roots.sort(key=lambda root: int(root["canonical_index"]))

    for identity_key in ("repository_head", "protocol_digest", "geometry_digest", "code_digest"):
        values = {root[identity_key] for root in roots}
        if len(values) != 1:
            raise ValueError(f"mixed {identity_key} across EXP-290 roots")

    decisions = [str(root["decision"]) for root in roots]
    decision = _cross_decision(decisions)
    authorized = decision == CROSS_RECURRENT
    payload: dict[str, Any] = {
        "schema": CROSS_SCHEMA,
        "experiment_id": "EXP-290",
        "evidence_level": "EV-E2",
        "repository_head": roots[0]["repository_head"],
        "protocol_digest": roots[0]["protocol_digest"],
        "geometry_digest": roots[0]["geometry_digest"],
        "code_digest": roots[0]["code_digest"],
        "canonical_indices": [0, 1, 2, 3],
        "source_root_artifact_digests": [root["artifact_digest"] for root in roots],
        "source_root_decisions": [
            {
                "canonical_index": int(root["canonical_index"]),
                "decision": root["decision"],
                "artifact_digest": root["artifact_digest"],
                "model_digest": root["post_evaluation_model_digest"],
                "source_clause_set_digest": root["source_clause_set_digest"],
            }
            for root in roots
        ],
        "decision": decision,
        "successor_design_authorized": authorized,
        "authorization_scope": EXP291_SCOPE if authorized else "NONE",
        "mechanism_successor_authorized": False,
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "arbitrary_value_symbol_remapping_claimed": False,
        "cross_domain_transfer_claimed": False,
        "lemma_generation_claimed": False,
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    validate_cross_receipt(payload)
    return payload


def validate_cross_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != CROSS_SCHEMA:
        raise ValueError("invalid EXP-290 cross schema")
    if receipt.get("experiment_id") != "EXP-290":
        raise ValueError("cross receipt must bind EXP-290")
    if receipt.get("evidence_level") != "EV-E2":
        raise ValueError("EXP-290 cross evidence level must remain EV-E2")
    _require_git_sha("repository_head", receipt.get("repository_head"))
    for key in ("protocol_digest", "geometry_digest", "code_digest", "artifact_digest"):
        _require_sha256(key, receipt.get(key))
    if receipt.get("canonical_indices") != [0, 1, 2, 3]:
        raise ValueError("cross receipt must bind exact canonical roots {0,1,2,3}")

    source_digests = receipt.get("source_root_artifact_digests")
    source_rows = receipt.get("source_root_decisions")
    if not isinstance(source_digests, list) or len(source_digests) != 4:
        raise ValueError("cross receipt must bind four root artifact digests")
    if not isinstance(source_rows, list) or len(source_rows) != 4:
        raise ValueError("cross receipt must bind four root decision rows")
    for digest in source_digests:
        _require_sha256("source_root_artifact_digest", digest)
    rows = sorted(source_rows, key=lambda row: int(row["canonical_index"]))
    if [int(row["canonical_index"]) for row in rows] != [0, 1, 2, 3]:
        raise ValueError("cross source decisions must bind roots 0,1,2,3 exactly once")
    for row in rows:
        _require_sha256("root artifact_digest", row["artifact_digest"])
        _require_sha256("root model_digest", row["model_digest"])
        _require_sha256("root source_clause_set_digest", row["source_clause_set_digest"])
    if [row["artifact_digest"] for row in rows] != source_digests:
        raise ValueError("source root artifact digest order mismatch")

    expected_decision = _cross_decision([str(row["decision"]) for row in rows])
    if receipt.get("decision") != expected_decision:
        raise ValueError("cross decision does not match frozen priority rule")
    expected_authorized = expected_decision == CROSS_RECURRENT
    if receipt.get("successor_design_authorized") is not expected_authorized:
        raise ValueError("cross successor_design_authorized does not match decision")
    expected_scope = EXP291_SCOPE if expected_authorized else "NONE"
    if receipt.get("authorization_scope") != expected_scope:
        raise ValueError("cross authorization_scope does not match decision")
    if receipt.get("mechanism_successor_authorized") is not False:
        raise ValueError("EXP-290 never directly authorizes successor mechanism implementation")
    for key in (
        "scientific_evidence_eligible",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "promotion_claimed",
        "stage_a_protocol_modified",
        "arbitrary_value_symbol_remapping_claimed",
        "cross_domain_transfer_claimed",
        "lemma_generation_claimed",
    ):
        _require_false(receipt, key)
    if receipt.get("artifact_digest") != _artifact_digest(receipt):
        raise ValueError("cross artifact_digest mismatch")
