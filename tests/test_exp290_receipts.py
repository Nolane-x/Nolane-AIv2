from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest


PROTOCOL = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
GEOMETRY = "a" * 64
CODE = "b" * 64
MODEL = "c" * 64
REPOSITORY_HEAD = "f" * 40
EPISODES = 20
SLOT_CAPACITY = 8


def _canonical_without_artifact_digest(value: dict[str, object]) -> bytes:
    payload = deepcopy(value)
    payload.pop("artifact_digest", None)
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _metrics(*, established: bool = True, oracle: bool = True) -> dict[str, object]:
    if not oracle:
        return {
            "primary_endpoint": "source_equivalent_target_dead_end_rate",
            "control_source_equivalent_target_dead_end_rate": 0.25,
            "learned_source_equivalent_target_dead_end_rate": 0.20,
            "oracle_source_equivalent_target_dead_end_rate": 0.25,
            "oracle_headroom": 0.0,
            "learned_headroom": 0.05,
            "learned_oracle_value_capture": 0.0,
            "learned_valid_state_overprune_rate": 0.0,
            "control_verified_solution_rate": 1.0,
            "learned_verified_solution_rate": 1.0,
            "oracle_verified_solution_rate": 1.0,
            "learned_transferred_prune_count": 4,
            "heldout_nonidentity_permutation_rate": 1.0,
            "evaluation_mapping_used_by_learned_mode": False,
            "oracle_mapping_used_by_learned_mode": False,
            "local_only_transfer_activated": False,
            "oracle_transfer_mode_deployable": False,
            "epsilon_denominator_rescue_used": False,
        }
    return {
        "primary_endpoint": "source_equivalent_target_dead_end_rate",
        "control_source_equivalent_target_dead_end_rate": 0.50,
        "learned_source_equivalent_target_dead_end_rate": 0.20 if established else 0.45,
        "oracle_source_equivalent_target_dead_end_rate": 0.0,
        "oracle_headroom": 0.50,
        "learned_headroom": 0.30 if established else 0.05,
        "learned_oracle_value_capture": 0.60 if established else 0.10,
        "learned_valid_state_overprune_rate": 0.0,
        "control_verified_solution_rate": 1.0,
        "learned_verified_solution_rate": 1.0,
        "oracle_verified_solution_rate": 1.0,
        "learned_transferred_prune_count": 4,
        "heldout_nonidentity_permutation_rate": 1.0,
        "evaluation_mapping_used_by_learned_mode": False,
        "oracle_mapping_used_by_learned_mode": False,
        "local_only_transfer_activated": False,
        "oracle_transfer_mode_deployable": False,
        "epsilon_denominator_rescue_used": False,
    }


def _mode(
    mode: str,
    *,
    rate: float,
    transferred_prunes: int,
    oracle_mapping: bool,
) -> dict[str, object]:
    dead_ends = int(round(rate * EPISODES))
    rows: list[dict[str, object]] = []
    for index in range(EPISODES):
        rows.append(
            {
                "episode_index": index,
                "mode": mode,
                "source_clause_count": 1,
                "source_equivalent_opportunities": 1,
                "source_equivalent_target_dead_ends": 1 if index < dead_ends else 0,
                "transferred_prune_count": 1 if index < transferred_prunes else 0,
                "transferred_valid_state_prune_count": 0,
                "target_local_prune_count": 0,
                "target_local_insertion_count": 0,
                "target_local_memory_enabled": True,
                "target_local_insertion_requires_observed_contradiction": True,
                "transfer_scorer_executed": True,
                "fixed_transferred_slot_accounting": True,
                "transferred_slot_capacity": SLOT_CAPACITY,
                "transferred_slot_comparisons_charged": SLOT_CAPACITY,
                "oracle_mapping_delivered": oracle_mapping,
                "evaluation_mapping_delivered": oracle_mapping,
                "evaluator_truth_gates_target_action": False,
                "verified_solution": True,
                "search_steps": 1,
            }
        )
    return {
        "mode": mode,
        "model_digest": MODEL,
        "episodes": EPISODES,
        "source_equivalent_opportunities": EPISODES,
        "source_equivalent_target_dead_ends": dead_ends,
        "source_equivalent_target_dead_end_rate": rate,
        "transferred_prune_count": transferred_prunes,
        "transferred_valid_state_prune_count": 0,
        "valid_state_overprune_rate": 0.0,
        "verified_solution_rate": 1.0,
        "target_local_memory_enabled": True,
        "target_local_insertion_requires_observed_contradiction": True,
        "transfer_scorer_executed": True,
        "fixed_transferred_slot_accounting": True,
        "transferred_slot_capacity": SLOT_CAPACITY,
        "transferred_slot_comparisons_charged": EPISODES * SLOT_CAPACITY,
        "oracle_mapping_delivered": oracle_mapping,
        "evaluation_mapping_delivered": oracle_mapping,
        "evaluator_truth_gates_target_action": False,
        "per_episode": rows,
    }


def _root(index: int, *, established: bool = True, oracle: bool = True) -> dict[str, object]:
    metrics = _metrics(established=established, oracle=oracle)
    if not oracle:
        decision = "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"
    elif established:
        decision = "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"
    else:
        decision = "LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED"
    return {
        "schema": "NLM-EXP-290-STRUCTURAL-CLAUSE-TRANSFER-ROOT-V1",
        "experiment_id": "EXP-290",
        "canonical_index": index,
        "root_seed": f"20260913-exp290-structural-clause-transfer-v1-dev-root-{index}",
        "repository_head": REPOSITORY_HEAD,
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "evaluation_mapping_used_for_training": False,
        "evaluation_mapping_used_by_learned_mode": False,
        "oracle_mapping_used_by_learned_mode": False,
        "oracle_transfer_mode_deployable": False,
        "arbitrary_value_symbol_remapping_claimed": False,
        "cross_domain_transfer_claimed": False,
        "lemma_generation_claimed": False,
        "protocol_digest": PROTOCOL,
        "geometry_digest": GEOMETRY,
        "code_digest": CODE,
        "post_training_model_digest": MODEL,
        "post_evaluation_model_digest": MODEL,
        "model_state_unchanged_during_evaluation": True,
        "training": {
            "rng_stream": "augmentation",
            "evaluation_mapping_used_for_training": False,
            "evaluation_lineage_consumed": False,
            "pair_digests": [f"{index + 1:064x}"],
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "pair_digests": [f"{index + 100:064x}"],
            "evaluation_episode_count": EPISODES,
            "heldout_nonidentity_episode_count": EPISODES,
            "modes": {
                "LOCAL_ONLY_CONTROL": _mode(
                    "LOCAL_ONLY_CONTROL",
                    rate=float(metrics["control_source_equivalent_target_dead_end_rate"]),
                    transferred_prunes=0,
                    oracle_mapping=False,
                ),
                "LEARNED_STRUCTURAL_TRANSFER": _mode(
                    "LEARNED_STRUCTURAL_TRANSFER",
                    rate=float(metrics["learned_source_equivalent_target_dead_end_rate"]),
                    transferred_prunes=int(metrics["learned_transferred_prune_count"]),
                    oracle_mapping=False,
                ),
                "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND": _mode(
                    "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
                    rate=float(metrics["oracle_source_equivalent_target_dead_end_rate"]),
                    transferred_prunes=5,
                    oracle_mapping=True,
                ),
            },
            "common_source_phases": [
                {
                    "source_clause_set_sealed_before_target": True,
                    "evaluator_truth_gated_insertion": False,
                    "oracle_mapping_used": False,
                    "episodes": [{"episode_index": 0, "source_clauses": [{"variable_index": 1, "value": 0}]}],
                }
            ],
        },
        "root_metrics": metrics,
        "decision": decision,
        "successor_design_authorized": False,
        "authorization_scope": "NONE",
        "mechanism_successor_authorized": False,
        "descriptive_notes": {
            "value_label_remapping_claimed": False,
            "cross_domain_transfer_claimed": False,
            "target_local_memory_matches_parent_exp289": True,
        },
    }


def _rehash(payload: dict[str, object]) -> None:
    payload["artifact_digest"] = hashlib.sha256(_canonical_without_artifact_digest(payload)).hexdigest()


def test_exp290_root_receipt_seals_canonical_bytes_and_digest() -> None:
    from nolane_ai.experiments.exp290_receipts import (
        canonical_receipt_bytes,
        seal_root_receipt,
        validate_root_receipt,
    )

    sealed = seal_root_receipt(_root(0))
    validate_root_receipt(sealed)
    encoded = canonical_receipt_bytes(sealed)
    assert encoded.endswith(b"\n")
    assert encoded == (json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert sealed["artifact_digest"] == hashlib.sha256(_canonical_without_artifact_digest(sealed)).hexdigest()
    assert len(sealed["source_clause_set_digest"]) == 64


def test_exp290_validator_recomputes_semantic_decision_even_after_rehash() -> None:
    from nolane_ai.experiments.exp290_receipts import seal_root_receipt, validate_root_receipt

    forged = seal_root_receipt(_root(0, established=False))
    forged["decision"] = "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"
    _rehash(forged)
    with pytest.raises(ValueError, match="decision"):
        validate_root_receipt(forged)


def test_exp290_validator_recomputes_mode_aggregates_from_per_episode_primitives() -> None:
    from nolane_ai.experiments.exp290_receipts import seal_root_receipt, validate_root_receipt

    forged = seal_root_receipt(_root(0, established=False))
    learned = forged["evaluation"]["modes"]["LEARNED_STRUCTURAL_TRANSFER"]
    learned["source_equivalent_target_dead_end_rate"] = 0.20
    learned["source_equivalent_target_dead_ends"] = 4
    forged["root_metrics"]["learned_source_equivalent_target_dead_end_rate"] = 0.20
    forged["root_metrics"]["learned_headroom"] = 0.30
    forged["root_metrics"]["learned_oracle_value_capture"] = 0.60
    forged["decision"] = "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"
    _rehash(forged)
    with pytest.raises(ValueError, match="aggregate|per-episode|primitive"):
        validate_root_receipt(forged)


def test_exp290_validator_binds_root_metrics_to_mode_aggregates() -> None:
    from nolane_ai.experiments.exp290_receipts import seal_root_receipt, validate_root_receipt

    forged = seal_root_receipt(_root(0, established=False))
    forged["root_metrics"]["learned_source_equivalent_target_dead_end_rate"] = 0.20
    forged["root_metrics"]["learned_headroom"] = 0.30
    forged["root_metrics"]["learned_oracle_value_capture"] = 0.60
    forged["decision"] = "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"
    _rehash(forged)
    with pytest.raises(ValueError, match="root_metrics|mode aggregate"):
        validate_root_receipt(forged)


def test_exp290_validator_binds_nonidentity_rate_to_evaluation_counts() -> None:
    from nolane_ai.experiments.exp290_receipts import seal_root_receipt, validate_root_receipt

    forged = seal_root_receipt(_root(0))
    forged["evaluation"]["heldout_nonidentity_episode_count"] = EPISODES - 1
    _rehash(forged)
    with pytest.raises(ValueError, match="nonidentity"):
        validate_root_receipt(forged)


def test_exp290_root_validator_rejects_boundary_and_lineage_leakage() -> None:
    from nolane_ai.experiments.exp290_receipts import seal_root_receipt, validate_root_receipt

    cases = []
    leaked = _root(0)
    leaked["evaluation_mapping_used_by_learned_mode"] = True
    cases.append(leaked)
    trained = _root(0)
    trained["evaluation_mapping_used_for_training"] = True
    cases.append(trained)
    promoted = _root(0)
    promoted["promotion_claimed"] = True
    cases.append(promoted)
    remap_claim = _root(0)
    remap_claim["arbitrary_value_symbol_remapping_claimed"] = True
    cases.append(remap_claim)
    lemma_claim = _root(0)
    lemma_claim["lemma_generation_claimed"] = True
    cases.append(lemma_claim)
    drifted = _root(0)
    drifted["post_evaluation_model_digest"] = "d" * 64
    cases.append(drifted)
    overlap = _root(0)
    overlap["evaluation"]["pair_digests"] = list(overlap["training"]["pair_digests"])
    cases.append(overlap)

    for payload in cases:
        sealed = seal_root_receipt(payload)
        with pytest.raises(ValueError):
            validate_root_receipt(sealed)


def test_exp290_cross_reducer_requires_exact_four_unique_matching_roots() -> None:
    from nolane_ai.experiments.exp290_receipts import reduce_cross_roots, seal_root_receipt

    roots = [seal_root_receipt(_root(index)) for index in range(4)]
    recurrent = reduce_cross_roots(roots)
    assert recurrent["decision"] == "STRUCTURAL_CLAUSE_TRANSFER_RECURRENT"
    assert recurrent["successor_design_authorized"] is True
    assert recurrent["authorization_scope"] == "DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY"
    assert recurrent["mechanism_successor_authorized"] is False
    assert recurrent["scientific_evidence_eligible"] is False
    assert recurrent["confirmatory_data_consumed"] is False
    assert recurrent["challenge_materialized"] is False
    assert recurrent["promotion_claimed"] is False
    assert recurrent["repository_head"] == REPOSITORY_HEAD

    with pytest.raises(ValueError, match="canonical roots"):
        reduce_cross_roots(roots[:3])
    with pytest.raises(ValueError, match="canonical roots"):
        reduce_cross_roots([roots[0], roots[0], roots[2], roots[3]])

    mixed = deepcopy(roots)
    mixed[2]["code_digest"] = "e" * 64
    _rehash(mixed[2])
    with pytest.raises(ValueError, match="code_digest"):
        reduce_cross_roots(mixed)


def test_exp290_cross_decision_priority_is_frozen() -> None:
    from nolane_ai.experiments.exp290_receipts import reduce_cross_roots, seal_root_receipt

    established = [seal_root_receipt(_root(index)) for index in range(4)]
    intermittent = deepcopy(established)
    intermittent[1] = seal_root_receipt(_root(1, established=False))
    result = reduce_cross_roots(intermittent)
    assert result["decision"] == "STRUCTURAL_CLAUSE_TRANSFER_INTERMITTENT"
    assert result["successor_design_authorized"] is False
    assert result["authorization_scope"] == "NONE"

    negative = [seal_root_receipt(_root(index, established=False)) for index in range(4)]
    result = reduce_cross_roots(negative)
    assert result["decision"] == "STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED"
    assert result["successor_design_authorized"] is False

    incomplete = [seal_root_receipt(_root(index)) for index in range(4)]
    incomplete[3] = seal_root_receipt(_root(3, oracle=False))
    result = reduce_cross_roots(incomplete)
    assert result["decision"] == "ORACLE_TRANSFER_REPLICATION_INCOMPLETE"
    assert result["successor_design_authorized"] is False


def test_exp290_cross_receipt_is_self_validating_and_canonical() -> None:
    from nolane_ai.experiments.exp290_receipts import (
        canonical_receipt_bytes,
        reduce_cross_roots,
        seal_root_receipt,
        validate_cross_receipt,
    )

    roots = [seal_root_receipt(_root(index)) for index in range(4)]
    cross = reduce_cross_roots(roots)
    validate_cross_receipt(cross)
    assert set(cross["source_root_artifact_digests"]) == {
        root["artifact_digest"] for root in roots
    }
    encoded = canonical_receipt_bytes(cross)
    assert encoded.endswith(b"\n")
    assert cross["artifact_digest"] == hashlib.sha256(_canonical_without_artifact_digest(cross)).hexdigest()

    forged = deepcopy(cross)
    forged["successor_design_authorized"] = False
    _rehash(forged)
    with pytest.raises(ValueError):
        validate_cross_receipt(forged)
