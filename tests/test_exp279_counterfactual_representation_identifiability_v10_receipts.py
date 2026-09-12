from __future__ import annotations

from copy import deepcopy
import json

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_representation_identifiability_v10 import (
    REPRESENTATION_DIMENSIONS,
    REPRESENTATION_VIEWS,
    classify_representation_root,
)
from nolane_ai.experiments.exp279_counterfactual_representation_identifiability_v10_runner import (
    FROZEN_GEOMETRY,
    FROZEN_NN,
    FROZEN_OPTIMIZER,
    PROTOCOL_DIGEST,
    build_root_schedule,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.experiments.exp279_counterfactual_representation_identifiability_v10_receipts import (
    build_budget_receipt,
    build_cross_receipt,
    canonical_receipt_bytes,
    receipt_sha256,
    validate_budget_receipt,
    validate_cross_receipt,
    validate_shard_receipt,
)


def _metrics():
    return {
        "episodes": 2048,
        "route_count": 256,
        "route_fraction": 0.125,
        "policy_solutions": 1100,
        "policy_flops": 230400.0,
        "stop_successes": 900,
        "branch_successes": 1050,
        "stop_flops": 204800.0,
        "branch_flops": 409600.0,
        "policy_utility": 1100 / 230400.0,
        "stop_utility": 900 / 204800.0,
        "branch_utility": 1050 / 409600.0,
        "selected_rescues": 220,
        "selected_harms": 10,
        "selected_both_success": 20,
        "selected_both_failure": 6,
        "raw_rescues": 300,
        "raw_rescue_prevalence": 300 / 2048,
        "selected_rescue_prevalence": 220 / 256,
        "rescue_enrichment": (220 / 256) / (300 / 2048),
        "provenance_closed": True,
    }


def _shard(budget: int, canonical: int):
    metrics = _metrics()
    roots = build_root_schedule(budget, canonical)
    receipt = {
        "schema": "NLM-EXP-279-V10-CRIC-SHARD-V1",
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "analysis_scope": "development_counterfactual_representation_identifiability_upper_bound",
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": PROTOCOL_DIGEST,
        "code_digest": "a" * 64,
        "scientific_branch_head": "b" * 40,
        "executed_commit": "c" * 40,
        "train_replicates": budget,
        "canonical_index": canonical,
        "roots": roots,
        "model_init_seed": 1234 + canonical,
        "world_model_geometry": dict(FROZEN_GEOMETRY),
        "optimizer": dict(FROZEN_OPTIMIZER),
        "nearest_neighbor_contract": dict(FROZEN_NN),
        "fit_replicates_per_root": 128,
        "heldout_replicates_per_root": 128,
        "training_batches_digest": "d" * 64,
        "fit_root_batch_digests": {root: "e" * 64 for root in roots["fit_roots"]},
        "heldout_root_batch_digests": {root: "f" * 64 for root in roots["heldout_roots"]},
        "training_loss_summary": {"count": budget, "mean": 0.5, "final": 0.4},
        "canonical_model_digest": "1" * 64,
        "canonical_model_digest_after_court": "1" * 64,
        "matched_resource_audit": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "active_functional_parameter_match": True,
            "optimizer_visible_parameter_match": True,
            "reclaimed_parameter_assignment_closed": True,
            "compute_budget_closed": True,
        },
        "compute_ledger": {
            "stop_accounted_flops_per_episode": 100.0,
            "branch_accounted_flops_per_episode": 200.0,
        },
        "fit_outcome_counts": {
            "RESCUE": 300,
            "HARM": 100,
            "BOTH_SUCCESS": 700,
            "BOTH_FAILURE": 948,
        },
        "representations": {
            view: {
                "dimension": REPRESENTATION_DIMENSIONS[view],
                "fit_branch_labels": 300,
                "fit_stop_labels": 1748,
                "heldout_metrics": deepcopy(metrics),
                "root_classification": classify_representation_root(metrics),
            }
            for view in REPRESENTATION_VIEWS
        },
        "evaluation_rng_used": False,
        "evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "mechanism_successor_authorized": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
    }
    receipt["artifact_digest"] = canonical_sha256(receipt)
    return receipt


def test_shard_receipt_validates_and_roundtrips_canonically():
    receipt = _shard(60, 0)
    validate_shard_receipt(receipt)
    payload = canonical_receipt_bytes(receipt)
    assert payload == canonical_receipt_bytes(json.loads(payload))
    assert receipt_sha256(payload) == receipt_sha256(receipt)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda r: r.__setitem__("protocol_digest", "0" * 64),
        lambda r: r.__setitem__("evaluation_rng_used", True),
        lambda r: r.__setitem__("canonical_model_digest_after_court", "2" * 64),
        lambda r: r["representations"][REPRESENTATION_VIEWS[0]].__setitem__(
            "root_classification", "REPRESENTATION_NOT_IDENTIFIABLE"
        ),
        lambda r: r["nearest_neighbor_contract"].__setitem__("k", 3),
    ],
)
def test_shard_receipt_rejects_boundary_or_classification_forgery(mutator):
    receipt = _shard(60, 0)
    mutator(receipt)
    receipt["artifact_digest"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "artifact_digest"}
    )
    with pytest.raises((ValueError, RuntimeError)):
        validate_shard_receipt(receipt)


def test_budget_builder_requires_four_roots_and_recomputes_classifications():
    shards = [_shard(60, i) for i in range(4)]
    budget = build_budget_receipt(shards)
    validate_budget_receipt(budget)
    assert budget["canonical_indices"] == [0, 1, 2, 3]
    assert len(budget["source_shard_receipt_sha256"]) == 4
    assert set(budget["representations"]) == set(REPRESENTATION_VIEWS)
    payload = canonical_receipt_bytes(budget)
    assert payload == canonical_receipt_bytes(json.loads(payload))

    with pytest.raises(ValueError, match="four"):
        build_budget_receipt(shards[:3])


def test_cross_builder_uses_string_budget_keys_and_is_canonical():
    train60 = build_budget_receipt([_shard(60, i) for i in range(4)])
    train120 = build_budget_receipt([_shard(120, i) for i in range(4)])
    cross = build_cross_receipt(train60, train120)
    validate_cross_receipt(cross)

    assert set(cross["train_budget_classifications"]) == {"60", "120"}
    assert set(cross["source_budget_receipt_sha256"]) == {"60", "120"}
    assert cross["decision"] == "FRONTIER_SIGNAL_IDENTIFIED"
    assert cross["authorized_representation_view"] == "EXECUTION_FRONTIER_STATE"
    assert cross["mechanism_successor_authorized"] is False
    payload = canonical_receipt_bytes(cross)
    assert payload == canonical_receipt_bytes(json.loads(payload))
