from __future__ import annotations

import copy
import hashlib
import json

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9 import (
    FROZEN_PROTOCOL_DIGEST,
    PRIMARY_FAMILY,
    SCHEMA_SHARD,
    STUDENT_GEOMETRY,
    STUDENT_OPTIMIZER,
    canonical_root,
    decision_root,
    fit_root,
)
from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9_receipts import (
    build_budget_receipt,
    build_cross_receipt,
    canonical_receipt_bytes,
    receipt_sha256,
    validate_budget_receipt,
    validate_cross_receipt,
    validate_shard_receipt,
)

HEAD = "a" * 40
EXECUTED = "b" * 40
CODE = "c" * 64


def _root_metrics() -> dict[str, object]:
    return {
        "episodes": 21312,
        "routed_episodes": 100,
        "raw_rescues": 120,
        "raw_harms": 80,
        "selected_rescues": 20,
        "selected_harms": 2,
        "stop_solutions": 1000,
        "branch_solutions": 900,
        "policy_solutions": 1100,
        "stop_total_accounted_flops": 2_300_000_000,
        "branch_total_accounted_flops": 4_700_000_000,
        "policy_total_accounted_flops": 2_400_000_000,
        "route_fraction": 100 / 21312,
        "raw_rescue_prevalence": 120 / 21312,
        "selected_rescue_prevalence": 0.2,
        "stop_baseline_utility": 1000 / 2_300_000_000,
        "branch_baseline_utility": 900 / 4_700_000_000,
        "policy_utility": 1100 / 2_400_000_000,
        "evidence_boundary_closed": True,
    }


def _shard(*, train: int = 60, canonical: int = 0) -> dict[str, object]:
    return {
        "schema": SCHEMA_SHARD,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": FROZEN_PROTOCOL_DIGEST,
        "code_digest": CODE,
        "scientific_branch_head": HEAD,
        "executed_commit": EXECUTED,
        "train_replicates": train,
        "canonical_index": canonical,
        "canonical_root": canonical_root(train, canonical),
        "fit_roots": [fit_root(train, canonical, 0), fit_root(train, canonical, 1)],
        "decision_roots": [decision_root(train, canonical, 0), decision_root(train, canonical, 1)],
        "rng_streams": {"canonical_training": "augmentation", "fit": "augmentation", "decision": "augmentation"},
        "student_family": PRIMARY_FAMILY,
        "student_geometry": dict(STUDENT_GEOMETRY),
        "student_optimizer": dict(STUDENT_OPTIMIZER),
        "fit_replicates_per_root": 1332,
        "decision_replicates_per_root": 1332,
        "root_metrics": _root_metrics(),
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }


def _budget(train: int = 60) -> dict[str, object]:
    return build_budget_receipt([_shard(train=train, canonical=index) for index in range(4)])


def test_v9_receipt_bytes_and_sha256_are_deterministic() -> None:
    receipt = _shard()
    first = canonical_receipt_bytes(receipt)
    second = canonical_receipt_bytes(dict(reversed(list(receipt.items()))))
    assert first == second
    assert first.endswith(b"\n")
    assert receipt_sha256(receipt) == hashlib.sha256(first).hexdigest()


def test_v9_cross_receipt_is_canonical_after_json_round_trip() -> None:
    built = build_cross_receipt(_budget(60), _budget(120))
    payload = canonical_receipt_bytes(built)
    reparsed = json.loads(payload.decode("utf-8"))
    assert payload == canonical_receipt_bytes(reparsed)


def test_v9_shard_validator_requires_absolute_frozen_identity_and_boundaries() -> None:
    assert validate_shard_receipt(_shard()) == []

    wrong_protocol = _shard()
    wrong_protocol["protocol_digest"] = "d" * 64
    assert validate_shard_receipt(wrong_protocol)

    wrong_root = _shard()
    wrong_root["canonical_root"] = canonical_root(120, 0)
    assert validate_shard_receipt(wrong_root)

    leaked_rng = _shard()
    leaked_rng["rng_streams"] = {"canonical_training": "augmentation", "fit": "augmentation", "decision": "evaluation"}
    assert validate_shard_receipt(leaked_rng)

    opened_boundary = _shard()
    opened_boundary["confirmatory_data_consumed"] = True
    assert validate_shard_receipt(opened_boundary)


def test_v9_budget_validator_rejects_consistent_but_wrong_or_duplicate_shards() -> None:
    good = _budget(60)
    assert validate_budget_receipt(good) == []

    duplicate = copy.deepcopy(good)
    duplicate["root_receipts"][3] = copy.deepcopy(duplicate["root_receipts"][0])
    assert validate_budget_receipt(duplicate)

    wrong_absolute_protocol = copy.deepcopy(good)
    wrong_absolute_protocol["protocol_digest"] = "d" * 64
    for shard in wrong_absolute_protocol["root_receipts"]:
        shard["protocol_digest"] = "d" * 64
    assert validate_budget_receipt(wrong_absolute_protocol)

    mixed_head = copy.deepcopy(good)
    mixed_head["root_receipts"][2]["scientific_branch_head"] = "e" * 40
    assert validate_budget_receipt(mixed_head)


def test_v9_budget_builder_derives_identity_digests_and_classification() -> None:
    shards = [_shard(train=60, canonical=index) for index in range(4)]
    built = build_budget_receipt(shards)
    assert validate_budget_receipt(built) == []
    assert built["train_replicates"] == 60
    assert built["classification"] == "QUARTET_POLICY_RECURRENTLY_ECONOMIC"
    assert built["source_shard_digests"] == [receipt_sha256(shard) for shard in shards]
    assert built["root_metrics"] == [shard["root_metrics"] for shard in shards]

    with pytest.raises(ValueError):
        build_budget_receipt([shards[0], shards[1], shards[2], copy.deepcopy(shards[0])])


def test_v9_cross_validator_requires_exact_60_120_pair_and_closed_boundaries() -> None:
    train60 = _budget(60)
    train120 = _budget(120)
    cross = build_cross_receipt(train60, train120)
    assert validate_cross_receipt(cross) == []

    duplicated_budget = copy.deepcopy(cross)
    duplicated_budget["budget_receipts"] = {"60": train60, "120": copy.deepcopy(train60)}
    assert validate_cross_receipt(duplicated_budget)

    promoted = copy.deepcopy(cross)
    promoted["promotion_claimed"] = True
    assert validate_cross_receipt(promoted)


def test_v9_cross_builder_derives_decision_and_never_opens_scientific_boundaries() -> None:
    train60 = build_budget_receipt([_shard(train=60, canonical=index) for index in range(4)])
    train120 = build_budget_receipt([_shard(train=120, canonical=index) for index in range(4)])
    built = build_cross_receipt(train60, train120)
    assert validate_cross_receipt(built) == []
    assert built["decision"] == "QUARTET_MECHANISM_COURT_PASSED"
    assert built["authorization_scope"] == "IMPLEMENT_QUARTET_SUCCESSOR_DEVELOPMENT_ONLY"
    assert built["mechanism_successor_authorized"] is True
    assert built["scientific_evidence_eligible"] is False
    assert built["fresh_evaluation_lineage_may_be_reserved"] is False
    assert built["fresh_evaluation_lineage_consumed"] is False
    assert built["confirmatory_data_consumed"] is False
    assert built["challenge_materialized"] is False
    assert built["promotion_claimed"] is False
