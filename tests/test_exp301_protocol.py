from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nolane_ai.protocol.v017 import (
    EXP301_ARM_IDS,
    EXP301_CHALLENGE_LOOPS,
    EXP301_EXPECTED_DIGEST,
    EXP301_ROOTS,
    EXP301_SUPERSEDED_V1_DIGEST,
    EXP301_TRAINING_LOOPS,
    canonical_registration_payload,
    load_exp301_registration,
    registration_digest,
    validate_exp301_registration,
)


PROTOCOL_PATH = Path("protocols/v017/exp301_preregistration_v2.json")
V1_PATH = Path("protocols/v017/exp301_preregistration_v1.json")
EXPECTED_V2_DIGEST = "1660a728990290f1c605941d531be1d52fe82591c619a327d104e4b87c1e6bad"


def test_exp301_semantic_digest_matches_current_predata_amendment() -> None:
    payload = load_exp301_registration(PROTOCOL_PATH)

    assert EXP301_EXPECTED_DIGEST == EXPECTED_V2_DIGEST
    assert registration_digest(payload) == EXPECTED_V2_DIGEST
    assert payload["registration_digest"] == f"sha256:{EXPECTED_V2_DIGEST}"
    validate_exp301_registration(payload)


def test_exp301_v1_is_preserved_but_explicitly_superseded_before_data() -> None:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v2 = load_exp301_registration(PROTOCOL_PATH)
    amendment = v2["protocol_amendment"]

    assert registration_digest(v1) == EXP301_SUPERSEDED_V1_DIGEST
    assert amendment["supersedes_registration_digest"] == f"sha256:{EXP301_SUPERSEDED_V1_DIGEST}"
    assert amendment["amendment_kind"] == "PRE_DATA_FAIRNESS_CORRECTION"
    assert amendment["scientific_outcome_consumed"] is False
    assert amendment["challenge_materialized"] is False
    assert amendment["hypothesis_changed"] is False


def test_exp301_digest_is_key_order_and_whitespace_invariant() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    semantically_identical = json.loads(json.dumps(payload, indent=None, sort_keys=False))

    assert canonical_registration_payload(payload) == canonical_registration_payload(semantically_identical)
    assert registration_digest(payload) == registration_digest(semantically_identical)


def test_exp301_scientific_mutation_breaks_frozen_digest() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutated["training"]["roots"] = [0, 1, 2, 99]

    assert registration_digest(mutated) != EXP301_EXPECTED_DIGEST
    with pytest.raises(ValueError, match="registration digest"):
        validate_exp301_registration(mutated)


def test_exp301_frozen_authority_fields_are_exact() -> None:
    payload = load_exp301_registration(PROTOCOL_PATH)

    assert EXP301_ARM_IDS == ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
    assert EXP301_ROOTS == (0, 1, 2, 3)
    assert EXP301_TRAINING_LOOPS == (1, 2, 4, 8)
    assert EXP301_CHALLENGE_LOOPS == (1, 2, 4, 8, 12, 16)
    assert tuple(arm["id"] for arm in payload["arms"]) == EXP301_ARM_IDS
    assert tuple(payload["training"]["roots"]) == EXP301_ROOTS
    assert tuple(payload["training"]["loop_training_distribution"]) == EXP301_TRAINING_LOOPS
    assert tuple(payload["training"]["challenge_loop_budgets"]) == EXP301_CHALLENGE_LOOPS


def test_exp301_v2_freezes_stateless_fixed_restart_compute_rival() -> None:
    payload = load_exp301_registration(PROTOCOL_PATH)
    fixed = payload["arms"][0]
    compute = payload["compute_matching"]

    assert fixed["id"] == "A_FIXED"
    assert "stateless repeated restarts" in fixed["role"]
    assert "no latent state carries across restarts" in fixed["constraints"]
    assert tuple(compute["effort_multipliers"]) == EXP301_CHALLENGE_LOOPS
    assert compute["max_relative_flop_mismatch"] == pytest.approx(0.002)
    assert "no cross-restart latent state" in compute["a_fixed_policy"]
    assert compute["unmatched_budget_policy"].startswith("INVALID_COMPUTE_MATCH")


def test_exp301_loader_rejects_wrong_experiment_even_with_recomputed_self_digest(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["experiment_id"] = "EXP-999"
    payload["registration_digest"] = f"sha256:{registration_digest(payload)}"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment_id"):
        load_exp301_registration(path)
