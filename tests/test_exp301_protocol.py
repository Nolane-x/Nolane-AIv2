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
    EXP301_TRAINING_LOOPS,
    canonical_registration_payload,
    load_exp301_registration,
    registration_digest,
    validate_exp301_registration,
)


PROTOCOL_PATH = Path("protocols/v017/exp301_preregistration_v1.json")


def test_exp301_semantic_digest_matches_frozen_registration() -> None:
    payload = load_exp301_registration(PROTOCOL_PATH)

    assert registration_digest(payload) == EXP301_EXPECTED_DIGEST
    assert payload["registration_digest"] == f"sha256:{EXP301_EXPECTED_DIGEST}"
    validate_exp301_registration(payload)


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


def test_exp301_loader_rejects_wrong_experiment_even_with_recomputed_self_digest(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["experiment_id"] = "EXP-999"
    payload["registration_digest"] = f"sha256:{registration_digest(payload)}"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment_id"):
        load_exp301_registration(path)
