from copy import deepcopy
from hashlib import sha256
import inspect

from nolane_ai.experiments.exp297_beacon import (
    build_test_beacon_receipt,
    derive_exp297_challenge_seed,
    validate_exp297_beacon_receipt,
)
from nolane_ai.protocol.evidence import canonical_sha256


def _receipt(*, published_at="2026-09-08T06:10:00Z"):
    return build_test_beacon_receipt(
        source="TEST-ONLY",
        beacon_id="synthetic-beacon-1",
        published_at_utc=published_at,
        entropy_hex="ab" * 32,
        evidence_reference="test://exp297/synthetic-beacon-1",
    )


def _rehash(receipt):
    clean = deepcopy(receipt)
    clean.pop("receipt_digest", None)
    receipt["receipt_digest"] = canonical_sha256(clean)


def test_exp297_test_beacon_receipt_is_canonical_and_explicitly_non_scientific():
    receipt = _receipt()
    assert receipt["schema"] == "NLM-EXP-297-PUBLIC-BEACON-RECEIPT-V1"
    assert receipt["test_only"] is True
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["receipt_digest"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    assert validate_exp297_beacon_receipt(
        receipt,
        freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
    ) == []


def test_exp297_beacon_must_be_strictly_later_than_gate_a_freeze():
    equal = _receipt(published_at="2026-09-08T06:00:00Z")
    early = _receipt(published_at="2026-09-08T05:59:59Z")
    assert validate_exp297_beacon_receipt(
        equal,
        freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
    )
    assert validate_exp297_beacon_receipt(
        early,
        freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
    )


def test_exp297_beacon_validator_rejects_rehashed_timestamp_entropy_and_identity_tampering():
    receipt = _receipt()
    mutations = []

    changed = deepcopy(receipt)
    changed["published_at_utc"] = "not-a-timestamp"
    mutations.append(changed)

    changed = deepcopy(receipt)
    changed["entropy_hex"] = "xyz"
    mutations.append(changed)

    changed = deepcopy(receipt)
    changed["entropy_hex"] = "aa"
    mutations.append(changed)

    changed = deepcopy(receipt)
    changed["beacon_id"] = ""
    mutations.append(changed)

    changed = deepcopy(receipt)
    changed["scientific_evidence_eligible"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash(changed)
        assert validate_exp297_beacon_receipt(
            changed,
            freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
        )


def test_exp297_challenge_seed_derivation_is_exact_deterministic_and_binds_lineage():
    receipt = _receipt()
    protocol_digest = "p" * 64
    freeze_sha = "f" * 40
    expected_material = (
        f"{protocol_digest}|{freeze_sha}|{receipt['receipt_digest']}|"
        "EXP-297|challenge|7001"
    )
    expected = int.from_bytes(sha256(expected_material.encode("utf-8")).digest()[:8], "big")

    first = derive_exp297_challenge_seed(
        protocol_digest=protocol_digest,
        freeze_commit_sha=freeze_sha,
        freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=7001,
    )
    second = derive_exp297_challenge_seed(
        protocol_digest=protocol_digest,
        freeze_commit_sha=freeze_sha,
        freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=7001,
    )
    assert first == second == expected
    assert first != derive_exp297_challenge_seed(
        protocol_digest=protocol_digest,
        freeze_commit_sha=freeze_sha,
        freeze_commit_timestamp_utc="2026-09-08T06:00:00Z",
        beacon_receipt=receipt,
        stream="challenge",
        replicate=7002,
    )


def test_exp297_seed_api_exposes_no_operator_seed_override():
    parameters = inspect.signature(derive_exp297_challenge_seed).parameters
    assert "seed" not in parameters
    assert "challenge_seed" not in parameters
    assert set(parameters) == {
        "protocol_digest",
        "freeze_commit_sha",
        "freeze_commit_timestamp_utc",
        "beacon_receipt",
        "stream",
        "replicate",
    }
