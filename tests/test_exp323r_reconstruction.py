from __future__ import annotations

from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp323_contract import AUTHORIZATION_FLAGS
from nolane_ai.experiments.exp323_evidence import expected_reconstruction_payload
from nolane_ai.experiments.exp323r_reconstruction import (
    RECEIPT_SCHEMA,
    canonical_json_bytes,
    receipt_digest,
)


def test_reconstruction_receipt_digest_is_tamper_evident() -> None:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "attempt_index": 0,
        "parent_exp322_run_id": 35339004168,
        "parent_exp322_evidence_digest": "1" * 64,
        "parent_exp322_execution_digest": "2" * 64,
        "reconstruction": expected_reconstruction_payload(),
        "checkpoint_sha256": "3" * 64,
        "post_2048_optimizer_steps": 0,
        **AUTHORIZATION_FLAGS,
    }
    digest = receipt_digest(payload)
    sealed = dict(payload, receipt_digest=digest)
    assert receipt_digest(sealed) == digest

    forged = deepcopy(sealed)
    forged["post_2048_optimizer_steps"] = 1
    assert receipt_digest(forged) != digest


def test_reconstruction_authority_keeps_every_authorization_false() -> None:
    assert AUTHORIZATION_FLAGS
    assert not any(AUTHORIZATION_FLAGS.values())


def test_materialization_workflow_is_bounded_and_has_no_scientific_arm() -> None:
    text = open(
        ".github/workflows/exp323r-reconstruction-materialization.yml",
        encoding="utf-8",
    ).read()
    assert "attempt: [0, 1, 2, 3]" in text
    assert "--arm" not in text
    assert "--learning-rate" not in text
    assert "post-2048" not in text.lower()
    assert "require-candidate" in text
