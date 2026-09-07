import json
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.experiments.exp297_registry import (
    extend_neural_arm_registry_with_exp297,
    validate_exp297_neural_arm_registry,
)
from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _execution():
    return run_exp297_paired_development(
        root_seed="exp297-registry-test",
        eval_replicates=1,
        eval_start_replicate=500,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )


def _registry(execution=None):
    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    model_audit = audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))
    base = build_neural_arm_registry(
        protocol=protocol,
        protocol_digest="p" * 64,
        model_audit=model_audit,
    )
    pair = execution["resource_match"]["pair_audit"] if execution is not None else None
    return extend_neural_arm_registry_with_exp297(
        base_registry=base,
        protocol=protocol,
        exp297_pair_audit=pair,
        exp297_execution_artifact=execution,
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("registry_digest", None)
    payload["registry_digest"] = canonical_sha256(clean)


def test_exp297_registry_admits_development_pair_without_opening_match_court():
    execution = _execution()
    registry = _registry(execution)
    assert validate_exp297_neural_arm_registry(registry) == []

    item = registry["experiments"]["EXP-297"]
    assert item["protocol_arm_ids"] == ["compile_only", "fidelity_court"]
    assert item["development_match_status"] == "PAIRED_FIDELITY_COURT_DEV_READY"
    assert item["match_court"] == "BLOCKED"
    assert item["evidence_level"] == "EV-E2"
    assert item["decision"] == "UNVERIFIED"
    assert item["paired_execution_evidence"]["semantic_authority_promoted"] is False
    assert item["paired_execution_evidence"]["confirmatory_data_consumed"] is False


def test_exp297_registry_rejects_rehashed_execution_and_promotion_tamper():
    execution = _execution()
    registry = _registry(execution)

    changed = deepcopy(registry)
    changed["experiments"]["EXP-297"]["match_court"] = "CONFIRMATORY_READY"
    _rehash(changed)
    assert validate_exp297_neural_arm_registry(changed)

    changed = deepcopy(registry)
    changed["experiments"]["EXP-297"]["paired_execution_evidence"]["semantic_authority_promoted"] = True
    _rehash(changed)
    assert validate_exp297_neural_arm_registry(changed)

    changed = deepcopy(registry)
    changed["experiments"]["EXP-297"]["paired_execution_evidence"]["execution_digest"] = "0" * 64
    _rehash(changed)
    assert validate_exp297_neural_arm_registry(changed)
