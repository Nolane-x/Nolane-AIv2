from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_confirmatory_prep import (
    build_exp297_confirmatory_prep,
    validate_exp297_confirmatory_prep,
)
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))


def _exp297():
    return next(item for item in _protocol()["experiments"] if item["experiment_id"] == "EXP-297")


def _pilot():
    return run_exp297_paired_development(
        root_seed="exp297-confirmatory-prep-test",
        eval_replicates=32,
        eval_start_replicate=2000,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )


def _registry(execution):
    return {
        "registry_digest": canonical_sha256({"execution": execution["artifact_digest"]}),
        "experiments": {
            "EXP-297": {
                "evidence_level": "EV-E2",
                "decision": "UNVERIFIED",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {
                    "execution_digest": execution["artifact_digest"],
                },
            }
        },
    }


def _build():
    execution = _pilot()
    return build_exp297_confirmatory_prep(
        experiment=_exp297(),
        execution_artifact=execution,
        arm_registry=_registry(execution),
        analysis_code_digest="a" * 64,
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("prep_digest", None)
    payload["prep_digest"] = canonical_sha256(clean)


def test_exp297_confirmatory_prep_freezes_paired_n_without_consuming_confirmatory_data():
    prep = _build()
    assert prep["schema"] == "NLM-EXP-297-CONFIRMATORY-GATE-A-PREP-V1"
    assert prep["evidence_level"] == "EV-E2"
    assert prep["decision"] == "UNVERIFIED"
    assert prep["confirmatory_data_consumed"] is False
    assert prep["seed_materialization_status"] == "NOT_EXECUTED"
    assert prep["challenge_materialized"] is False
    assert prep["decision_rule_executed"] is False
    assert prep["semantic_authority_promoted"] is False
    assert prep["pilot_summary"]["n"] == 32
    assert len(prep["pilot_summary"]["paired_balanced_accuracy_effects"]) == 32
    assert prep["frozen_analysis"]["mesi_absolute_gain"] == 0.10
    assert prep["frozen_analysis"]["familywise_alpha"] == 0.05
    assert prep["frozen_analysis"]["alpha_per_endpoint"] == pytest.approx(0.05 / 3.0)
    assert prep["sample_size_freeze"]["min_n"] == 32
    assert prep["sample_size_freeze"]["max_n"] == 128
    assert prep["sample_size_freeze"]["paired"] is True
    assert prep["sample_size_freeze"]["pilot_reuse_as_confirmatory"] is False
    if prep["status"] == "CONFIRMATORY_GATE_A_PREPARED":
        n = prep["sample_size_freeze"]["confirmatory_n"]
        reserved = prep["confirmatory_lineage"]["reserved_replicate_ids"]
        development = set(prep["pilot_summary"]["replicate_ids"])
        assert 32 <= n <= 128
        assert len(reserved) == n
        assert reserved == list(range(reserved[0], reserved[0] + n))
        assert not development.intersection(reserved)
    assert validate_exp297_confirmatory_prep(prep) == []


def test_exp297_confirmatory_prep_reconstructs_each_pilot_effect_from_raw_rows():
    execution = _pilot()
    prep = build_exp297_confirmatory_prep(
        experiment=_exp297(),
        execution_artifact=execution,
        arm_registry=_registry(execution),
        analysis_code_digest="a" * 64,
    )
    effects = prep["pilot_summary"]["paired_balanced_accuracy_effects"]
    assert len(set(effects)) == 1
    assert effects[0] == pytest.approx(0.5)
    assert prep["pilot_summary"]["paired_sd"] == pytest.approx(0.0)
    assert prep["sample_size_freeze"]["unclamped_required_n"] == 1
    assert prep["sample_size_freeze"]["confirmatory_n"] == 32


def test_exp297_confirmatory_prep_requires_at_least_32_development_replicates():
    execution = run_exp297_paired_development(
        root_seed="exp297-confirmatory-prep-short",
        eval_replicates=31,
        eval_start_replicate=5000,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    with pytest.raises(ValueError, match="32"):
        build_exp297_confirmatory_prep(
            experiment=_exp297(),
            execution_artifact=execution,
            arm_registry=_registry(execution),
            analysis_code_digest="a" * 64,
        )


def test_exp297_confirmatory_prep_rejects_rehashed_freeze_and_lineage_tampering():
    prep = _build()
    mutations = []

    changed = deepcopy(prep)
    changed["pilot_summary"]["paired_sd"] = 0.75
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["frozen_analysis"]["mesi_absolute_gain"] = 0.01
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["frozen_analysis"]["alpha_per_endpoint"] = 0.05
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["sample_size_freeze"]["unclamped_required_n"] += 1
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["sample_size_freeze"]["confirmatory_n"] = 33
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["confirmatory_lineage"]["reserved_replicate_ids"][0] = changed["pilot_summary"]["replicate_ids"][0]
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["confirmatory_data_consumed"] = True
    mutations.append(changed)

    changed = deepcopy(prep)
    changed["semantic_authority_promoted"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash(changed)
        assert validate_exp297_confirmatory_prep(changed)
