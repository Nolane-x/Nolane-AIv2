from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_beacon import build_test_beacon_receipt
from nolane_ai.experiments.exp297_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp297_confirmatory_analysis import (
    bootstrap_exp297_paired_gain,
    build_exp297_confirmatory_analysis,
    decide_exp297_confirmatory_outcome,
    validate_exp297_confirmatory_analysis,
    wilson_upper_bound,
)
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    authorize_exp297_confirmatory_execution,
)
from nolane_ai.experiments.exp297_confirmatory_executor import (
    execute_exp297_confirmatory_challenge,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    ALPHA_PER_ENDPOINT,
    BOOTSTRAP_SAMPLES,
    build_exp297_confirmatory_prep,
)
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.experiments.exp297_reconstruction_court import (
    authorize_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "f" * 40
FREEZE_TIME = "2026-09-08T05:00:00Z"
BEACON_TIME = "2026-09-08T05:00:01Z"
ANALYSIS_CODE_DIGEST = "a" * 64


@pytest.fixture(scope="module")
def raw():
    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    experiment = next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-297")
    execution = run_exp297_paired_development(
        root_seed="exp297-analysis-test",
        eval_replicates=32,
        eval_start_replicate=9000,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    registry = {
        "registry_digest": canonical_sha256({"execution": execution["artifact_digest"]}),
        "experiments": {
            "EXP-297": {
                "evidence_level": "EV-E2",
                "decision": "UNVERIFIED",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {"execution_digest": execution["artifact_digest"]},
            }
        },
    }
    prep = build_exp297_confirmatory_prep(
        experiment=experiment,
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=ANALYSIS_CODE_DIGEST,
    )
    authorization = authorize_exp297_confirmatory_execution(
        prep_artifact=prep,
        challenge_contract_digest=challenge_contract_digest(),
        evaluator_code_digest="e" * 64,
        execution_code_digest="x" * 64,
    )
    reconstruction = authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_code_digest="r" * 64,
    )
    beacon = build_test_beacon_receipt(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp297-analysis-test-beacon",
        published_at_utc=BEACON_TIME,
        entropy_hex="ab" * 32,
        evidence_reference="unit-test-only",
    )
    return execute_exp297_confirmatory_challenge(
        reconstruction_authorization=reconstruction,
        beacon_receipt=beacon,
        freeze_commit_sha=FREEZE_SHA,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        executor_code_digest="q" * 64,
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("analysis_digest", None)
    payload["analysis_digest"] = canonical_sha256(clean)


def test_exp297_bootstrap_and_wilson_statistics_are_deterministic_and_frozen():
    values = [0.5] * 32
    first = bootstrap_exp297_paired_gain(
        values,
        seed_material="exp297-bootstrap-test",
        samples=BOOTSTRAP_SAMPLES,
        alpha=ALPHA_PER_ENDPOINT,
    )
    second = bootstrap_exp297_paired_gain(
        values,
        seed_material="exp297-bootstrap-test",
        samples=BOOTSTRAP_SAMPLES,
        alpha=ALPHA_PER_ENDPOINT,
    )
    assert first == second
    assert first["observed_mean"] == 0.5
    assert first["one_sided_lower"] == 0.5
    assert first["one_sided_upper"] == 0.5
    assert first["samples"] == 10_000
    assert first["alpha"] == ALPHA_PER_ENDPOINT

    wrong_upper = wilson_upper_bound(0, 256, alpha=ALPHA_PER_ENDPOINT)
    faithful_upper = wilson_upper_bound(0, 256, alpha=ALPHA_PER_ENDPOINT)
    assert 0.0 < wrong_upper < 0.05
    assert 0.0 < faithful_upper < 0.10
    with pytest.raises(ValueError):
        wilson_upper_bound(2, 1, alpha=ALPHA_PER_ENDPOINT)


def test_exp297_frozen_three_way_decision_rule():
    assert decide_exp297_confirmatory_outcome(
        primary_lower=0.11,
        primary_upper=0.20,
        wrong_authority_upper=0.04,
        faithful_rejection_upper=0.09,
    ) == "PROMOTE_TO_NEXT_STAGE"
    assert decide_exp297_confirmatory_outcome(
        primary_lower=0.20,
        primary_upper=0.30,
        wrong_authority_upper=0.051,
        faithful_rejection_upper=0.01,
    ) == "KILL_SUBSYSTEM"
    assert decide_exp297_confirmatory_outcome(
        primary_lower=0.01,
        primary_upper=0.09,
        wrong_authority_upper=0.01,
        faithful_rejection_upper=0.01,
    ) == "KILL_SUBSYSTEM"
    assert decide_exp297_confirmatory_outcome(
        primary_lower=0.08,
        primary_upper=0.15,
        wrong_authority_upper=0.01,
        faithful_rejection_upper=0.01,
    ) == "HOLD_UNSTABLE"


def test_exp297_test_only_analysis_cannot_promote_scientific_evidence(raw):
    analysis = build_exp297_confirmatory_analysis(
        raw_artifact=raw,
        analysis_code_digest=ANALYSIS_CODE_DIGEST,
    )
    assert analysis["schema"] == "NLM-EXP-297-CONFIRMATORY-ANALYSIS-V1"
    assert analysis["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["test_only"] is True
    assert analysis["scientific_evidence_eligible"] is False
    assert analysis["confirmatory_data_consumed"] is False
    assert analysis["synthetic_challenge_data_consumed"] is True
    assert analysis["decision_rule_executed"] is False
    assert analysis["test_only_decision_rule_executed"] is True
    assert analysis["semantic_authority_promoted"] is False
    assert analysis["test_only_would_be_decision"] == "PROMOTE_TO_NEXT_STAGE"
    assert analysis["primary_effect"]["one_sided_lower"] >= 0.10
    assert analysis["protected_endpoints"]["wrong_authority"]["pass"] is True
    assert analysis["protected_endpoints"]["faithful_rejection"]["pass"] is True
    assert len(analysis["raw_per_replicate_metrics"]) == raw["confirmatory_n"]
    assert validate_exp297_confirmatory_analysis(analysis) == []


def test_exp297_analysis_rejects_rehashed_statistics_safety_contract_and_promotion_tampering(raw):
    analysis = build_exp297_confirmatory_analysis(
        raw_artifact=raw,
        analysis_code_digest=ANALYSIS_CODE_DIGEST,
    )
    mutations = []

    changed = deepcopy(analysis)
    changed["primary_effect"]["one_sided_lower"] = 0.99
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["protected_endpoints"]["wrong_authority"]["events"] += 1
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["protected_endpoints"]["faithful_rejection"]["total"] -= 1
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["frozen_contract"]["alpha_per_endpoint"] = 0.05
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["test_only_would_be_decision"] = "KILL_SUBSYSTEM"
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["scientific_evidence_eligible"] = True
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["decision"] = "PROMOTE_TO_NEXT_STAGE"
    mutations.append(changed)

    changed = deepcopy(analysis)
    changed["semantic_authority_promoted"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash(changed)
        assert validate_exp297_confirmatory_analysis(changed)
