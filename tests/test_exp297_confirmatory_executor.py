from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_beacon import (
    build_test_beacon_receipt,
    derive_exp297_challenge_seed,
)
from nolane_ai.experiments.exp297_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    authorize_exp297_confirmatory_execution,
)
from nolane_ai.experiments.exp297_confirmatory_executor import (
    execute_exp297_confirmatory_challenge,
    validate_exp297_confirmatory_raw,
)
from nolane_ai.experiments.exp297_confirmatory_prep import build_exp297_confirmatory_prep
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.experiments.exp297_reconstruction_court import (
    authorize_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "f" * 40
FREEZE_TIME = "2026-09-08T05:00:00Z"
BEACON_TIME = "2026-09-08T05:00:01Z"


@pytest.fixture(scope="module")
def reconstruction():
    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    experiment = next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-297")
    execution = run_exp297_paired_development(
        root_seed="exp297-executor-test",
        eval_replicates=32,
        eval_start_replicate=7000,
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
        analysis_code_digest="a" * 64,
    )
    authorization = authorize_exp297_confirmatory_execution(
        prep_artifact=prep,
        challenge_contract_digest=challenge_contract_digest(),
        evaluator_code_digest="e" * 64,
        execution_code_digest="x" * 64,
    )
    return authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_code_digest="r" * 64,
    )


@pytest.fixture(scope="module")
def beacon():
    return build_test_beacon_receipt(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp297-test-beacon-1",
        published_at_utc=BEACON_TIME,
        entropy_hex="ab" * 32,
        evidence_reference="unit-test-only",
    )


@pytest.fixture(scope="module")
def raw(reconstruction, beacon):
    return execute_exp297_confirmatory_challenge(
        reconstruction_authorization=reconstruction,
        beacon_receipt=beacon,
        freeze_commit_sha=FREEZE_SHA,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        executor_code_digest="q" * 64,
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    payload["artifact_digest"] = canonical_sha256(clean)


def test_exp297_test_only_executor_materializes_only_beacon_derived_challenge(raw, reconstruction, beacon):
    assert raw["schema"] == "NLM-EXP-297-CONFIRMATORY-CHALLENGE-RAW-V1"
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["test_only"] is True
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["synthetic_challenge_data_consumed"] is True
    assert raw["seed_materialization_status"] == "TEST_ONLY_EXECUTED"
    assert raw["challenge_materialized"] is True
    assert raw["decision_rule_executed"] is False
    assert raw["semantic_authority_promoted"] is False
    assert raw["confirmatory_n"] == reconstruction["confirmatory_n"]
    assert raw["reserved_replicate_ids"] == reconstruction["reserved_replicate_ids"]
    rows = raw["raw_candidates"]
    assert len(rows) == raw["confirmatory_n"] * 16
    first_replicate = raw["reserved_replicate_ids"][0]
    expected_seed = derive_exp297_challenge_seed(
        protocol_digest=reconstruction["lineage"]["protocol_digest"],
        freeze_commit_sha=FREEZE_SHA,
        freeze_commit_timestamp_utc=FREEZE_TIME,
        beacon_receipt=beacon,
        stream="challenge",
        replicate=first_replicate,
    )
    first_rows = [row for row in rows if row["replicate"] == first_replicate]
    assert len(first_rows) == 16
    assert {row["challenge_seed"] for row in first_rows} == {expected_seed}
    assert [row["candidate_index"] for row in first_rows] == list(range(16))
    assert validate_exp297_confirmatory_raw(raw) == []


def test_exp297_executor_rows_preserve_evaluator_boundary_and_exact_cost_receipts(raw):
    for row in raw["raw_candidates"]:
        assert row["arm_input_receipt"]["evaluator_truth_in_causal_path"] is False
        assert row["arm_input_receipt"]["trap_family_in_causal_path"] is False
        assert row["arms"]["compile_only"]["semantic_verification_operations"] == 0
        assert row["arms"]["fidelity_court"]["semantic_verification_operations"] == row["court_receipt"]["semantic_verification_operations"]
        assert row["arms"]["compile_only"]["hardware_profiler_flops_claimed"] is False
        assert row["arms"]["fidelity_court"]["hardware_profiler_flops_claimed"] is False
        if row["court_receipt"]["decision"] != "court_accept":
            assert row["arms"]["fidelity_court"]["authority_granted"] is False


def test_exp297_executor_rejects_early_beacon(reconstruction):
    early = build_test_beacon_receipt(
        source="TEST-ONLY deterministic beacon",
        beacon_id="exp297-test-beacon-early",
        published_at_utc=FREEZE_TIME,
        entropy_hex="cd" * 32,
        evidence_reference="unit-test-only",
    )
    with pytest.raises(ValueError, match="strictly after"):
        execute_exp297_confirmatory_challenge(
            reconstruction_authorization=reconstruction,
            beacon_receipt=early,
            freeze_commit_sha=FREEZE_SHA,
            freeze_commit_timestamp_utc=FREEZE_TIME,
            executor_code_digest="q" * 64,
        )


def test_exp297_executor_rejects_rehashed_beacon_seed_candidate_truth_witness_cost_and_scientific_tampering(raw):
    wrong_index = next(
        index
        for index, row in enumerate(raw["raw_candidates"])
        if row["is_faithful"] is False and row["court_receipt"].get("witness") is not None
    )
    mutations = []

    changed = deepcopy(raw)
    changed["beacon_receipt"]["entropy_hex"] = "ef" * 32
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][0]["challenge_seed"] += 1
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][0], changed["raw_candidates"][1] = changed["raw_candidates"][1], changed["raw_candidates"][0]
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"].pop()
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][0]["candidate_digest"] = "0" * 64
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][0]["is_faithful"] = not changed["raw_candidates"][0]["is_faithful"]
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][0]["stratum"] = "forged_stratum"
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][wrong_index]["court_receipt"]["witness"]["assignment"][next(iter(changed["raw_candidates"][wrong_index]["court_receipt"]["witness"]["assignment"]))] = 999
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][wrong_index]["arms"]["fidelity_court"]["authority_granted"] = True
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][0]["arms"]["compile_only"]["neural_accounted_flops"] -= 1
    changed["raw_candidates"][0]["arms"]["compile_only"]["total_accounted_cost_proxy"] -= 1
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][wrong_index]["arms"]["fidelity_court"]["semantic_verification_operations"] = 0
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["raw_candidates"][wrong_index]["court_receipt"]["decision"] = "court_accept"
    changed["raw_candidates"][wrong_index]["arms"]["fidelity_court"]["authority_granted"] = True
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["scientific_evidence_eligible"] = True
    mutations.append(changed)

    changed = deepcopy(raw)
    changed["semantic_authority_promoted"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash(changed)
        assert validate_exp297_confirmatory_raw(changed)
