from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


def _fixture(tmp_path):
    from tests.test_exp279_reconstruction_execution import _fixture as build_fixture

    return build_fixture(tmp_path)


def _real_classified_beacon(test_beacon):
    from nolane_ai.experiments.exp279_beacon import _receipt_digest

    beacon = deepcopy(test_beacon)
    beacon["test_only"] = False
    beacon["scientific_evidence_eligible"] = True
    beacon["authenticity_status"] = "EXTERNAL_EVIDENCE_RECORDED"
    beacon["receipt_digest"] = _receipt_digest(beacon)
    return beacon


def _execute(fixture, **overrides):
    from nolane_ai.experiments.exp279_confirmatory_ceremony import (
        execute_exp279_gate_b_ceremony,
    )

    reconstruction = fixture["reconstruction"]
    kwargs = {
        "seal": fixture["seal"],
        "reconstruction_authorization": reconstruction,
        "beacon_receipt": fixture["beacon"],
        "checkpoint_path": fixture["checkpoint_path"],
        "checkpoint_receipt": fixture["checkpoint_receipt"],
        "current_source_tree_digest": reconstruction["source_tree_digest"],
        "executor_code_digest": "3" * 64,
        "test_only": True,
        "arm_scientific_lane": False,
    }
    kwargs.update(overrides)
    return execute_exp279_gate_b_ceremony(**kwargs)


def test_exp279_test_only_ceremony_is_e2_unverified_unarmed_and_publishes_raw_first(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_ceremony import (
        validate_exp279_gate_b_ceremony,
    )

    fixture = _fixture(tmp_path)
    published = []
    ceremony = _execute(
        fixture,
        raw_publisher=lambda raw: published.append(deepcopy(raw)),
    )

    assert ceremony["schema"] == "NLM-EXP-279-CONFIRMATORY-CEREMONY-V1"
    assert ceremony["status"] == "TEST_ONLY_CEREMONY_COMPLETED"
    assert ceremony["test_only"] is True
    assert ceremony["scientific_lane_armed"] is False
    assert ceremony["raw"]["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert ceremony["raw"]["evidence_level"] == "EV-E2"
    assert ceremony["raw"]["decision"] == "UNVERIFIED"
    assert ceremony["analysis"]["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert ceremony["analysis"]["evidence_level"] == "EV-E2"
    assert ceremony["analysis"]["decision"] == "UNVERIFIED"
    assert ceremony["analysis"]["scientific_evidence_eligible"] is False
    assert ceremony["analysis"]["decision_rule_executed"] is False
    assert ceremony["analysis"]["test_only_would_be_decision"] in {
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "KILL_SUBSYSTEM",
    }
    assert len(published) == 1
    assert published[0]["artifact_digest"] == ceremony["raw"]["artifact_digest"]
    assert validate_exp279_gate_b_ceremony(
        ceremony,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


def test_exp279_real_beacon_is_refused_before_execution_unless_scientific_lane_is_armed(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    real_beacon = _real_classified_beacon(fixture["beacon"])
    published = []

    with pytest.raises(RuntimeError, match="scientific lane is not armed"):
        _execute(
            fixture,
            beacon_receipt=real_beacon,
            test_only=False,
            arm_scientific_lane=False,
            raw_publisher=lambda raw: published.append(deepcopy(raw)),
        )
    assert published == []


def test_exp279_test_only_beacon_cannot_cross_into_scientific_mode_before_execution(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    published = []
    with pytest.raises(RuntimeError, match="TEST-ONLY"):
        _execute(
            fixture,
            test_only=False,
            arm_scientific_lane=True,
            raw_publisher=lambda raw: published.append(deepcopy(raw)),
        )
    assert published == []


def test_exp279_ceremony_publishes_invalid_raw_before_fail_closed_stop(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    published = []
    with pytest.raises(RuntimeError, match="INVALID_RUN"):
        _execute(
            fixture,
            current_source_tree_digest="9" * 64,
            raw_publisher=lambda raw: published.append(deepcopy(raw)),
        )
    assert len(published) == 1
    raw = published[0]
    assert raw["status"] == "INVALID_RUN"
    assert raw["decision"] == "INVALID_RUN"
    assert raw["challenge_materialized"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["per_replicate"] == []
    assert raw["integrity_errors"]


def test_exp279_ceremony_publishes_valid_raw_before_analysis_runs(tmp_path, monkeypatch) -> None:
    import nolane_ai.experiments.exp279_confirmatory_ceremony as ceremony_module

    fixture = _fixture(tmp_path)
    published = []

    def fail_analysis(**_kwargs):
        raise RuntimeError("analysis sentinel failure")

    monkeypatch.setattr(
        ceremony_module,
        "build_exp279_confirmatory_analysis",
        fail_analysis,
    )
    with pytest.raises(RuntimeError, match="analysis sentinel failure"):
        _execute(
            fixture,
            raw_publisher=lambda raw: published.append(deepcopy(raw)),
        )
    assert len(published) == 1
    assert published[0]["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert published[0]["decision"] == "UNVERIFIED"


def test_exp279_ceremony_validator_rejects_rehashed_lineage_or_mode_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_ceremony import (
        _ceremony_digest,
        validate_exp279_gate_b_ceremony,
    )

    fixture = _fixture(tmp_path)
    ceremony = _execute(fixture)

    bad = deepcopy(ceremony)
    bad["scientific_lane_armed"] = True
    bad["ceremony_digest"] = _ceremony_digest(bad)
    errors = validate_exp279_gate_b_ceremony(
        bad,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert any("scientific" in error.lower() or "armed" in error.lower() for error in errors)

    bad = deepcopy(ceremony)
    bad["analysis"]["raw_artifact_digest"] = "9" * 64
    from nolane_ai.experiments.exp279_confirmatory_analysis import _analysis_digest

    bad["analysis"]["analysis_digest"] = _analysis_digest(bad["analysis"])
    bad["ceremony_digest"] = _ceremony_digest(bad)
    errors = validate_exp279_gate_b_ceremony(
        bad,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert any("raw" in error.lower() or "analysis" in error.lower() for error in errors)
