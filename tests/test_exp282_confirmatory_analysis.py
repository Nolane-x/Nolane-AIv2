from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.exp282_confirmatory_fixtures import (
    CANONICAL_PROTOCOL_DIGEST,
    DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
    prepared_chain,
    set_raw_outcomes,
)


def _build(tmp_path: Path, *, accuracy_delta: float, brier_delta: float = -0.01):
    from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis

    protocol, prep, execution, reconstruction, raw = prepared_chain(tmp_path)
    raw = set_raw_outcomes(raw, accuracy_delta=accuracy_delta, brier_delta=brier_delta)
    result = build_exp282_confirmatory_analysis(
        protocol=protocol,
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        prep_artifact=prep,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        raw_artifact=raw,
        analysis_code_digest=DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
    )
    return result, protocol, prep, execution, reconstruction, raw


def test_confirmatory_analysis_promotes_only_when_mesi_and_guards_close(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import BOOTSTRAP_SAMPLES, validate_exp282_confirmatory_analysis

    result, _, _, _, _, raw = _build(tmp_path, accuracy_delta=0.05)
    assert BOOTSTRAP_SAMPLES == 10_000
    assert result["schema"] == "NLM-EXP-282-CONFIRMATORY-ANALYSIS-V1"
    assert result["evidence_level"] == "EV-E3"
    assert result["decision"] == "PROMOTE_TO_NEXT_STAGE"
    assert result["primary_effect"]["one_sided_lower_95"] >= 0.03
    assert result["protected_endpoints"]["brier"]["pass"] is True
    assert result["protected_endpoints"]["compute"]["pass"] is True
    assert result["raw_per_replicate_metrics"] == raw["per_replicate"]
    assert result["remaining_blockers"] == [
        "100M post-freeze challenge replication remains open for EV-E4",
        "independent clean-room replication remains open for EV-E5",
    ]
    assert validate_exp282_confirmatory_analysis(result) == []


def test_confirmatory_analysis_holds_when_primary_crosses_mesi_boundary(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis
    from nolane_ai.experiments.exp282_confirmatory_executor import _raw_digest

    protocol, prep, execution, reconstruction, raw = prepared_chain(tmp_path)
    raw = deepcopy(raw)
    for index, row in enumerate(raw["per_replicate"]):
        delta = 0.00 if index < 16 else 0.06
        recurrent_accuracy = 0.40
        explicit_accuracy = recurrent_accuracy + delta
        for arm, accuracy in (("recurrent_hidden", recurrent_accuracy), ("explicit_belief", explicit_accuracy)):
            row[arm] = {
                "grounded_decision_accuracy": accuracy,
                "correct": int(round(accuracy * 100)),
                "total": 100,
                "brier_score": 0.20,
                "brier_sum": 20.0,
                "brier_count": 100,
            }
        row["explicit_minus_recurrent_accuracy"] = delta
        row["explicit_minus_recurrent_brier"] = 0.0
    raw["artifact_digest"] = _raw_digest(raw)
    result = build_exp282_confirmatory_analysis(
        protocol=protocol,
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        prep_artifact=prep,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        raw_artifact=raw,
        analysis_code_digest=DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
    )
    assert result["primary_effect"]["one_sided_lower_95"] < 0.03
    assert result["primary_effect"]["one_sided_upper_95"] >= 0.03
    assert result["decision"] == "HOLD_UNSTABLE"


def test_confirmatory_analysis_kills_when_upper_bound_cannot_clear_mesi(tmp_path: Path):
    result, *_ = _build(tmp_path, accuracy_delta=0.0)
    assert result["primary_effect"]["one_sided_upper_95"] < 0.03
    assert result["decision"] == "KILL_SUBSYSTEM"


def test_confirmatory_analysis_brier_guard_blocks_promotion_without_inventing_kill(tmp_path: Path):
    result, *_ = _build(tmp_path, accuracy_delta=0.05, brier_delta=0.03)
    assert result["primary_effect"]["one_sided_lower_95"] >= 0.03
    assert result["protected_endpoints"]["brier"]["delta"] > 0.02
    assert result["protected_endpoints"]["brier"]["pass"] is False
    assert result["decision"] == "HOLD_UNSTABLE"


def test_confirmatory_analysis_bootstrap_is_deterministic_and_pre_result_seeded(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import bootstrap_paired_effect
    from nolane_ai.protocol.evidence import canonical_sha256

    _, _, prep, *_ = _build(tmp_path, accuracy_delta=0.05)
    frozen_analysis_digest = canonical_sha256(prep["frozen_analysis"])
    seed_material = f"{CANONICAL_PROTOCOL_DIGEST}|{frozen_analysis_digest}|EXP-282|paired-bootstrap-v1"
    first = bootstrap_paired_effect([0.0, 0.04, 0.08], seed_material=seed_material)
    second = bootstrap_paired_effect([0.0, 0.04, 0.08], seed_material=seed_material)
    assert first == second
    assert first["samples"] == 10_000
    assert first["alpha"] == 0.05
    assert len(first["seed_digest"]) == 64


def test_confirmatory_analysis_rejects_noncanonical_protocol_digest(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis

    protocol, prep, execution, reconstruction, raw = prepared_chain(tmp_path)
    with pytest.raises(ValueError, match="canonical frozen protocol digest"):
        build_exp282_confirmatory_analysis(
            protocol=protocol,
            protocol_digest="f" * 64,
            prep_artifact=prep,
            paired_execution_artifact=execution,
            reconstruction_authorization=reconstruction,
            raw_artifact=raw,
            analysis_code_digest=DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
        )


def test_confirmatory_analysis_rejects_tampered_raw_before_statistics(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis

    protocol, prep, execution, reconstruction, raw = prepared_chain(tmp_path)
    raw = deepcopy(raw)
    raw["per_replicate"][0]["explicit_minus_recurrent_accuracy"] = 1.0
    with pytest.raises(ValueError, match="invalid confirmatory raw artifact"):
        build_exp282_confirmatory_analysis(
            protocol=protocol,
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            prep_artifact=prep,
            paired_execution_artifact=execution,
            reconstruction_authorization=reconstruction,
            raw_artifact=raw,
            analysis_code_digest=DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
        )


def test_confirmatory_analysis_rejects_analysis_code_drift_after_prep_freeze(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis

    protocol, prep, execution, reconstruction, raw = prepared_chain(tmp_path)
    with pytest.raises(ValueError, match="frozen analysis code digest mismatch"):
        build_exp282_confirmatory_analysis(
            protocol=protocol,
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            prep_artifact=prep,
            paired_execution_artifact=execution,
            reconstruction_authorization=reconstruction,
            raw_artifact=raw,
            analysis_code_digest="b" * 64,
        )


def test_confirmatory_analysis_rejects_prep_pilot_summary_not_reconstructible_from_paired_execution(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import build_exp282_confirmatory_analysis
    from nolane_ai.experiments.exp282_confirmatory_prep import _prep_digest

    protocol, prep, execution, reconstruction, raw = prepared_chain(tmp_path)
    tampered = deepcopy(prep)
    tampered["pilot_summary"]["mean_accuracy_gain"] = 0.99
    tampered["pilot_summary"]["paired_effect_digest"] = "f" * 64
    tampered["prep_digest"] = _prep_digest(tampered)
    with pytest.raises(ValueError, match="pilot summary does not match paired execution"):
        build_exp282_confirmatory_analysis(
            protocol=protocol,
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            prep_artifact=tampered,
            paired_execution_artifact=execution,
            reconstruction_authorization=reconstruction,
            raw_artifact=raw,
            analysis_code_digest=DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
        )


def test_confirmatory_analysis_validator_rejects_digest_tamper(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import validate_exp282_confirmatory_analysis

    result, *_ = _build(tmp_path, accuracy_delta=0.05)
    tampered = deepcopy(result)
    tampered["primary_effect"]["observed_mean"] += 0.01
    errors = validate_exp282_confirmatory_analysis(tampered)
    assert "confirmatory analysis digest mismatch" in errors
