from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.protocol.evidence import canonical_sha256
from tests.exp289_confirmatory_fixtures import CODE_DIGEST, task5_inputs
from tests.test_exp289_confirmatory_authorization import _authorize
from tests.test_exp289_confirmatory_ceremony import _seal


ROOT = Path(__file__).resolve().parents[1]
GATE_A_SEAL_SCRIPT = ROOT / "scripts" / "seal_exp289_confirmatory_gate_a.py"
GATE_B_SCRIPT = ROOT / "scripts" / "run_exp289_confirmatory_gate_b.py"
VALID_SCIENTIFIC_DECISIONS = {
    "PROMOTE_TO_NEXT_STAGE",
    "HOLD_UNSTABLE",
    "KILL_SUBSYSTEM",
}


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _external_beacon(seal: dict, *, published_at_utc: str | None = None) -> dict:
    from nolane_ai.experiments.exp289_beacon import validate_exp289_beacon_receipt

    if published_at_utc is None:
        published_at_utc = (
            _parse_utc(seal["seal_created_at_utc"]) + timedelta(seconds=1)
        ).isoformat().replace("+00:00", "Z")
    payload = {
        "schema": "NLM-EXP-289-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-289",
        "source": "drand-public-http-cross-recorded",
        "beacon_id": "drand:test-chain:289000001",
        "published_at_utc": published_at_utc,
        "entropy_hex": "cd" * 32,
        "evidence_reference": "https://api.drand.sh/public/289000001;https://api2.drand.sh/public/289000001",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": "EXTERNAL_EVIDENCE_RECORDED",
        "receipt_digest": "",
    }
    payload["receipt_digest"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_digest"}
    )
    assert validate_exp289_beacon_receipt(payload) == []
    return payload


def test_exp289_operational_ceremony_cli_entrypoints_exist_and_expose_no_raw_seed() -> None:
    assert GATE_A_SEAL_SCRIPT.exists(), "EXP-289 immutable Gate-A sealing CLI is missing"
    assert GATE_B_SCRIPT.exists(), "EXP-289 Gate-B execution CLI is missing"

    gate_a = subprocess.run(
        [sys.executable, str(GATE_A_SEAL_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert gate_a.returncode == 0, gate_a.stderr + gate_a.stdout
    assert "--seed" not in gate_a.stdout
    assert "--beacon" not in gate_a.stdout
    assert "--checkpoint-output" in gate_a.stdout
    assert "--seal-output" in gate_a.stdout

    gate_b = subprocess.run(
        [sys.executable, str(GATE_B_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert gate_b.returncode == 0, gate_b.stderr + gate_b.stdout
    assert "--seed" not in gate_b.stdout
    assert "--test-only" in gate_b.stdout
    assert "--checkpoint" in gate_b.stdout
    assert "--checkpoint-receipt" in gate_b.stdout

    rejected = subprocess.run(
        [sys.executable, str(GATE_B_SCRIPT), "--seed", "123"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "unrecognized arguments" in rejected.stderr + rejected.stdout


def test_exp289_gate_a_seal_geometry_match_treats_tiny_as_manifest_only_marker() -> None:
    spec = importlib.util.spec_from_file_location("exp289_seal_cli", GATE_A_SEAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest_geometry = {
        "tiny": False,
        "root_seed": "authority-test",
        "d_model": 64,
        "hidden_size": 48,
    }
    execution_configuration = {
        "root_seed": "authority-test",
        "d_model": 64,
        "hidden_size": 48,
    }
    assert module._geometry_matches_execution(
        execution_configuration,
        manifest_geometry,
    )
    drifted = deepcopy(execution_configuration)
    drifted["d_model"] = 32
    assert not module._geometry_matches_execution(drifted, manifest_geometry)


def test_exp289_executor_rejects_beacon_after_freeze_but_before_actual_seal(
    task5_inputs: dict,
) -> None:
    from nolane_ai.experiments.exp289_confirmatory_executor import execute_exp289_confirmatory_challenge

    authorization = _authorize(task5_inputs)
    seal = _seal(task5_inputs, authorization)
    freeze = _parse_utc(seal["freeze_commit_timestamp_utc"])
    sealed = _parse_utc(seal["seal_created_at_utc"])
    assert sealed > freeze
    pre_seal_time = (freeze + timedelta(microseconds=1)).isoformat().replace("+00:00", "Z")
    assert _parse_utc(pre_seal_time) < sealed
    beacon = _external_beacon(seal, published_at_utc=pre_seal_time)

    with pytest.raises(ValueError, match="beacon|seal"):
        execute_exp289_confirmatory_challenge(
            seal=seal,
            beacon_receipt=beacon,
            checkpoint_path=task5_inputs["checkpoint_path"],
            checkpoint_receipt=task5_inputs["checkpoint_receipt"],
            current_source_tree_digest=CODE_DIGEST,
            executor_code_digest=CODE_DIGEST,
        )


def test_exp289_real_beacon_executes_raw_before_analysis_and_promotes_only_analysis(
    task5_inputs: dict,
) -> None:
    from nolane_ai.experiments.exp289_confirmatory_executor import (
        execute_exp289_confirmatory_challenge,
        validate_exp289_confirmatory_raw,
    )
    from nolane_ai.experiments.exp289_confirmatory_analysis import (
        build_exp289_confirmatory_analysis,
        validate_exp289_confirmatory_analysis,
    )

    authorization = _authorize(task5_inputs)
    seal = _seal(task5_inputs, authorization)
    beacon = _external_beacon(seal)
    raw = execute_exp289_confirmatory_challenge(
        seal=seal,
        beacon_receipt=beacon,
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
        current_source_tree_digest=CODE_DIGEST,
        executor_code_digest=CODE_DIGEST,
    )
    assert validate_exp289_confirmatory_raw(
        raw,
        seal=seal,
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
    ) == []
    assert raw["status"] == "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["test_only"] is False
    assert raw["scientific_evidence_eligible"] is True
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["confirmatory_data_consumed"] is True
    assert raw["synthetic_challenge_data_consumed"] is False
    assert raw["seed_materialization_status"] == "EXECUTED"
    assert raw["challenge_materialized"] is True
    assert raw["decision_rule_executed"] is False
    assert [row["replicate"] for row in raw["per_replicate"]] == seal["reserved_replicate_ids"]

    analysis = build_exp289_confirmatory_analysis(
        raw_artifact=raw,
        seal=seal,
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
        analysis_code_digest=CODE_DIGEST,
    )
    assert validate_exp289_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        seal=seal,
        checkpoint_path=task5_inputs["checkpoint_path"],
        checkpoint_receipt=task5_inputs["checkpoint_receipt"],
    ) == []
    assert analysis["status"] == "CONFIRMATORY_CHALLENGE_ANALYZED"
    assert analysis["test_only"] is False
    assert analysis["scientific_evidence_eligible"] is True
    assert analysis["evidence_level"] == "EV-E3"
    assert analysis["decision"] in VALID_SCIENTIFIC_DECISIONS
    assert analysis["confirmatory_data_consumed"] is True
    assert analysis["synthetic_challenge_data_consumed"] is False
    assert analysis["decision_rule_executed"] is True
    assert analysis["test_only_decision_executed"] is False
    assert analysis["test_only_would_be_decision"] is None
    assert analysis["raw_artifact_digest"] == raw["artifact_digest"]
    assert analysis["lineage"]["beacon_receipt_digest"] == beacon["receipt_digest"]


def test_exp289_non_test_beacon_must_be_explicitly_scientific(task5_inputs: dict) -> None:
    from nolane_ai.experiments.exp289_confirmatory_executor import execute_exp289_confirmatory_challenge

    authorization = _authorize(task5_inputs)
    seal = _seal(task5_inputs, authorization)
    forged = deepcopy(_external_beacon(seal))
    forged["scientific_evidence_eligible"] = False
    forged["receipt_digest"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "receipt_digest"}
    )
    with pytest.raises(ValueError, match="scientific|beacon"):
        execute_exp289_confirmatory_challenge(
            seal=seal,
            beacon_receipt=forged,
            checkpoint_path=task5_inputs["checkpoint_path"],
            checkpoint_receipt=task5_inputs["checkpoint_receipt"],
            current_source_tree_digest=CODE_DIGEST,
            executor_code_digest=CODE_DIGEST,
        )