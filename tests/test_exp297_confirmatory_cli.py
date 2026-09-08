from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_beacon import build_test_beacon_receipt
from nolane_ai.experiments.exp297_confirmatory_analysis import (
    validate_exp297_confirmatory_analysis,
)
from nolane_ai.experiments.exp297_confirmatory_executor import (
    validate_exp297_confirmatory_raw,
)
from nolane_ai.experiments.exp297_confirmatory_ceremony import (
    validate_exp297_confirmatory_gate_a_seal,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    validate_exp297_confirmatory_prep,
)
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    validate_exp297_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.experiments.exp297_reconstruction_court import (
    validate_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import source_tree_digest

ROOT = Path(__file__).resolve().parents[1]
GATE_A_SCRIPT = ROOT / "scripts" / "prepare_exp297_confirmatory_gate_a.py"
GATE_B_SCRIPT = ROOT / "scripts" / "run_exp297_confirmatory_gate_b.py"
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST_FILE = ROOT / "protocols" / "stage_a_v1.sha256"
FREEZE_SHA = "f" * 40
FREEZE_TIME = "2026-09-08T07:00:00Z"
BEACON_TIME = "2026-09-08T07:01:00Z"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _registry_for(execution: dict) -> dict:
    return {
        "registry_digest": canonical_sha256({"execution": execution["artifact_digest"]}),
        "experiments": {
            "EXP-297": {
                "evidence_level": "EV-E2",
                "decision": "UNVERIFIED",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {
                    "execution_digest": execution["artifact_digest"]
                },
            }
        },
    }


def _development(*, code_digest: str, replicates: int, start: int, root_seed: str) -> dict:
    protocol_digest = PROTOCOL_DIGEST_FILE.read_text(encoding="utf-8").strip()
    return run_exp297_paired_development(
        root_seed=root_seed,
        eval_replicates=replicates,
        eval_start_replicate=start,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )


def test_exp297_confirmatory_cli_entrypoints_exist():
    assert GATE_A_SCRIPT.exists(), "Gate A CLI entrypoint is missing"
    assert GATE_B_SCRIPT.exists(), "Gate B CLI entrypoint is missing"


@pytest.fixture(scope="module")
def cli_inputs(tmp_path_factory: pytest.TempPathFactory):
    if not GATE_A_SCRIPT.exists() or not GATE_B_SCRIPT.exists():
        pytest.skip("confirmatory CLI entrypoints not implemented yet")
    tmp_path = tmp_path_factory.mktemp("exp297-confirmatory-cli")
    code_digest = source_tree_digest(ROOT)
    execution = _development(
        code_digest=code_digest,
        replicates=32,
        start=20_000,
        root_seed="exp297-confirmatory-cli-test",
    )
    registry = _registry_for(execution)
    execution_path = tmp_path / "execution.json"
    registry_path = tmp_path / "registry.json"
    _write_json(execution_path, execution)
    _write_json(registry_path, registry)
    return tmp_path, code_digest, execution_path, registry_path


def _gate_a_command(
    execution_path: Path,
    registry_path: Path,
    *,
    prep: Path,
    auth: Path,
    reconstruction: Path,
    seal: Path,
    protocol: Path = PROTOCOL,
    protocol_digest_file: Path = PROTOCOL_DIGEST_FILE,
) -> list[str]:
    return [
        sys.executable,
        str(GATE_A_SCRIPT),
        "--protocol",
        str(protocol),
        "--protocol-digest-file",
        str(protocol_digest_file),
        "--execution",
        str(execution_path),
        "--registry",
        str(registry_path),
        "--freeze-commit-sha",
        FREEZE_SHA,
        "--freeze-commit-timestamp-utc",
        FREEZE_TIME,
        "--prep-output",
        str(prep),
        "--authorization-output",
        str(auth),
        "--reconstruction-output",
        str(reconstruction),
        "--seal-output",
        str(seal),
    ]


@pytest.fixture(scope="module")
def gate_a_outputs(cli_inputs):
    tmp_path, _, execution_path, registry_path = cli_inputs
    output_dir = tmp_path / "gate-a"
    output_dir.mkdir()
    prep = output_dir / "prep.json"
    auth = output_dir / "authorization.json"
    reconstruction = output_dir / "reconstruction.json"
    seal = output_dir / "seal.json"
    completed = subprocess.run(
        _gate_a_command(
            execution_path,
            registry_path,
            prep=prep,
            auth=auth,
            reconstruction=reconstruction,
            seal=seal,
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    return prep, auth, reconstruction, seal


def test_gate_a_cli_transactionally_writes_self_validating_pre_beacon_chain(gate_a_outputs):
    prep_path, auth_path, reconstruction_path, seal_path = gate_a_outputs
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert validate_exp297_confirmatory_prep(prep) == []
    assert validate_exp297_confirmatory_execution_authorization(auth) == []
    assert validate_exp297_confirmatory_reconstruction(reconstruction) == []
    assert validate_exp297_confirmatory_gate_a_seal(seal) == []
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_ready"] is True
    assert seal["confirmatory_data_consumed"] is False
    assert seal["challenge_materialized"] is False
    assert seal["semantic_authority_promoted"] is False
    rendered = repr(seal).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered


def test_gate_a_cli_refuses_overwrite_before_publication(cli_inputs):
    tmp_path, _, execution_path, registry_path = cli_inputs
    out = tmp_path / "overwrite"
    out.mkdir()
    prep = out / "prep.json"
    auth = out / "authorization.json"
    reconstruction = out / "reconstruction.json"
    seal = out / "seal.json"
    auth.write_text("sentinel\n", encoding="utf-8")
    completed = subprocess.run(
        _gate_a_command(
            execution_path,
            registry_path,
            prep=prep,
            auth=auth,
            reconstruction=reconstruction,
            seal=seal,
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr + completed.stdout
    assert not prep.exists()
    assert auth.read_text(encoding="utf-8") == "sentinel\n"
    assert not reconstruction.exists()
    assert not seal.exists()


def test_gate_a_cli_rejects_forged_protocol_even_with_matching_digest(cli_inputs):
    tmp_path, _, execution_path, registry_path = cli_inputs
    forged = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    forged["rng"]["root_seed"] = "forged-exp297-confirmatory-root"
    forged_path = tmp_path / "forged-protocol.json"
    _write_json(forged_path, forged)
    digest_path = tmp_path / "forged-protocol.sha256"
    digest_path.write_text(hashlib.sha256(forged_path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    out = tmp_path / "forged-out"
    out.mkdir()
    paths = [out / name for name in ("prep.json", "auth.json", "reconstruction.json", "seal.json")]
    completed = subprocess.run(
        _gate_a_command(
            execution_path,
            registry_path,
            prep=paths[0],
            auth=paths[1],
            reconstruction=paths[2],
            seal=paths[3],
            protocol=forged_path,
            protocol_digest_file=digest_path,
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "canonical frozen protocol digest" in completed.stderr + completed.stdout
    assert not any(path.exists() for path in paths)


def test_gate_a_cli_rejects_pilot_below_32_and_any_beacon_argument(cli_inputs):
    tmp_path, code_digest, _, _ = cli_inputs
    execution = _development(
        code_digest=code_digest,
        replicates=2,
        start=30_000,
        root_seed="exp297-confirmatory-cli-too-small",
    )
    registry = _registry_for(execution)
    execution_path = tmp_path / "small-execution.json"
    registry_path = tmp_path / "small-registry.json"
    _write_json(execution_path, execution)
    _write_json(registry_path, registry)
    out = tmp_path / "small-out"
    out.mkdir()
    paths = [out / name for name in ("prep.json", "auth.json", "reconstruction.json", "seal.json")]
    command = _gate_a_command(
        execution_path,
        registry_path,
        prep=paths[0],
        auth=paths[1],
        reconstruction=paths[2],
        seal=paths[3],
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "at least 32" in completed.stderr + completed.stdout
    assert not any(path.exists() for path in paths)

    completed = subprocess.run(
        command + ["--beacon", str(tmp_path / "forbidden.json")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr + completed.stdout


def test_gate_a_atomic_publish_rolls_back_if_link_fails(cli_inputs, monkeypatch, tmp_path: Path):
    spec = importlib.util.spec_from_file_location("exp297_gate_a_cli", GATE_A_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    real_link = module.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", fail_second_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        module._commit_outputs(((first, {"a": 1}), (second, {"b": 2})))
    assert not first.exists()
    assert not second.exists()


def _gate_b_command(seal: Path, beacon: Path, raw: Path, analysis: Path) -> list[str]:
    return [
        sys.executable,
        str(GATE_B_SCRIPT),
        "--protocol",
        str(PROTOCOL),
        "--protocol-digest-file",
        str(PROTOCOL_DIGEST_FILE),
        "--seal",
        str(seal),
        "--beacon",
        str(beacon),
        "--test-only",
        "--raw-output",
        str(raw),
        "--analysis-output",
        str(analysis),
    ]


@pytest.fixture(scope="module")
def beacon_path(cli_inputs):
    tmp_path, _, _, _ = cli_inputs
    path = tmp_path / "beacon.json"
    _write_json(
        path,
        build_test_beacon_receipt(
            source="synthetic-ci",
            beacon_id="exp297-cli-test-beacon",
            published_at_utc=BEACON_TIME,
            entropy_hex="ab" * 32,
            evidence_reference="TEST-ONLY",
        ),
    )
    return path


def test_gate_b_cli_exposes_no_raw_seed_option(cli_inputs):
    completed = subprocess.run(
        [sys.executable, str(GATE_B_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert "--seed" not in completed.stdout
    completed = subprocess.run(
        [sys.executable, str(GATE_B_SCRIPT), "--seed", "123"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr + completed.stdout


def test_gate_b_cli_test_only_execution_stays_non_scientific(gate_a_outputs, beacon_path, cli_inputs):
    tmp_path, _, _, _ = cli_inputs
    seal = gate_a_outputs[3]
    raw_path = tmp_path / "gate-b-raw.json"
    analysis_path = tmp_path / "gate-b-analysis.json"
    completed = subprocess.run(
        _gate_b_command(seal, beacon_path, raw_path, analysis_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert validate_exp297_confirmatory_raw(raw) == []
    assert validate_exp297_confirmatory_analysis(analysis) == []
    assert raw["test_only"] is True
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["test_only_would_be_decision"] in {
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "KILL_SUBSYSTEM",
    }
    assert analysis["decision_rule_executed"] is False
    assert analysis["semantic_authority_promoted"] is False


def test_gate_b_cli_rejects_missing_or_early_beacon_without_partial_outputs(gate_a_outputs, cli_inputs):
    tmp_path, _, _, _ = cli_inputs
    seal = gate_a_outputs[3]
    for label, beacon in (
        ("missing", tmp_path / "does-not-exist.json"),
        ("early", tmp_path / "early-beacon.json"),
    ):
        if label == "early":
            _write_json(
                beacon,
                build_test_beacon_receipt(
                    source="synthetic-ci",
                    beacon_id="early",
                    published_at_utc=FREEZE_TIME,
                    entropy_hex="cd" * 32,
                    evidence_reference="TEST-ONLY",
                ),
            )
        raw = tmp_path / f"{label}-raw.json"
        analysis = tmp_path / f"{label}-analysis.json"
        completed = subprocess.run(
            _gate_b_command(seal, beacon, raw, analysis),
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert completed.returncode != 0
        if label == "early":
            assert "strictly after" in completed.stderr + completed.stdout
        assert not raw.exists()
        assert not analysis.exists()


def test_gate_b_cli_refuses_overwrite_before_execution(gate_a_outputs, beacon_path, cli_inputs):
    tmp_path, _, _, _ = cli_inputs
    raw = tmp_path / "overwrite-raw.json"
    analysis = tmp_path / "overwrite-analysis.json"
    analysis.write_text("sentinel\n", encoding="utf-8")
    completed = subprocess.run(
        _gate_b_command(gate_a_outputs[3], beacon_path, raw, analysis),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr + completed.stdout
    assert not raw.exists()
    assert analysis.read_text(encoding="utf-8") == "sentinel\n"
