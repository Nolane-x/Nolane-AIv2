from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GATE_A = ROOT / "scripts" / "prepare_exp286_confirmatory_gate_a.py"
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def _load_script(path: Path, name: str):
    assert path.exists(), f"missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _development() -> tuple[dict, dict]:
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from nolane_ai.protocol.identity import source_tree_digest
    from tests.test_neural_arm_registry import _audit, _protocol_subset

    protocol_digest = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()
    code_digest = source_tree_digest(ROOT)
    execution = run_exp286_paired_development(
        root_seed="exp286-gate-a-cli-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=32,
        eval_start_replicate=20_000,
        batch_size=2,
        timesteps=3,
        variables=5,
        decoys=2,
        max_search_steps=8,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )
    registry = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest=protocol_digest,
        model_audit=_audit(),
        exp286_pair_audit=execution["resource_match"]["pair_audit"],
        exp286_execution_artifact=execution,
    )
    return execution, registry


def test_exp286_gate_a_cli_surface_is_seedless_and_beaconless() -> None:
    module = _load_script(GATE_A, "exp286_gate_a_cli_surface")
    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--seed", "123"])
    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--beacon", "beacon.json"])


def test_exp286_gate_a_cli_persists_self_validating_e2_prep_without_challenge(tmp_path: Path) -> None:
    from nolane_ai.experiments.exp286_confirmatory_prep import validate_exp286_confirmatory_prep
    from nolane_ai.protocol.identity import source_tree_digest

    execution, registry = _development()
    execution_path = tmp_path / "execution.json"
    registry_path = tmp_path / "registry.json"
    prep_path = tmp_path / "prep.json"
    _write_json(execution_path, execution)
    _write_json(registry_path, registry)

    module = _load_script(GATE_A, "exp286_gate_a_cli_write")
    assert module.main(
        [
            "--protocol", str(PROTOCOL),
            "--protocol-digest-file", str(PROTOCOL_DIGEST),
            "--execution", str(execution_path),
            "--registry", str(registry_path),
            "--output", str(prep_path),
        ]
    ) == 0

    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    assert validate_exp286_confirmatory_prep(prep) == []
    assert prep["evidence_level"] == "EV-E2"
    assert prep["decision"] == "UNVERIFIED"
    assert prep["analysis_code_digest"] == source_tree_digest(ROOT)
    assert prep["confirmatory_data_consumed"] is False
    assert prep["challenge_materialized"] is False
    assert prep["decision_rule_executed"] is False
    assert prep["confirmatory_lineage"]["seed_materialization_status"] == "NOT_EXECUTED"
    assert prep["status"] in {
        "CONFIRMATORY_GATE_A_PREPARED",
        "NOT_READY_VARIANCE_EXCEEDS_MAX_N",
    }


def test_exp286_gate_a_cli_refuses_overwrite_before_publication(tmp_path: Path) -> None:
    module = _load_script(GATE_A, "exp286_gate_a_cli_overwrite")
    output = tmp_path / "prep.json"
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="output already exists"):
        module.main(
            [
                "--execution", str(tmp_path / "missing-execution.json"),
                "--registry", str(tmp_path / "missing-registry.json"),
                "--output", str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"
