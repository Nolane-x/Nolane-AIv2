from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GATE_A = ROOT / "scripts" / "prepare_exp279_confirmatory_gate_a.py"
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"
FREEZE_SHA = "f" * 40
FREEZE_TIME = "2026-09-09T12:00:00Z"
CHECKPOINT_SEAL_TIME = "2026-09-09T12:00:30Z"
PRIMARY = "verified_utility_per_accounted_flop_on_structure_dense_stratum"


def _load_script(path: Path, name: str):
    assert path.exists(), f"missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_stable_development() -> tuple[dict, dict]:
    from nolane_ai.experiments.exp279_paired_runner import (
        _aggregate_rows,
        _artifact_digest,
        run_exp279_paired_development,
        validate_exp279_paired_development,
    )
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from nolane_ai.protocol.identity import source_tree_digest
    from tests.test_neural_arm_registry import _audit, _protocol_subset

    protocol_digest = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()
    code_digest = source_tree_digest(ROOT)
    execution = run_exp279_paired_development(
        root_seed="exp279-gate-a-cli-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=33,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )

    rows = execution["evaluation"]["per_replicate"]
    max_flops = max(
        float(row[arm]["accounted_flops_per_episode"])
        for row in rows
        for arm in ("propagation_only", "branch_only", "hybrid")
    )
    base = 0.10 / max_flops
    stratum_scale = {
        "PROPAGATION_FIT": 1.00,
        "BRANCH_FIT": 1.03,
        "MIXED_RESIDUAL": 0.97,
    }
    arm_scale = {
        "propagation_only": 1.00,
        "branch_only": 1.01,
        "hybrid": 1.12,
    }
    for row in rows:
        for arm in ("propagation_only", "branch_only", "hybrid"):
            utility = base * stratum_scale[row["stratum"]] * arm_scale[arm]
            flops = float(row[arm]["accounted_flops_per_episode"])
            row[arm]["verified_solution_rate"] = utility * flops
            row[arm][PRIMARY] = utility
    execution["evaluation"]["aggregate"] = _aggregate_rows(rows)
    execution["artifact_digest"] = _artifact_digest(execution)
    assert validate_exp279_paired_development(execution) == []

    registry = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest=protocol_digest,
        model_audit=_audit(),
        exp279_pair_audit=execution["resource_match"]["pair_audit"],
        exp279_execution_artifact=execution,
    )
    return execution, registry


@pytest.fixture(scope="module")
def gate_a_inputs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("exp279-gate-a-cli")
    execution, registry = _make_stable_development()
    execution_path = root / "execution.json"
    registry_path = root / "registry.json"
    _write_json(execution_path, execution)
    _write_json(registry_path, registry)
    return root, execution_path, registry_path


def _gate_a_args(
    root: Path,
    execution: Path,
    registry: Path,
) -> tuple[list[str], dict[str, Path]]:
    out = root / "gate-a"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoint": out / "checkpoint.pt",
        "checkpoint_receipt": out / "checkpoint-receipt.json",
        "prep": out / "prep.json",
        "authorization": out / "authorization.json",
        "seal": out / "seal.json",
        "reconstruction": out / "reconstruction.json",
    }
    args = [
        "--protocol", str(PROTOCOL),
        "--protocol-digest-file", str(PROTOCOL_DIGEST),
        "--execution", str(execution),
        "--registry", str(registry),
        "--freeze-commit-sha", FREEZE_SHA,
        "--freeze-commit-timestamp-utc", FREEZE_TIME,
        "--checkpoint-seal-created-at-utc", CHECKPOINT_SEAL_TIME,
        "--checkpoint-output", str(paths["checkpoint"]),
        "--checkpoint-receipt-output", str(paths["checkpoint_receipt"]),
        "--prep-output", str(paths["prep"]),
        "--authorization-output", str(paths["authorization"]),
        "--seal-output", str(paths["seal"]),
        "--reconstruction-output", str(paths["reconstruction"]),
    ]
    return args, paths


def test_exp279_gate_a_cli_surface_is_seedless_and_beaconless() -> None:
    module = _load_script(GATE_A, "exp279_gate_a_cli_surface")
    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--seed", "123"])
    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--beacon", "beacon.json"])


def test_exp279_gate_a_cli_transactionally_builds_self_validating_pre_beacon_chain(
    gate_a_inputs,
    tmp_path: Path,
) -> None:
    from nolane_ai.experiments.exp279_checkpoint import validate_exp279_checkpoint_receipt
    from nolane_ai.experiments.exp279_confirmatory_authorization import (
        validate_exp279_gate_a_authorization,
        validate_exp279_gate_a_seal,
    )
    from nolane_ai.experiments.exp279_confirmatory_prep import validate_exp279_confirmatory_prep
    from nolane_ai.experiments.exp279_reconstruction_court import validate_exp279_reconstruction_authorization
    from nolane_ai.protocol.identity import file_sha256, source_tree_digest

    _, execution, registry = gate_a_inputs
    module = _load_script(GATE_A, "exp279_gate_a_cli_transaction")
    args, paths = _gate_a_args(tmp_path, execution, registry)
    assert module.main(args) == 0
    assert all(path.exists() for path in paths.values())

    checkpoint_receipt = json.loads(
        paths["checkpoint_receipt"].read_text(encoding="utf-8")
    )
    prep = json.loads(paths["prep"].read_text(encoding="utf-8"))
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    reconstruction = json.loads(
        paths["reconstruction"].read_text(encoding="utf-8")
    )
    assert validate_exp279_checkpoint_receipt(checkpoint_receipt) == []
    assert file_sha256(paths["checkpoint"]) == checkpoint_receipt[
        "checkpoint_file_sha256"
    ]
    assert validate_exp279_confirmatory_prep(prep) == []
    assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"
    assert validate_exp279_gate_a_authorization(authorization) == []
    assert validate_exp279_gate_a_seal(seal) == []
    assert validate_exp279_reconstruction_authorization(reconstruction) == []

    code_digest = source_tree_digest(ROOT)
    assert authorization["source_tree_digest"] == code_digest
    assert authorization["analysis_code_digest"] == code_digest
    assert set(authorization["machinery_digests"].values()) == {code_digest}
    assert authorization["primary_contrasts"] == [
        "hybrid_vs_propagation_only",
        "hybrid_vs_branch_only",
    ]
    assert authorization["stratum_schedule"] == [
        "PROPAGATION_FIT",
        "BRANCH_FIT",
        "MIXED_RESIDUAL",
    ]
    assert authorization["pilot_best_simple_not_carried_into_confirmatory"] is True
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["challenge_materialized"] is False


def test_exp279_gate_a_cli_refuses_partial_overwrite_before_checkpoint_creation(
    gate_a_inputs,
    tmp_path: Path,
) -> None:
    module = _load_script(GATE_A, "exp279_gate_a_cli_overwrite")
    _, execution, registry = gate_a_inputs
    args, paths = _gate_a_args(tmp_path, execution, registry)
    paths["authorization"].write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="output already exists"):
        module.main(args)
    assert paths["authorization"].read_text(encoding="utf-8") == "sentinel\n"
    for name, path in paths.items():
        if name != "authorization":
            assert not path.exists()
