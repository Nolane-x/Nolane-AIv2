from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GATE_A = ROOT / "scripts" / "prepare_exp277_confirmatory_gate_a.py"
GATE_B = ROOT / "scripts" / "run_exp277_confirmatory_gate_b.py"
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"
FREEZE_SHA = "f" * 40
FREEZE_TIME = "2026-09-08T12:00:00Z"
CHECKPOINT_SEAL_TIME = "2026-09-08T12:00:30Z"
BEACON_TIME = "2026-09-08T12:01:00Z"


def _load_script(path: Path, name: str):
    assert path.exists(), f"missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _make_stable_development() -> tuple[dict, dict]:
    from nolane_ai.experiments.exp277_paired_runner import (
        _artifact_digest,
        run_exp277_paired_development,
        validate_exp277_paired_development,
    )
    from nolane_ai.protocol.evidence import canonical_sha256
    from nolane_ai.protocol.identity import source_tree_digest

    protocol_digest = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()
    code_digest = source_tree_digest(ROOT)
    execution = run_exp277_paired_development(
        root_seed="exp277-gate-a-cli-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=1,
        eval_replicates=32,
        eval_start_replicate=10_000,
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
    arcs_flops = int(rows[0]["arcs_branch"]["accounted_flops_per_episode"])
    oracle_flops = int(rows[0]["oracle_cbrf"]["accounted_flops_per_episode"])
    utility = 0.1 / max(arcs_flops, oracle_flops)
    for row in rows:
        arcs_solution = utility * arcs_flops
        oracle_solution = utility * 1.20 * oracle_flops
        row["arcs_branch"]["verified_solution_rate"] = arcs_solution
        row["arcs_branch"]["verified_decision_accuracy"] = arcs_solution
        row["arcs_branch"]["verified_utility_per_accounted_flop"] = utility
        row["oracle_cbrf"]["verified_solution_rate"] = oracle_solution
        row["oracle_cbrf"]["verified_decision_accuracy"] = oracle_solution
        row["oracle_cbrf"]["verified_utility_per_accounted_flop"] = utility * 1.20
        row["oracle_relative_verified_utility_gain"] = 0.20

    aggregate = execution["evaluation"]["aggregate"]
    aggregate["mean_arcs_utility"] = utility
    aggregate["mean_oracle_utility"] = utility * 1.20
    aggregate["oracle_relative_utility_gain"] = 0.20
    aggregate["oracle_minus_arcs_verified_solution_rate"] = (
        utility * 1.20 * oracle_flops - utility * arcs_flops
    )
    execution["artifact_digest"] = _artifact_digest(execution)
    assert validate_exp277_paired_development(execution) == []

    registry = {
        "schema": "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "protocol_digest": protocol_digest,
        "registry_digest": canonical_sha256(
            {"experiment": "EXP-277", "execution": execution["artifact_digest"]}
        ),
        "experiments": {
            "EXP-277": {
                "development_match_status": "PAIRED_STRUCTURE_DENSE_DEV_READY",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {
                    "artifact_digest": execution["artifact_digest"],
                    "code_digest": code_digest,
                },
            }
        },
    }
    return execution, registry


@pytest.fixture(scope="module")
def gate_a_inputs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("exp277-gate-a-cli")
    execution, registry = _make_stable_development()
    execution_path = root / "execution.json"
    registry_path = root / "registry.json"
    _write_json(execution_path, execution)
    _write_json(registry_path, registry)
    return root, execution_path, registry_path


def _gate_a_args(root: Path, execution: Path, registry: Path) -> tuple[list[str], dict[str, Path]]:
    out = root / "gate-a"
    out.mkdir(exist_ok=True)
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


def test_exp277_gate_a_cli_surface_is_seedless_and_beaconless() -> None:
    module = _load_script(GATE_A, "exp277_gate_a_cli_surface")
    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--seed", "123"])
    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--beacon", "beacon.json"])


def test_exp277_gate_a_cli_transactionally_builds_self_validating_pre_beacon_chain(gate_a_inputs) -> None:
    from nolane_ai.experiments.exp277_checkpoint import validate_exp277_checkpoint_receipt
    from nolane_ai.experiments.exp277_confirmatory_authorization import (
        validate_exp277_gate_a_authorization,
        validate_exp277_gate_a_seal,
    )
    from nolane_ai.experiments.exp277_confirmatory_prep import validate_exp277_confirmatory_prep
    from nolane_ai.experiments.exp277_reconstruction_court import validate_exp277_reconstruction_authorization
    from nolane_ai.protocol.identity import file_sha256, source_tree_digest

    root, execution, registry = gate_a_inputs
    module = _load_script(GATE_A, "exp277_gate_a_cli_transaction")
    args, paths = _gate_a_args(root, execution, registry)
    assert module.main(args) == 0
    assert all(path.exists() for path in paths.values())

    checkpoint_receipt = json.loads(paths["checkpoint_receipt"].read_text(encoding="utf-8"))
    prep = json.loads(paths["prep"].read_text(encoding="utf-8"))
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    reconstruction = json.loads(paths["reconstruction"].read_text(encoding="utf-8"))
    assert validate_exp277_checkpoint_receipt(checkpoint_receipt) == []
    assert file_sha256(paths["checkpoint"]) == checkpoint_receipt["checkpoint_file_sha256"]
    assert validate_exp277_confirmatory_prep(prep) == []
    assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"
    assert validate_exp277_gate_a_authorization(authorization) == []
    assert validate_exp277_gate_a_seal(seal) == []
    assert validate_exp277_reconstruction_authorization(reconstruction) == []
    assert authorization["source_tree_digest"] == source_tree_digest(ROOT)
    assert authorization["analysis_code_digest"] == source_tree_digest(ROOT)
    assert set(authorization["machinery_digests"].values()) == {source_tree_digest(ROOT)}
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["challenge_materialized"] is False
    rendered = repr(seal).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered


def test_exp277_gate_a_cli_refuses_partial_overwrite_before_checkpoint_creation(gate_a_inputs, tmp_path: Path) -> None:
    module = _load_script(GATE_A, "exp277_gate_a_cli_overwrite")
    _, execution, registry = gate_a_inputs
    args, paths = _gate_a_args(tmp_path, execution, registry)
    paths["authorization"].write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="output already exists"):
        module.main(args)
    assert paths["authorization"].read_text(encoding="utf-8") == "sentinel\n"
    for name, path in paths.items():
        if name != "authorization":
            assert not path.exists()


def test_exp277_full_gate_a_to_gate_b_test_only_e2e_never_creates_scientific_evidence(gate_a_inputs, tmp_path: Path) -> None:
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt
    from nolane_ai.experiments.exp277_confirmatory_analysis import validate_exp277_confirmatory_analysis
    from nolane_ai.experiments.exp277_confirmatory_executor import validate_exp277_confirmatory_raw

    root, execution, registry = gate_a_inputs
    gate_a = _load_script(GATE_A, "exp277_gate_a_cli_e2e")
    args, paths = _gate_a_args(tmp_path / "e2e", execution, registry)
    assert gate_a.main(args) == 0

    beacon_path = tmp_path / "test-only-beacon.json"
    _write_json(
        beacon_path,
        build_test_beacon_receipt(
            source="synthetic-exp277-e2e",
            beacon_id="test-only-e2e-round",
            published_at_utc=BEACON_TIME,
            entropy_hex="34" * 32,
            evidence_reference="test-only://exp277-e2e",
        ),
    )
    raw_path = tmp_path / "raw.json"
    analysis_path = tmp_path / "analysis.json"
    gate_b = _load_script(GATE_B, "exp277_gate_b_cli_e2e")
    assert gate_b.main(
        [
            "--protocol", str(PROTOCOL),
            "--protocol-digest-file", str(PROTOCOL_DIGEST),
            "--seal", str(paths["seal"]),
            "--reconstruction", str(paths["reconstruction"]),
            "--beacon", str(beacon_path),
            "--checkpoint", str(paths["checkpoint"]),
            "--checkpoint-receipt", str(paths["checkpoint_receipt"]),
            "--test-only",
            "--raw-output", str(raw_path),
            "--analysis-output", str(analysis_path),
        ]
    ) == 0
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    checkpoint_receipt = json.loads(paths["checkpoint_receipt"].read_text(encoding="utf-8"))
    assert validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=paths["checkpoint"],
        checkpoint_receipt=checkpoint_receipt,
    ) == []
    assert validate_exp277_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=paths["checkpoint"],
        checkpoint_receipt=checkpoint_receipt,
    ) == []
    assert raw["test_only"] is True
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert analysis["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["scientific_evidence_eligible"] is False
    assert analysis["confirmatory_data_consumed"] is False
    assert analysis["decision_rule_executed"] is False
    assert analysis["test_only_would_be_decision"] in {
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "KILL_SUBSYSTEM",
    }
