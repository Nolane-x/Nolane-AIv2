import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _protocol_fixture(tmp_path: Path):
    # Reuse the repository's structurally valid frozen fixture, patching only the
    # exact arm authority expected by the Match Court. This is test-only.
    source = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    source.setdefault("global_sample_size_plan", {"familywise_alpha": 0.05})
    expected = {
        "EXP-277": [
            {"id": "arcs_branch", "description": "V0.15 ARCS recurrent-depth + branch bank + verifier court"},
            {"id": "oracle_cbrf", "description": "same substrate with ground-truth constraint/factor representation"},
        ],
        "EXP-279": [
            {"id": "propagation_only", "description": "constraint propagation without branch search"},
            {"id": "branch_only", "description": "ARCS branch search without CBRF propagation"},
            {"id": "hybrid", "description": "propagation followed by branch search when residual uncertainty remains"},
        ],
        "EXP-282": [
            {"id": "recurrent_hidden", "description": "matched recurrent state without explicit belief representation"},
            {"id": "explicit_belief", "description": "explicit calibrated belief state over hidden world variables"},
        ],
    }
    resources = {
        "EXP-277": {"parameter_budget": "matched active parameter count", "inference_budget": "same max accounted FLOPs per episode", "world_pairing": "same world lineage and replicate index"},
        "EXP-279": {"parameter_budget": "reclaimed parameters assigned to simpler rivals", "inference_budget": "equal max accounted FLOPs", "structure_fit": "predeclared strata"},
        "EXP-282": {"parameter_budget": "equal state/controller parameters", "observation_history": "identical", "compute_budget": "matched"},
    }
    for experiment in source["experiments"]:
        if experiment["experiment_id"] in expected:
            experiment["arms"] = expected[experiment["experiment_id"]]
            experiment["resource_match"] = resources[experiment["experiment_id"]]
        if experiment["experiment_id"] == "EXP-282":
            experiment["primary_endpoint"] = {"metric": "grounded_decision_accuracy", "direction": "higher"}
            experiment["protected_endpoints"] = [
                {"metric": "brier_score", "floor": "explicit_belief <= recurrent_hidden + 0.02"},
                {"metric": "accounted_flops", "floor": "difference <= 0.05 relative unless included in primary cost normalization"},
            ]
            experiment["mesi"] = {"type": "absolute_gain", "value": 0.03, "unit": "accuracy_fraction"}
            experiment["sample_size_plan"] = {"power_target": 0.90, "min_n": 32, "max_n": 128, "paired": True}
            experiment["analysis_method"] = "paired accuracy difference with bootstrap CI plus calibration guard"
            experiment["multiplicity_family"] = "BELIEF_STATE"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(source, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    import hashlib
    digest = tmp_path / "protocol.sha256"
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    return protocol, digest


def test_confirmatory_prep_cli_uses_real_paired_and_registry_artifacts_without_consuming_confirmatory_data(tmp_path: Path):
    protocol, digest_file = _protocol_fixture(tmp_path)
    execution = tmp_path / "execution.json"
    registry = tmp_path / "registry.json"
    prep = tmp_path / "prep.json"

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "run_exp282_paired_dev.py"),
        "--tiny",
        "--protocol", str(protocol),
        "--protocol-digest-file", str(digest_file),
        "--train-replicates", "2",
        "--eval-replicates", "32",
        "--batch-size", "2",
        "--timesteps", "3",
        "--variables", "2",
        "--output", str(execution),
        "--registry-output", str(registry),
    ], cwd=ROOT, check=True, capture_output=True, text=True)

    completed = subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "prepare_exp282_confirmatory_open.py"),
        "--protocol", str(protocol),
        "--protocol-digest-file", str(digest_file),
        "--execution", str(execution),
        "--registry", str(registry),
        "--output", str(prep),
    ], cwd=ROOT, check=True, capture_output=True, text=True)

    payload = json.loads(prep.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    assert payload["schema"] == "NLM-EXP-282-CONFIRMATORY-OPEN-PREP-V1"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["pilot_summary"]["n"] == 32
    assert payload["confirmatory_data_consumed"] is False
    assert payload["status"] in {"CONFIRMATORY_OPEN_PREPARED", "NOT_READY_VARIANCE_EXCEEDS_MAX_N"}
    assert stdout["prep_digest"] == payload["prep_digest"]
    assert stdout["status"] == payload["status"]
