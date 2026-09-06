import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_heldout_evaluation_cli_writes_ev_e2_artifact(tmp_path: Path):
    output = tmp_path / "heldout.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_stage_a_pilot.py"),
            "--tiny",
            "--train-steps", "2",
            "--eval-batches", "2",
            "--batch-size", "2",
            "--variables", "3",
            "--constraints", "2",
            "--eval-start-replicate", "100",
            "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    report = artifact["report"]
    assert artifact["schema"] == "NLM-STAGE-A-HELDOUT-DEV-ARTIFACT-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["execution_contract"]["training"]["rng_stream"] == "augmentation"
    assert artifact["execution_contract"]["evaluation"]["rng_stream"] == "evaluation"
    assert artifact["class_support"]["aggregate"]["belief"]["positive"] + artifact["class_support"]["aggregate"]["belief"]["negative"] == 2 * 2 * 3
    assert report["schema"] == "NLM-STAGE-A-HELDOUT-DEV-EVAL-V1"
    assert report["evidence_level"] == "EV-E2"
    assert report["decision"] == "UNVERIFIED"
    assert report["training_stream"] == "augmentation"
    assert report["evaluation_stream"] == "evaluation"
    assert report["pre"]["batch_digests"] == report["post"]["batch_digests"]
    assert report["model_before_digest"] != report["model_after_digest"]
    assert stdout["report_digest"] == report["report_digest"]
    assert stdout["artifact_digest"] == artifact["artifact_digest"]
