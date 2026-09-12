from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SHARD_SCRIPT = ROOT / "scripts" / "run_exp279_powered_support_stability_v8_dev.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_exp279_powered_support_stability_v8_dev.py"
BRANCH_HEAD = "a" * 40
EXECUTED_COMMIT = "b" * 40


def _shard_command(output: Path) -> list[str]:
    return [
        sys.executable,
        str(SHARD_SCRIPT),
        "--tiny-contract",
        "--train-replicates", "60",
        "--canonical-index", "0",
        "--scientific-branch-head", BRANCH_HEAD,
        "--executed-commit", EXECUTED_COMMIT,
        "--output", str(output),
    ]


def test_v8_shard_cli_writes_augmentation_only_receipt_with_exact_provenance(tmp_path: Path) -> None:
    output = tmp_path / "v8-shard.json"
    completed = subprocess.run(_shard_command(output), cwd=ROOT, check=True, text=True, capture_output=True)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())

    assert receipt["schema"] == "NLM-EXP-279-POWERED-SUPPORT-SHARD-V8"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["scientific_branch_head"] == BRANCH_HEAD
    assert receipt["executed_commit"] == EXECUTED_COMMIT
    assert receipt["training_rng_stream"] == "augmentation"
    assert receipt["probe_rng_stream"] == "augmentation"
    assert receipt["evaluation_rng_stream_used"] is False
    assert receipt["evaluation_targets_used"] is False
    assert receipt["fresh_evaluation_lineage_may_be_reserved"] is False
    assert receipt["fresh_evaluation_lineage_consumed"] is False
    assert receipt["confirmatory_data_consumed"] is False
    assert receipt["challenge_materialized"] is False
    assert receipt["promotion_claimed"] is False
    assert stdout == {
        "artifact_digest": receipt["artifact_digest"],
        "canonical_index": 0,
        "scientific_branch_head": BRANCH_HEAD,
        "train_replicates": 60,
    }


def test_v8_shard_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    output = tmp_path / "v8-shard.json"
    output.write_text("sentinel\n", encoding="utf-8")
    command = _shard_command(output) + ["--protocol", str(tmp_path / "missing.json")]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_v8_shard_cli_rejects_noncanonical_protocol_even_with_matching_local_digest(tmp_path: Path) -> None:
    output = tmp_path / "v8-shard.json"
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes((ROOT / "protocols" / "stage_a_v1.json").read_bytes() + b"\n")
    digest = tmp_path / "protocol.sha256"
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    command = _shard_command(output) + [
        "--protocol", str(protocol),
        "--protocol-digest-file", str(digest),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "canonical frozen protocol digest mismatch" in completed.stderr
    assert not output.exists()


def test_v8_shard_cli_exposes_no_evaluation_or_scientific_geometry_knobs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SHARD_SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    help_text = completed.stdout.lower()
    for forbidden in (
        "--eval-",
        "confirmatory",
        "challenge",
        "fresh-lineage",
        "fresh-evaluation",
        "60000",
        "--probe-replicates",
        "--batch-size",
        "--d-model",
        "--route-threshold",
        "--lr",
    ):
        assert forbidden not in help_text
    for required in (
        "--train-replicates",
        "--canonical-index",
        "--scientific-branch-head",
        "--executed-commit",
        "--output",
    ):
        assert required in help_text


def test_v8_budget_cli_refuses_existing_output_before_loading_shards(tmp_path: Path) -> None:
    output = tmp_path / "v8-budget.json"
    output.write_text("sentinel\n", encoding="utf-8")
    missing = tmp_path / "missing.json"
    command = [
        sys.executable,
        str(AGGREGATE_SCRIPT),
        "--shard", str(missing),
        "--shard", str(missing),
        "--shard", str(missing),
        "--shard", str(missing),
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_v8_budget_cli_rejects_non_frozen_or_duplicate_shards(tmp_path: Path) -> None:
    shard = tmp_path / "tiny.json"
    subprocess.run(_shard_command(shard), cwd=ROOT, check=True, text=True, capture_output=True)
    output = tmp_path / "budget.json"
    command = [sys.executable, str(AGGREGATE_SCRIPT)]
    for _ in range(4):
        command += ["--shard", str(shard)]
    command += ["--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert not output.exists()
