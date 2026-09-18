from __future__ import annotations

import json
from pathlib import Path

from nolane_ai.experiments.exp323r_repair import (
    load_locked_reconstruction,
    validate_selection_lock,
)


LOCK = Path("protocols/v017/exp323r_reconstruction_selection_lock_v1.json")


def test_selection_lock_binds_lowest_exact_candidate_and_checkpoint_bytes() -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_selection_lock(payload)
    assert payload["selection_rule"] == "lowest_verified_attempt_index"
    assert payload["verified_candidate_attempts"] == [0, 2]
    assert payload["selected_attempt_index"] == 0
    assert payload["selected_artifact_id"] == 10547681681
    assert payload["selected_checkpoint_sha256"] == (
        "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
    )
    assert payload["alternate_exact_candidate"]["checkpoint_sha256"] == (
        payload["selected_checkpoint_sha256"]
    )


def test_repaired_runtime_exposes_locked_checkpoint_loader() -> None:
    assert callable(load_locked_reconstruction)


def test_repaired_scientific_workflow_consumes_selected_artifact_not_replay() -> None:
    text = Path(
        ".github/workflows/exp323r-second-decay-convergence.yml"
    ).read_text(encoding="utf-8")
    assert "10547681681" in text
    assert "exp323r-reconstruction-candidate-35345351869-0" in text
    assert "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5" in text
    assert "reconstruct_exp322_decay_state" not in text
    assert "exp323r_run_arm.py" in text
