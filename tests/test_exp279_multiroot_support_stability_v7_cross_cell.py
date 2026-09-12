from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from nolane_ai.experiments.exp279_multiroot_support_stability_v7 import (
    FROZEN_ARM_GEOMETRY,
    FROZEN_OPTIMIZER,
    FROZEN_PROTOCOL_DIGEST,
    FROZEN_ROUTE_THRESHOLD,
    FROZEN_WORLD_GEOMETRY,
    PROBE_FOLDS,
    PROBE_REPLICATES,
    PROTOCOL_ID,
    ROOT_PREFIX,
    SCHEMA_BUDGET,
    expected_root_map,
    wilson_lower_bound,
)
from nolane_ai.experiments.exp279_multiroot_support_stability_v7_cross_cell import (
    SCHEMA_CROSS_CELL,
    classify_exp279_multiroot_support_stability_v7_cross_cell,
)
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "classify_exp279_multiroot_support_stability_v7_dev.py"
PAIR_EPISODES = PROBE_REPLICATES * FROZEN_WORLD_GEOMETRY["batch_size"]
FOLD_EPISODES = PAIR_EPISODES // PROBE_FOLDS


def _digest(receipt: dict) -> str:
    return canonical_sha256({key: value for key, value in receipt.items() if key != "artifact_digest"})


def _budget(train: int, classification: str, *, floor: float = 0.01) -> dict:
    root_map = expected_root_map(train)
    canonical_records = []
    pair_records = []
    fold_records = []
    for canonical_index, canonical_root in enumerate(root_map["canonical_roots"]):
        canonical_rescues = 0
        canonical_harms = 0
        canonical_both_success = 0
        canonical_both_failure = 0
        canonical_probe_supported = True
        canonical_folds_supported = True
        for probe_index in range(4):
            probe_root = root_map["probe_roots"][canonical_index * 4 + probe_index]
            probe_rescues = 0
            probe_harms = 0
            probe_both_success = 0
            probe_both_failure = 0
            fold_support_closed = True
            for fold_index in range(3):
                rescues = 1
                if classification == "MODEL_ROOT_SUPPORT_COLLAPSE" and canonical_index == 3:
                    rescues = 0
                elif (
                    classification == "PROBE_SUPPORT_INTERMITTENT"
                    and canonical_index == 2
                    and probe_index == 1
                    and fold_index == 1
                ):
                    rescues = 0
                both_success = 1
                harms = 0
                both_failure = FOLD_EPISODES - rescues - both_success
                nonrescue = FOLD_EPISODES - rescues
                rescue_supported = rescues > 0
                nonrescue_supported = nonrescue > 0
                fold_support_closed = fold_support_closed and rescue_supported and nonrescue_supported
                fold_records.append(
                    {
                        "canonical_index": canonical_index,
                        "probe_index": probe_index,
                        "canonical_root": canonical_root,
                        "probe_root": probe_root,
                        "fold": fold_index,
                        "episodes": FOLD_EPISODES,
                        "branch_rescues": rescues,
                        "branch_harms": harms,
                        "both_success": both_success,
                        "both_failure": both_failure,
                        "nonrescue_count": nonrescue,
                        "rescue_prevalence": rescues / FOLD_EPISODES,
                        "fold_rescue_supported": rescue_supported,
                        "fold_nonrescue_supported": nonrescue_supported,
                    }
                )
                probe_rescues += rescues
                probe_harms += harms
                probe_both_success += both_success
                probe_both_failure += both_failure
            probe_nonrescue = PAIR_EPISODES - probe_rescues
            pair_records.append(
                {
                    "canonical_index": canonical_index,
                    "probe_index": probe_index,
                    "canonical_root": canonical_root,
                    "probe_root": probe_root,
                    "episodes": PAIR_EPISODES,
                    "branch_rescues": probe_rescues,
                    "branch_harms": probe_harms,
                    "both_success": probe_both_success,
                    "both_failure": probe_both_failure,
                    "nonrescue_count": probe_nonrescue,
                    "rescue_prevalence": probe_rescues / PAIR_EPISODES,
                    "probe_rescue_supported": probe_rescues > 0,
                    "probe_nonrescue_supported": probe_nonrescue > 0,
                    "fold_support_closed": fold_support_closed,
                    "wilson_lower_bound": wilson_lower_bound(probe_rescues, PAIR_EPISODES),
                }
            )
            canonical_rescues += probe_rescues
            canonical_harms += probe_harms
            canonical_both_success += probe_both_success
            canonical_both_failure += probe_both_failure
            canonical_probe_supported = canonical_probe_supported and probe_rescues > 0 and probe_nonrescue > 0
            canonical_folds_supported = canonical_folds_supported and fold_support_closed
        canonical_episodes = PAIR_EPISODES * 4
        canonical_nonrescue = canonical_episodes - canonical_rescues
        canonical_records.append(
            {
                "canonical_index": canonical_index,
                "canonical_root": canonical_root,
                "final_canonical_digest": f"{canonical_index + 1:064x}",
                "artifact_digest": f"{canonical_index + 11:064x}",
                "canonical_total_episodes": canonical_episodes,
                "canonical_total_rescues": canonical_rescues,
                "canonical_total_harms": canonical_harms,
                "canonical_total_both_success": canonical_both_success,
                "canonical_total_both_failure": canonical_both_failure,
                "canonical_total_nonrescue": canonical_nonrescue,
                "canonical_has_any_rescue": canonical_rescues > 0,
                "canonical_all_probe_roots_supported": canonical_probe_supported,
                "canonical_all_folds_supported": canonical_folds_supported,
                "pooled_rescue_prevalence": canonical_rescues / canonical_episodes,
                "pooled_wilson_lower_bound": wilson_lower_bound(canonical_rescues, canonical_episodes),
            }
        )

    receipt = {
        "schema": SCHEMA_BUDGET,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": FROZEN_PROTOCOL_DIGEST,
        "code_digest": "c" * 64,
        "root_prefix": ROOT_PREFIX,
        "train_replicates": train,
        "route_config": {"canonical_threshold": FROZEN_ROUTE_THRESHOLD, "threshold_tuned": False},
        "world_geometry": dict(FROZEN_WORLD_GEOMETRY),
        "arm_geometry": dict(FROZEN_ARM_GEOMETRY),
        "optimizer": dict(FROZEN_OPTIMIZER),
        "probe_config": {
            "replicates": PROBE_REPLICATES,
            "folds": PROBE_FOLDS,
            "probe_roots_per_canonical": 4,
        },
        "training_rng_stream": "augmentation",
        "probe_rng_stream": "augmentation",
        "evaluation_rng_stream_used": False,
        "evaluation_targets_used": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
        "selector_trained": False,
        "cdd_distiller_trained": False,
        "branch_preview_selector_trained": False,
        "threshold_tuned": False,
        "root_map": root_map,
        "root_map_digest": canonical_sha256(root_map),
        "canonical_records": canonical_records,
        "pair_records": pair_records,
        "fold_records": fold_records,
        "canonical_prevalence_floor": floor,
        "train_cell_classification": classification,
        "total_episodes": PAIR_EPISODES * 16,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = _digest(receipt)
    return receipt


def _mutated_pair(mutator) -> dict[int, dict]:
    cells = {60: _budget(60, "SUPPORT_RECURRENT"), 120: _budget(120, "SUPPORT_RECURRENT")}
    for receipt in cells.values():
        mutator(receipt)
        receipt["artifact_digest"] = _digest(receipt)
    return cells


def test_v7_cross_cell_disposition_priority_and_authorization_scope() -> None:
    collapse = classify_exp279_multiroot_support_stability_v7_cross_cell(
        {60: _budget(60, "MODEL_ROOT_SUPPORT_COLLAPSE"), 120: _budget(120, "SUPPORT_RECURRENT")}
    )
    assert collapse["schema"] == SCHEMA_CROSS_CELL
    assert collapse["decision"] == "MODEL_ROOT_SUPPORT_COLLAPSE"
    assert collapse["successor_design_authorized"] is False
    assert collapse["future_probe_episodes_per_canonical_root"] is None

    intermittent = classify_exp279_multiroot_support_stability_v7_cross_cell(
        {60: _budget(60, "SUPPORT_RECURRENT", floor=0.02), 120: _budget(120, "PROBE_SUPPORT_INTERMITTENT", floor=0.01)}
    )
    assert intermittent["decision"] == "PROBE_SUPPORT_INTERMITTENT"
    assert intermittent["successor_design_authorized"] is False
    assert intermittent["future_probe_episodes_per_canonical_root"] == 459

    recurrent = classify_exp279_multiroot_support_stability_v7_cross_cell(
        {60: _budget(60, "SUPPORT_RECURRENT", floor=0.02), 120: _budget(120, "SUPPORT_RECURRENT", floor=0.01)}
    )
    assert recurrent["decision"] == "SUPPORT_RECURRENT"
    assert recurrent["successor_design_authorized"] is True
    assert recurrent["authorization_scope"] == "DESIGN_NEW_MECHANISM_COURT_ONLY"
    assert recurrent["future_probe_episodes_per_canonical_root"] == 459
    assert recurrent["fresh_evaluation_lineage_may_be_reserved"] is False
    assert recurrent["fresh_evaluation_lineage_consumed"] is False
    assert recurrent["confirmatory_data_consumed"] is False
    assert recurrent["challenge_materialized"] is False
    assert recurrent["promotion_claimed"] is False


def test_v7_cross_cell_recomputes_cell_classification_from_sufficient_statistics() -> None:
    wrong = _budget(60, "PROBE_SUPPORT_INTERMITTENT")
    wrong["train_cell_classification"] = "SUPPORT_RECURRENT"
    wrong["artifact_digest"] = _digest(wrong)
    with pytest.raises(ValueError, match="classification"):
        classify_exp279_multiroot_support_stability_v7_cross_cell(
            {60: wrong, 120: _budget(120, "SUPPORT_RECURRENT")}
        )


def test_v7_cross_cell_requires_exact_train60_train120_and_disjoint_roots() -> None:
    with pytest.raises(ValueError, match="60.*120|decision budgets"):
        classify_exp279_multiroot_support_stability_v7_cross_cell(
            {60: _budget(60, "SUPPORT_RECURRENT")}
        )

    cells = {60: _budget(60, "SUPPORT_RECURRENT"), 120: _budget(120, "SUPPORT_RECURRENT")}
    cells[120]["root_map"] = copy.deepcopy(cells[60]["root_map"])
    cells[120]["root_map_digest"] = canonical_sha256(cells[120]["root_map"])
    cells[120]["artifact_digest"] = _digest(cells[120])
    with pytest.raises(ValueError, match="root"):
        classify_exp279_multiroot_support_stability_v7_cross_cell(cells)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda r: r.__setitem__("protocol_digest", "0" * 64),
        lambda r: r["route_config"].__setitem__("canonical_threshold", 0.6),
        lambda r: r["world_geometry"].__setitem__("d_model", 65),
        lambda r: r.__setitem__("root_prefix", ROOT_PREFIX + "-wrong"),
        lambda r: r["root_map"]["probe_roots"].pop(),
        lambda r: r["probe_config"].__setitem__("replicates", 197),
        lambda r: r.__setitem__("probe_rng_stream", "evaluation"),
        lambda r: r.__setitem__("evidence_level", "EV-E3"),
        lambda r: r.__setitem__("fresh_evaluation_lineage_may_be_reserved", True),
        lambda r: r.__setitem__("confirmatory_data_consumed", True),
        lambda r: r.__setitem__("challenge_materialized", True),
        lambda r: r.__setitem__("promotion_claimed", True),
    ],
)
def test_v7_cross_cell_rejects_mutually_consistent_wrong_frozen_identity(mutator) -> None:
    cells = _mutated_pair(mutator)
    for receipt in cells.values():
        if "root_map" in receipt:
            receipt["root_map_digest"] = canonical_sha256(receipt["root_map"])
            receipt["artifact_digest"] = _digest(receipt)
    with pytest.raises(ValueError):
        classify_exp279_multiroot_support_stability_v7_cross_cell(cells)


def test_v7_cross_cell_sample_size_formula_is_frozen() -> None:
    assert math.ceil(math.log(0.01) / math.log(1.0 - 0.01)) == 459
    zero_floor = _budget(60, "PROBE_SUPPORT_INTERMITTENT", floor=0.0)
    result = classify_exp279_multiroot_support_stability_v7_cross_cell(
        {60: zero_floor, 120: _budget(120, "SUPPORT_RECURRENT", floor=0.01)}
    )
    assert result["decision"] == "PROBE_SUPPORT_INTERMITTENT"
    assert result["future_probe_episodes_per_canonical_root"] is None


def test_v7_cross_cell_cli_refuses_existing_output_before_loading_cells(tmp_path: Path) -> None:
    output = tmp_path / "cross.json"
    output.write_text("sentinel\n", encoding="utf-8")
    missing = tmp_path / "missing.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train60", str(missing),
            "--train120", str(missing),
            "--output", str(output),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_v7_cross_cell_cli_writes_deterministic_decision_receipt(tmp_path: Path) -> None:
    train60 = tmp_path / "train60.json"
    train120 = tmp_path / "train120.json"
    output = tmp_path / "cross.json"
    train60.write_text(json.dumps(_budget(60, "SUPPORT_RECURRENT"), sort_keys=True) + "\n", encoding="utf-8")
    train120.write_text(json.dumps(_budget(120, "SUPPORT_RECURRENT"), sort_keys=True) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train60", str(train60),
            "--train120", str(train120),
            "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    assert receipt["decision"] == "SUPPORT_RECURRENT"
    assert receipt["fresh_evaluation_lineage_may_be_reserved"] is False
    assert stdout["artifact_digest"] == receipt["artifact_digest"]
