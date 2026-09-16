from __future__ import annotations

import copy
import hashlib
import importlib
import json
from dataclasses import is_dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v017" / "exp319_preregistration_v1.json"
SIDECAR = ROOT / "protocols" / "v017" / "exp319_preregistration_v1.sha256"


def _contract_module():
    try:
        return importlib.import_module("nolane_ai.experiments.exp319_contract")
    except ModuleNotFoundError as exc:
        pytest.fail(f"EXP-319 contract module is not implemented yet: {exc}")


def test_exp319_contract_freezes_geometry_and_authorization_boundary() -> None:
    contract = _contract_module()

    assert contract.SCHEMA_VERSION == "EXP319-LEARNABILITY-FOUNDATION-DIAGNOSTIC-V1"
    assert contract.EXPERIMENT_ID == "EXP-319"
    assert contract.COMMON_GATING_EFFORT == 4
    assert contract.MAX_GENERATION_TOKENS == 96
    assert contract.TRAINING_EFFORT_CYCLE == (1, 2, 4, 8)
    assert contract.PRIMARY_ARMS == ("A_FIXED", "C_NRS_CORE")
    assert contract.DIAGNOSTIC_ARMS == ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")

    assert is_dataclass(contract.CONTRACT)
    assert contract.CONTRACT.stage_a.root == 0
    assert contract.CONTRACT.stage_a.examples_per_family == 8
    assert contract.CONTRACT.stage_a.total_examples == 32
    assert contract.CONTRACT.stage_a.learning_rates == (1e-4, 3e-4)
    assert contract.CONTRACT.stage_a.checkpoints == (128, 512, 1024)
    assert contract.CONTRACT.stage_a.max_chunk_steps == 256

    assert contract.CONTRACT.stage_b.roots == (1, 2, 3, 4)
    assert contract.CONTRACT.stage_b.train_examples_per_family == 128
    assert contract.CONTRACT.stage_b.total_train_examples == 512
    assert contract.CONTRACT.stage_b.iid_examples_per_family == 64
    assert contract.CONTRACT.stage_b.total_iid_examples == 256
    assert contract.CONTRACT.stage_b.checkpoints == (512, 1024, 2048)
    assert contract.CONTRACT.stage_b.max_chunk_steps == 512

    assert contract.CONTRACT.stage_c.roots == (1, 2, 3, 4)
    assert contract.CONTRACT.stage_c.heldout_examples_per_family == 128
    assert contract.CONTRACT.stage_c.total_heldout_examples == 512
    assert contract.CONTRACT.stage_c.gradient_updates == 0
    assert contract.CONTRACT.stage_c.decision_order == ("A_FIXED", "C_NRS_CORE")

    assert contract.AUTHORIZATION_FLAGS == {
        "exp302_implementation_authorized": False,
        "exp320_implementation_authorized": False,
        "scale_authorized": False,
        "30m_authorized": False,
        "100m_authorized": False,
    }


def test_exp319_contract_freezes_thresholds_and_selection_rules() -> None:
    contract = _contract_module()

    assert contract.CONTRACT.stage_a.floor.exact_match == 0.90
    assert contract.CONTRACT.stage_a.floor.answer_token_accuracy == 0.99
    assert contract.CONTRACT.stage_a.floor.eos_correctness == 0.95
    assert contract.CONTRACT.stage_a.floor.final_loss_fraction_of_initial == 0.25
    assert contract.CONTRACT.stage_a.floor.invalid_output_rate_max == 0.01
    assert contract.CONTRACT.stage_a.floor.nan_inf_events_max == 0
    assert contract.CONTRACT.stage_a.lr_tie_break == (
        "higher_step_1024_exact_match",
        "higher_step_1024_answer_token_accuracy",
        "lower_step_1024_answer_only_loss",
        "lower_learning_rate",
    )

    assert contract.CONTRACT.stage_b.floor.train_exact_match == 0.90
    assert contract.CONTRACT.stage_b.floor.iid_answer_token_accuracy == 0.95
    assert contract.CONTRACT.stage_b.floor.iid_family_balanced_exact_match == 0.25
    assert contract.CONTRACT.stage_b.floor.min_families_exact_at_least == 3
    assert contract.CONTRACT.stage_b.floor.per_family_exact_min == 0.10
    assert contract.CONTRACT.stage_b.floor.eos_correctness == 0.95
    assert contract.CONTRACT.stage_b.floor.invalid_output_rate_max == 0.01
    assert contract.CONTRACT.stage_b.cross_root_min_passes == 3
    assert contract.CONTRACT.stage_b.checkpoint_rule == (
        "earliest_checkpoint_passing_all_floors",
        "otherwise_higher_iid_family_balanced_exact_match",
        "then_higher_iid_answer_token_accuracy",
        "then_lower_iid_answer_only_loss",
        "then_earlier_checkpoint",
    )

    assert contract.CONTRACT.stage_c.floor.heldout_answer_token_accuracy == 0.85
    assert contract.CONTRACT.stage_c.floor.heldout_family_balanced_exact_match == 0.05
    assert contract.CONTRACT.stage_c.floor.min_families_nonzero_exact == 2
    assert contract.CONTRACT.stage_c.floor.eos_correctness == 0.90
    assert contract.CONTRACT.stage_c.floor.invalid_output_rate_max == 0.01
    assert contract.CONTRACT.stage_c.cross_root_min_passes == 3


def test_exp319_contract_freezes_closed_disposition_set() -> None:
    contract = _contract_module()

    assert contract.DISPOSITIONS == (
        "INVALID_DIAGNOSTIC",
        "TRAINING_STACK_NOT_LEARNABLE",
        "NRS_LOCAL_TRAINABILITY_FAIL",
        "SUPERVISED_FOUNDATION_UNDERTRAINED",
        "NRS_IID_GENERALIZATION_FLOOR_FAIL",
        "SUPERVISED_HELDOUT_FOUNDATION_FAIL",
        "NRS_HELDOUT_GENERALIZATION_FLOOR_FAIL",
        "FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY",
    )


def test_exp319_preregistration_is_canonical_and_digest_bound() -> None:
    contract = _contract_module()

    assert PROTOCOL.exists(), "EXP-319 preregistration JSON must exist before implementation proceeds"
    assert SIDECAR.exists(), "EXP-319 preregistration digest sidecar must exist"

    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload == contract.preregistration_payload()

    canonical = contract.canonical_json_bytes(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    assert SIDECAR.read_text(encoding="utf-8").strip() == digest

    mutated = copy.deepcopy(payload)
    mutated["stage_c"]["floor"]["heldout_family_balanced_exact_match"] = 0.0500001
    assert hashlib.sha256(contract.canonical_json_bytes(mutated)).hexdigest() != digest


def test_exp319_preregistration_keeps_exp301_kill_binding() -> None:
    contract = _contract_module()
    payload = contract.preregistration_payload()

    assert payload["scientific_boundary"]["exp301_disposition"] == "KILL_H_RD_01"
    assert payload["scientific_boundary"]["rescues_exp301_h_rd_01"] is False
    assert payload["stage_a"]["population_role"] == "tuning_sanity_only"
    assert payload["stage_b"]["starts_from_fresh_initialization"] is True
    assert payload["stage_b"]["may_reuse_stage_a_weights"] is False
    assert payload["stage_c"]["heldout_materialized_after_stage_b_selection"] is True
    assert payload["stage_c"]["gradient_updates"] == 0
    assert payload["stage_c"]["decision_order"] == ["A_FIXED", "C_NRS_CORE"]
    assert payload["authorizations"] == contract.AUTHORIZATION_FLAGS
