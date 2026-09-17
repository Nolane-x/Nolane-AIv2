from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "EXP319-LEARNABILITY-FOUNDATION-DIAGNOSTIC-V1"
EXPERIMENT_ID = "EXP-319"
COMMON_GATING_EFFORT = 4
MAX_GENERATION_TOKENS = 96
TRAINING_EFFORT_CYCLE = (1, 2, 4, 8)
DIAGNOSTIC_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
PRIMARY_ARMS = ("A_FIXED", "C_NRS_CORE")
TASK_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
DISPOSITIONS = (
    "INVALID_DIAGNOSTIC",
    "TRAINING_STACK_NOT_LEARNABLE",
    "NRS_LOCAL_TRAINABILITY_FAIL",
    "SUPERVISED_FOUNDATION_UNDERTRAINED",
    "NRS_IID_GENERALIZATION_FLOOR_FAIL",
    "SUPERVISED_HELDOUT_FOUNDATION_FAIL",
    "NRS_HELDOUT_GENERALIZATION_FLOOR_FAIL",
    "FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY",
)
AUTHORIZATION_FLAGS = {
    "exp302_implementation_authorized": False,
    "exp320_implementation_authorized": False,
    "scale_authorized": False,
    "30m_authorized": False,
    "100m_authorized": False,
}


@dataclass(frozen=True)
class StageAFloor:
    exact_match: float = 0.90
    answer_token_accuracy: float = 0.99
    eos_correctness: float = 0.95
    final_loss_fraction_of_initial: float = 0.25
    invalid_output_rate_max: float = 0.01
    nan_inf_events_max: int = 0


@dataclass(frozen=True)
class StageAContract:
    root: int = 0
    examples_per_family: int = 8
    total_examples: int = 32
    learning_rates: tuple[float, ...] = (1e-4, 3e-4)
    checkpoints: tuple[int, ...] = (128, 512, 1024)
    max_chunk_steps: int = 256
    floor: StageAFloor = StageAFloor()
    lr_tie_break: tuple[str, ...] = (
        "higher_step_1024_exact_match",
        "higher_step_1024_answer_token_accuracy",
        "lower_step_1024_answer_only_loss",
        "lower_learning_rate",
    )


@dataclass(frozen=True)
class StageBFloor:
    train_exact_match: float = 0.90
    iid_answer_token_accuracy: float = 0.95
    iid_family_balanced_exact_match: float = 0.25
    min_families_exact_at_least: int = 3
    per_family_exact_min: float = 0.10
    eos_correctness: float = 0.95
    invalid_output_rate_max: float = 0.01


@dataclass(frozen=True)
class StageBContract:
    roots: tuple[int, ...] = (1, 2, 3, 4)
    train_examples_per_family: int = 128
    total_train_examples: int = 512
    iid_examples_per_family: int = 64
    total_iid_examples: int = 256
    checkpoints: tuple[int, ...] = (512, 1024, 2048)
    max_chunk_steps: int = 512
    floor: StageBFloor = StageBFloor()
    cross_root_min_passes: int = 3
    checkpoint_rule: tuple[str, ...] = (
        "earliest_checkpoint_passing_all_floors",
        "otherwise_higher_iid_family_balanced_exact_match",
        "then_higher_iid_answer_token_accuracy",
        "then_lower_iid_answer_only_loss",
        "then_earlier_checkpoint",
    )


@dataclass(frozen=True)
class StageCFloor:
    heldout_answer_token_accuracy: float = 0.85
    heldout_family_balanced_exact_match: float = 0.05
    min_families_nonzero_exact: int = 2
    eos_correctness: float = 0.90
    invalid_output_rate_max: float = 0.01


@dataclass(frozen=True)
class StageCContract:
    roots: tuple[int, ...] = (1, 2, 3, 4)
    heldout_examples_per_family: int = 128
    total_heldout_examples: int = 512
    gradient_updates: int = 0
    decision_order: tuple[str, ...] = PRIMARY_ARMS
    floor: StageCFloor = StageCFloor()
    cross_root_min_passes: int = 3


@dataclass(frozen=True)
class Exp319Contract:
    stage_a: StageAContract = StageAContract()
    stage_b: StageBContract = StageBContract()
    stage_c: StageCContract = StageCContract()


CONTRACT = Exp319Contract()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def preregistration_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "program": "NLM V0.17 10M Native Recursive Substrate",
        "scientific_boundary": {
            "lineage_role": "scientifically_distinct_post_exp301_learnability_diagnostic",
            "exp301_disposition": "KILL_H_RD_01",
            "rescues_exp301_h_rd_01": False,
            "exp301_outcome_reinterpreted": False,
        },
        "common_evaluation": {
            "gating_effort": COMMON_GATING_EFFORT,
            "max_generation_tokens": MAX_GENERATION_TOKENS,
            "decoding": "greedy",
            "answer_token_accuracy_includes_eos": True,
            "exact_answer_verifier": "deterministic_task_semantics",
        },
        "arms": {
            "diagnostic": list(DIAGNOSTIC_ARMS),
            "primary": list(PRIMARY_ARMS),
            "resident_trainable_parameter_ceiling": 10_000_000,
            "architecture_geometry_may_change": False,
        },
        "task_families": list(TASK_FAMILIES),
        "training": {
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "training_effort_cycle": list(TRAINING_EFFORT_CYCLE),
            "byte_tokenizer_unchanged": True,
            "answer_only_loss_unchanged": True,
        },
        "stage_a": {
            "population_role": "tuning_sanity_only",
            "root": CONTRACT.stage_a.root,
            "families": 4,
            "examples_per_family": CONTRACT.stage_a.examples_per_family,
            "total_examples": CONTRACT.stage_a.total_examples,
            "learning_rates": list(CONTRACT.stage_a.learning_rates),
            "checkpoints": list(CONTRACT.stage_a.checkpoints),
            "max_chunk_steps": CONTRACT.stage_a.max_chunk_steps,
            "floor": asdict(CONTRACT.stage_a.floor),
            "lr_tie_break": list(CONTRACT.stage_a.lr_tie_break),
            "confirmatory_evidence": False,
        },
        "stage_b": {
            "population_role": "confirmatory_iid_capability_floor",
            "roots": list(CONTRACT.stage_b.roots),
            "primary_arms": list(PRIMARY_ARMS),
            "starts_from_fresh_initialization": True,
            "may_reuse_stage_a_weights": False,
            "train_examples_per_family": CONTRACT.stage_b.train_examples_per_family,
            "total_train_examples": CONTRACT.stage_b.total_train_examples,
            "iid_examples_per_family": CONTRACT.stage_b.iid_examples_per_family,
            "total_iid_examples": CONTRACT.stage_b.total_iid_examples,
            "checkpoints": list(CONTRACT.stage_b.checkpoints),
            "max_chunk_steps": CONTRACT.stage_b.max_chunk_steps,
            "learning_rate_source": "frozen_stage_a_per_arm_selection",
            "floor": asdict(CONTRACT.stage_b.floor),
            "cross_root_min_passes": CONTRACT.stage_b.cross_root_min_passes,
            "checkpoint_rule": list(CONTRACT.stage_b.checkpoint_rule),
        },
        "stage_c": {
            "population_role": "confirmatory_generator_heldout_transfer",
            "roots": list(CONTRACT.stage_c.roots),
            "primary_arms": list(PRIMARY_ARMS),
            "heldout_examples_per_family": CONTRACT.stage_c.heldout_examples_per_family,
            "total_heldout_examples": CONTRACT.stage_c.total_heldout_examples,
            "heldout_generator_template_shift_required": True,
            "heldout_materialized_after_stage_b_selection": True,
            "run_bound_challenge_beacon_required": True,
            "gradient_updates": CONTRACT.stage_c.gradient_updates,
            "gating_effort": COMMON_GATING_EFFORT,
            "decision_order": list(CONTRACT.stage_c.decision_order),
            "floor": asdict(CONTRACT.stage_c.floor),
            "cross_root_min_passes": CONTRACT.stage_c.cross_root_min_passes,
        },
        "decision_tree": {
            "causal_order": list(DISPOSITIONS),
            "stage_c_control_first": True,
            "architecture_ranking_authorized": False,
        },
        "execution": {
            "runner": "ubuntu-latest",
            "device": "cpu",
            "larger_runner_required": False,
            "immutable_continuations_required": True,
            "stage_a_max_chunk_steps": CONTRACT.stage_a.max_chunk_steps,
            "stage_b_max_chunk_steps": CONTRACT.stage_b.max_chunk_steps,
            "rerun_second_accepted_artifact_for_same_chain_position_allowed": False,
        },
        "authorizations": dict(AUTHORIZATION_FLAGS),
    }
