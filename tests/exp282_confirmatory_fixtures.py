from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from statistics import NormalDist, fmean, stdev

from nolane_ai.experiments.exp282_confirmatory_executor import _raw_digest, execute_exp282_confirmatory_open
from nolane_ai.protocol.evidence import canonical_sha256

CANONICAL_PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST = "a" * 64


def protocol(root_seed: str = "20260906") -> dict:
    return {
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "status": "FROZEN_V1",
        "rng": {
            "root_seed": root_seed,
            "streams": [
                "model_init",
                "data_order",
                "environment",
                "intervention",
                "augmentation",
                "replay_sampling",
                "controller_noise",
                "evaluation",
            ],
        },
        "global_sample_size_plan": {"familywise_alpha": 0.05},
        "experiments": [
            {
                "experiment_id": "EXP-282",
                "primary_endpoint": {"metric": "grounded_decision_accuracy", "direction": "higher"},
                "protected_endpoints": [
                    {"metric": "brier_score", "floor": "explicit_belief <= recurrent_hidden + 0.02"},
                    {
                        "metric": "accounted_flops",
                        "floor": "difference <= 0.05 relative unless included in primary cost normalization",
                    },
                ],
                "mesi": {"type": "absolute_gain", "value": 0.03, "unit": "accuracy_fraction"},
                "sample_size_plan": {"power_target": 0.90, "min_n": 32, "max_n": 128, "paired": True},
                "analysis_method": "paired accuracy difference with bootstrap CI plus calibration guard",
                "multiplicity_family": "BELIEF_STATE",
            }
        ],
    }


def _expand_pilot_to_32(execution: dict) -> dict:
    from nolane_ai.experiments.exp282_paired_runner import _artifact_digest

    execution = deepcopy(execution)
    rows = []
    recurrent_accuracy = 0.40
    explicit_accuracy = 0.43
    recurrent_brier = 0.20
    explicit_brier = 0.19
    accuracy_delta = explicit_accuracy - recurrent_accuracy
    brier_delta = explicit_brier - recurrent_brier
    for replicate in range(100, 132):
        rows.append(
            {
                "replicate": replicate,
                "batch_digest": hashlib.sha256(f"exp282-test-pilot-{replicate}".encode("utf-8")).hexdigest(),
                "recurrent_hidden": {
                    "grounded_decision_accuracy": recurrent_accuracy,
                    "brier_score": recurrent_brier,
                },
                "explicit_belief": {
                    "grounded_decision_accuracy": explicit_accuracy,
                    "brier_score": explicit_brier,
                },
                "explicit_minus_recurrent_accuracy": accuracy_delta,
                "explicit_minus_recurrent_brier": brier_delta,
            }
        )
    execution["evaluation"] = {
        "rng_stream": "evaluation",
        "start_replicate": 100,
        "replicates": len(rows),
        "per_replicate": rows,
        "aggregate": {
            "n": len(rows),
            "mean_accuracy_gain": accuracy_delta,
            "mean_brier_difference": brier_delta,
        },
    }
    execution["artifact_digest"] = _artifact_digest(execution)
    return execution


def prepared_chain(
    tmp_path: Path,
    *,
    analysis_code_digest: str = DEFAULT_FROZEN_ANALYSIS_CODE_DIGEST,
):
    from nolane_ai.experiments.exp282_confirmatory_execution_court import authorize_exp282_confirmatory_execution
    from nolane_ai.experiments.exp282_confirmatory_prep import _prep_digest
    from nolane_ai.experiments.exp282_paired_runner import run_exp282_paired_development
    from nolane_ai.experiments.exp282_reconstruction_court import authorize_exp282_confirmatory_reconstruction

    checkpoint_path = tmp_path / "paired.pt"
    execution = run_exp282_paired_development(
        root_seed="development-root-must-not-drive-confirmatory-worlds",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=1,
        eval_replicates=2,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=2,
        visibility_rate=0.5,
        noise_std=0.25,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        code_digest="c" * 64,
        checkpoint_path=checkpoint_path,
    )
    execution = _expand_pilot_to_32(execution)
    pilot_rows = execution["evaluation"]["per_replicate"]
    effects = [float(row["explicit_minus_recurrent_accuracy"]) for row in pilot_rows]
    pilot_ids = [int(row["replicate"]) for row in pilot_rows]
    paired_sd = stdev(effects)
    mean_effect = fmean(effects)
    z_alpha = NormalDist().inv_cdf(0.95)
    z_power = NormalDist().inv_cdf(0.90)
    reserved = list(range(132, 164))
    prep = {
        "schema": "NLM-EXP-282-CONFIRMATORY-OPEN-PREP-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "CONFIRMATORY_OPEN_PREPARED",
        "scope": "exp282-confirmatory-open-preparation-only",
        "confirmatory_data_consumed": False,
        "frozen_analysis": {
            "primary_endpoint": "grounded_decision_accuracy",
            "primary_direction": "higher",
            "mesi_absolute_gain": 0.03,
            "power_target": 0.90,
            "familywise_alpha": 0.05,
            "inference_tail": "one_sided_lower_bound",
            "analysis_method": "paired accuracy difference with bootstrap CI plus calibration guard",
            "multiplicity_family": "BELIEF_STATE",
            "brier_guard": "explicit_belief <= recurrent_hidden + 0.02",
            "compute_guard": "difference <= 0.05 relative unless included in primary cost normalization",
        },
        "pilot_summary": {
            "n": len(effects),
            "mean_accuracy_gain": mean_effect,
            "paired_sd": paired_sd,
            "paired_effect_digest": canonical_sha256({"replicate_ids": pilot_ids, "effects": effects}),
            "replicate_start": min(pilot_ids),
            "replicate_end": max(pilot_ids),
            "replicate_ids": pilot_ids,
            "training_replicate_ids": [0],
            "source_lane": "DEVELOPMENT_PILOT_ONLY",
        },
        "sample_size_freeze": {
            "method": "paired-normal-approximation-from-pilot-sd-v1",
            "unclamped_required_n": 1,
            "confirmatory_n": 32,
            "min_n": 32,
            "max_n": 128,
            "paired": True,
            "pilot_reuse_as_confirmatory": False,
            "planning_constants": {
                "alpha_tail": "one_sided_lower_bound",
                "z_alpha": z_alpha,
                "z_power": z_power,
                "formula": "ceil(((z_alpha + z_power) * paired_sd / mesi)^2), then apply frozen min_n without max_n truncation",
            },
        },
        "confirmatory_lineage": {
            "lane": "CONFIRMATORY_OPEN_RESERVED_UNCONSUMED",
            "pilot_reuse_forbidden": True,
            "reserved_replicate_ids": reserved,
            "seed_materialization_status": "NOT_EXECUTED",
        },
        "lineage": {
            "protocol_digest": CANONICAL_PROTOCOL_DIGEST,
            "execution_artifact_digest": execution["artifact_digest"],
            "arm_registry_digest": "r" * 64,
            "analysis_code_digest": analysis_code_digest,
            "paired_checkpoint_sha256": execution["checkpoint"]["checkpoint_sha256"],
        },
        "remaining_blockers": [
            "confirmatory-open execution has not consumed any reserved confirmatory data",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "prep_digest": "",
    }
    prep["prep_digest"] = _prep_digest(prep)
    authorization = authorize_exp282_confirmatory_execution(
        prep_artifact=prep,
        execution_code_digest="x" * 64,
    )
    reconstruction = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=authorization,
    )
    frozen_protocol = protocol()
    raw = execute_exp282_confirmatory_open(
        protocol=frozen_protocol,
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        checkpoint_path=checkpoint_path,
        executor_code_digest="z" * 64,
    )
    return frozen_protocol, prep, execution, reconstruction, raw


def set_raw_outcomes(raw: dict, *, accuracy_delta: float, brier_delta: float = -0.01) -> dict:
    raw = deepcopy(raw)
    recurrent_accuracy = 0.40
    explicit_accuracy = recurrent_accuracy + accuracy_delta
    recurrent_brier = 0.20
    explicit_brier = recurrent_brier + brier_delta
    for row in raw["per_replicate"]:
        row["recurrent_hidden"] = {
            "grounded_decision_accuracy": recurrent_accuracy,
            "correct": int(round(recurrent_accuracy * 100)),
            "total": 100,
            "brier_score": recurrent_brier,
            "brier_sum": recurrent_brier * 100,
            "brier_count": 100,
        }
        row["explicit_belief"] = {
            "grounded_decision_accuracy": explicit_accuracy,
            "correct": int(round(explicit_accuracy * 100)),
            "total": 100,
            "brier_score": explicit_brier,
            "brier_sum": explicit_brier * 100,
            "brier_count": 100,
        }
        row["explicit_minus_recurrent_accuracy"] = accuracy_delta
        row["explicit_minus_recurrent_brier"] = brier_delta
    raw["artifact_digest"] = _raw_digest(raw)
    return raw
