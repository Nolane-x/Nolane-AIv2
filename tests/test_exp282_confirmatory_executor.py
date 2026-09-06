from copy import deepcopy
from pathlib import Path
from statistics import NormalDist

import pytest

torch = pytest.importorskip("torch")

CANONICAL_PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"


def _protocol(root_seed: str = "20260906") -> dict:
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
        "experiments": [
            {
                "experiment_id": "EXP-282",
                "primary_endpoint": {"metric": "grounded_decision_accuracy", "direction": "higher"},
                "protected_endpoints": [
                    {"metric": "brier_score", "floor": "explicit_belief <= recurrent_hidden + 0.02"},
                    {"metric": "accounted_flops", "floor": "difference <= 0.05 relative unless included in primary cost normalization"},
                ],
                "mesi": {"type": "absolute_gain", "value": 0.03, "unit": "accuracy_fraction"},
                "sample_size_plan": {"power_target": 0.90, "min_n": 32, "max_n": 128, "paired": True},
                "analysis_method": "paired accuracy difference with bootstrap CI plus calibration guard",
                "multiplicity_family": "BELIEF_STATE",
            }
        ],
    }


def _prepared_chain(tmp_path: Path, *, protocol_digest: str = CANONICAL_PROTOCOL_DIGEST):
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
        protocol_digest=protocol_digest,
        code_digest="c" * 64,
        checkpoint_path=checkpoint_path,
    )

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
            "n": 32,
            "mean_accuracy_gain": 0.03,
            "paired_sd": 0.01,
            "paired_effect_digest": "e" * 64,
            "replicate_start": 100,
            "replicate_end": 131,
            "replicate_ids": list(range(100, 132)),
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
            "protocol_digest": protocol_digest,
            "execution_artifact_digest": execution["artifact_digest"],
            "arm_registry_digest": "r" * 64,
            "analysis_code_digest": "d" * 64,
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
    return _protocol(), protocol_digest, execution, reconstruction, checkpoint_path


def test_confirmatory_executor_uses_protocol_root_exact_reserved_ids_and_no_training(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_executor import (
        execute_exp282_confirmatory_open,
        validate_exp282_confirmatory_open_raw,
    )
    from nolane_ai.protocol.identity import file_sha256

    protocol, protocol_digest, execution, reconstruction, checkpoint_path = _prepared_chain(tmp_path)
    checkpoint_sha_before = file_sha256(checkpoint_path)
    raw = execute_exp282_confirmatory_open(
        protocol=protocol,
        protocol_digest=protocol_digest,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        checkpoint_path=checkpoint_path,
        executor_code_digest="z" * 64,
    )
    assert raw["schema"] == "NLM-EXP-282-CONFIRMATORY-OPEN-RAW-V1"
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["status"] == "CONFIRMATORY_OPEN_EXECUTED_UNANALYZED"
    assert raw["confirmatory_data_consumed"] is True
    assert raw["seed_materialization_status"] == "EXECUTED"
    assert raw["confirmatory_n"] == 32
    assert raw["reserved_replicate_ids"] == list(range(132, 164))
    assert [row["replicate"] for row in raw["per_replicate"]] == list(range(132, 164))
    assert raw["confirmatory_world_root_seed"] == "20260906"
    assert raw["confirmatory_world_root_seed"] != execution["root_seed"]
    assert all(row["world"]["rng_stream"] == "evaluation" for row in raw["per_replicate"])
    assert all(row["world"]["scope"] == "synthetic-exp282-partial-observability-confirmatory-open" for row in raw["per_replicate"])
    assert all(row["world"]["root_seed"] == "20260906" for row in raw["per_replicate"])
    assert file_sha256(checkpoint_path) == checkpoint_sha_before
    assert validate_exp282_confirmatory_open_raw(raw) == []


def test_confirmatory_executor_rejects_checkpoint_sha_mismatch_before_world_materialization(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_executor import execute_exp282_confirmatory_open

    protocol, protocol_digest, execution, reconstruction, checkpoint_path = _prepared_chain(tmp_path)
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        execute_exp282_confirmatory_open(
            protocol=protocol,
            protocol_digest=protocol_digest,
            paired_execution_artifact=execution,
            reconstruction_authorization=reconstruction,
            checkpoint_path=checkpoint_path,
            executor_code_digest="z" * 64,
        )


def test_confirmatory_executor_rejects_reconstruction_contract_drift_even_if_rehashed(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_executor import execute_exp282_confirmatory_open
    from nolane_ai.experiments.exp282_reconstruction_court import _reconstruction_digest
    from nolane_ai.protocol.evidence import canonical_sha256

    protocol, protocol_digest, execution, reconstruction, checkpoint_path = _prepared_chain(tmp_path)
    tampered = deepcopy(reconstruction)
    tampered["execution_contract"]["world_geometry"]["variables"] += 1
    tampered["execution_contract_digest"] = canonical_sha256(tampered["execution_contract"])
    tampered["reconstruction_digest"] = _reconstruction_digest(tampered)
    with pytest.raises(ValueError, match="execution contract"):
        execute_exp282_confirmatory_open(
            protocol=protocol,
            protocol_digest=protocol_digest,
            paired_execution_artifact=execution,
            reconstruction_authorization=tampered,
            checkpoint_path=checkpoint_path,
            executor_code_digest="z" * 64,
        )


def test_confirmatory_raw_validator_rejects_reordered_reserved_rows_after_rehash(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_executor import (
        _raw_digest,
        execute_exp282_confirmatory_open,
        validate_exp282_confirmatory_open_raw,
    )

    protocol, protocol_digest, execution, reconstruction, checkpoint_path = _prepared_chain(tmp_path)
    raw = execute_exp282_confirmatory_open(
        protocol=protocol,
        protocol_digest=protocol_digest,
        paired_execution_artifact=execution,
        reconstruction_authorization=reconstruction,
        checkpoint_path=checkpoint_path,
        executor_code_digest="z" * 64,
    )
    raw["per_replicate"][0], raw["per_replicate"][1] = raw["per_replicate"][1], raw["per_replicate"][0]
    raw["artifact_digest"] = _raw_digest(raw)
    errors = validate_exp282_confirmatory_open_raw(raw)
    assert "confirmatory raw replicate lineage must exactly match reserved IDs" in errors


def test_confirmatory_executor_rejects_noncanonical_frozen_protocol_digest_even_when_lineage_matches(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_executor import execute_exp282_confirmatory_open

    forged_digest = "f" * 64
    protocol, protocol_digest, execution, reconstruction, checkpoint_path = _prepared_chain(
        tmp_path,
        protocol_digest=forged_digest,
    )
    with pytest.raises(ValueError, match="canonical frozen protocol digest"):
        execute_exp282_confirmatory_open(
            protocol=protocol,
            protocol_digest=protocol_digest,
            paired_execution_artifact=execution,
            reconstruction_authorization=reconstruction,
            checkpoint_path=checkpoint_path,
            executor_code_digest="z" * 64,
        )
