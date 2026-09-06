import pytest

torch = pytest.importorskip("torch")


def _run(root_seed="paired-exp282"):
    from nolane_ai.experiments.exp282_paired_runner import run_exp282_paired_development

    return run_exp282_paired_development(
        root_seed=root_seed,
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=3,
        eval_replicates=4,
        eval_start_replicate=100,
        batch_size=3,
        timesteps=4,
        variables=3,
        visibility_rate=0.5,
        noise_std=0.35,
        lr=2e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )


def test_exp282_paired_runner_binds_identical_init_worlds_and_compute():
    result = _run()
    assert result["schema"] == "NLM-EXP-282-PAIRED-DEV-EVAL-V1"
    assert result["evidence_level"] == "EV-E2"
    assert result["decision"] == "UNVERIFIED"
    assert result["confirmatory_ready"] is False
    assert result["initial_state"]["functional_digest_match"] is True
    assert result["resource_match"]["parameter_match"] is True
    assert result["resource_match"]["observation_history_match"] is True
    assert result["resource_match"]["accounted_flop_match"] is True
    assert result["resource_match"]["relative_accounted_flop_difference"] == pytest.approx(0.0)
    assert result["training"]["rng_stream"] == "augmentation"
    assert result["evaluation"]["rng_stream"] == "evaluation"
    assert len(result["training"]["batch_digests"]) == 3
    assert len(result["evaluation"]["per_replicate"]) == 4
    assert len({row["batch_digest"] for row in result["evaluation"]["per_replicate"]}) == 4


def test_exp282_paired_runner_retains_primary_and_protected_metrics_per_replicate():
    result = _run("metrics")
    for row in result["evaluation"]["per_replicate"]:
        assert 0.0 <= row["recurrent_hidden"]["grounded_decision_accuracy"] <= 1.0
        assert 0.0 <= row["explicit_belief"]["grounded_decision_accuracy"] <= 1.0
        assert row["recurrent_hidden"]["brier_score"] >= 0.0
        assert row["explicit_belief"]["brier_score"] >= 0.0
        assert row["explicit_minus_recurrent_accuracy"] == pytest.approx(
            row["explicit_belief"]["grounded_decision_accuracy"]
            - row["recurrent_hidden"]["grounded_decision_accuracy"]
        )
        assert row["explicit_minus_recurrent_brier"] == pytest.approx(
            row["explicit_belief"]["brier_score"] - row["recurrent_hidden"]["brier_score"]
        )
    aggregate = result["evaluation"]["aggregate"]
    assert aggregate["n"] == 4
    assert "mean_accuracy_gain" in aggregate
    assert "mean_brier_difference" in aggregate
    assert result["primary_endpoint"]["metric"] == "grounded_decision_accuracy"
    assert result["protected_endpoints"]["brier_score_guard"] == "explicit_belief <= recurrent_hidden + 0.02"
    assert result["protected_endpoints"]["accounted_flops_guard"] == "relative difference <= 0.05"


def test_exp282_paired_runner_is_replay_deterministic():
    first = _run("replay")
    second = _run("replay")
    assert first["artifact_digest"] == second["artifact_digest"]
    assert first["training"]["batch_digests"] == second["training"]["batch_digests"]
    assert first["evaluation"]["per_replicate"] == second["evaluation"]["per_replicate"]
    assert first["final_state"] == second["final_state"]


def test_exp282_paired_runner_rejects_missing_provenance_or_invalid_counts():
    from nolane_ai.experiments.exp282_paired_runner import run_exp282_paired_development

    kwargs = dict(
        root_seed="invalid",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=1,
        eval_replicates=1,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=2,
        visibility_rate=0.5,
        noise_std=0.25,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p",
        code_digest="c",
    )
    with pytest.raises(ValueError, match="protocol_digest and code_digest"):
        run_exp282_paired_development(**{**kwargs, "protocol_digest": ""})
    with pytest.raises(ValueError, match="execution counts"):
        run_exp282_paired_development(**{**kwargs, "eval_replicates": 0})


def test_exp282_paired_artifact_validator_rejects_tampering_and_promotion():
    from nolane_ai.experiments.exp282_paired_runner import validate_exp282_paired_development

    result = _run("tamper")
    assert validate_exp282_paired_development(result) == []
    result["decision"] = "PROMOTE_TO_NEXT_STAGE"
    result["training"]["rng_stream"] = "evaluation"
    result["resource_match"]["accounted_flop_match"] = False
    errors = validate_exp282_paired_development(result)
    assert "EXP-282 paired development artifact cannot promote a claim" in errors
    assert "EXP-282 training stream must be augmentation" in errors
    assert "EXP-282 accounted FLOP match must be closed" in errors
    assert "EXP-282 paired artifact digest mismatch" in errors


def test_exp282_paired_artifact_validator_requires_complete_ordered_evaluation_lineage():
    from nolane_ai.experiments.exp282_paired_runner import validate_exp282_paired_development

    result = _run("lineage")
    result["evaluation"]["per_replicate"].pop(1)
    errors = validate_exp282_paired_development(result)
    assert "EXP-282 evaluation replicate count does not match raw lineage" in errors
    assert "EXP-282 paired artifact digest mismatch" in errors


def test_exp282_paired_runner_records_training_lineage_and_rejects_train_eval_overlap():
    from nolane_ai.experiments.exp282_paired_runner import run_exp282_paired_development

    result = _run("disjoint-lineage")
    assert result["training"]["start_replicate"] == 0
    train_ids = set(range(result["training"]["start_replicate"], result["training"]["start_replicate"] + result["training"]["replicates"]))
    eval_ids = {row["replicate"] for row in result["evaluation"]["per_replicate"]}
    assert train_ids.isdisjoint(eval_ids)

    with pytest.raises(ValueError, match="training and evaluation replicate lineages must be disjoint"):
        run_exp282_paired_development(
            root_seed="overlap",
            d_model=8,
            hidden_size=6,
            target_parameters=5_000,
            train_replicates=3,
            eval_replicates=2,
            eval_start_replicate=2,
            batch_size=2,
            timesteps=3,
            variables=2,
            visibility_rate=0.5,
            noise_std=0.25,
            lr=1e-3,
            weight_decay=0.0,
            protocol_digest="p",
            code_digest="c",
        )
