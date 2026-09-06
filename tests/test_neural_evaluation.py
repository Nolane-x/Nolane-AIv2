from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.training.curriculum import StageACurriculum
from nolane_ai.training.evaluation import (
    evaluate_stage_a,
    run_heldout_development_evaluation,
    validate_heldout_report,
)
from nolane_ai.training.pilot import (
    StageAPilotTrainer,
    build_seeded_pilot_model,
    pilot_model_init_seed,
)


def test_evaluation_stream_is_deterministically_separate_from_training_stream():
    curriculum = StageACurriculum(root_seed="heldout")
    train = curriculum.make_batch(
        replicate=4, batch_size=4, variables=5, constraints=3, d_model=16
    )
    evaluate = curriculum.make_batch(
        replicate=4,
        batch_size=4,
        variables=5,
        constraints=3,
        d_model=16,
        rng_stream="evaluation",
    )
    evaluate_again = curriculum.make_batch(
        replicate=4,
        batch_size=4,
        variables=5,
        constraints=3,
        d_model=16,
        rng_stream="evaluation",
    )
    assert train.metadata["rng_stream"] == "augmentation"
    assert evaluate.metadata["rng_stream"] == "evaluation"
    assert train.digest != evaluate.digest
    assert evaluate.digest == evaluate_again.digest
    assert torch.equal(evaluate.batch.variable_states, evaluate_again.batch.variable_states)


def test_neural_evaluator_reports_bounded_metrics_and_does_not_mutate_model():
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    model = build_seeded_pilot_model(config, root_seed="eval-model")
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "capacity_reserve" not in name
    }
    result = evaluate_stage_a(
        model,
        curriculum=StageACurriculum(root_seed="eval-data"),
        start_replicate=100,
        batches=3,
        batch_size=4,
        variables=5,
        constraints=3,
    )
    assert result.rng_stream == "evaluation"
    assert result.examples == 12
    assert len(result.batch_digests) == 3
    assert len(result.per_batch) == 3
    assert tuple(item.batch_digest for item in result.per_batch) == result.batch_digests
    for value in (
        result.belief_accuracy,
        result.belief_brier,
        result.conflict_balanced_accuracy,
        result.conflict_brier,
        result.fidelity_balanced_accuracy,
        result.fidelity_brier,
    ):
        assert 0.0 <= value <= 1.0
    for name, parameter in model.named_parameters():
        if name in before:
            assert torch.equal(parameter.detach(), before[name])


def test_heldout_pre_post_report_reuses_exact_evaluation_worlds_and_binds_lineage():
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    root_seed = "heldout-report"
    model = build_seeded_pilot_model(config, root_seed=root_seed)
    curriculum = StageACurriculum(root_seed=root_seed)
    trainer = StageAPilotTrainer(
        model,
        curriculum=curriculum,
        lr=3e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
        model_init_seed=pilot_model_init_seed(root_seed),
    )
    report = run_heldout_development_evaluation(
        trainer,
        train_steps=4,
        train_start_replicate=0,
        eval_start_replicate=100,
        eval_batches=3,
        batch_size=4,
        variables=4,
        constraints=3,
    )
    assert report.schema == "NLM-STAGE-A-HELDOUT-DEV-EVAL-V1"
    assert report.evidence_level == "EV-E2"
    assert report.decision == "UNVERIFIED"
    assert report.training_stream == "augmentation"
    assert report.evaluation_stream == "evaluation"
    assert report.pre.batch_digests == report.post.batch_digests
    assert len(report.paired_batch_deltas) == 3
    assert tuple(item.batch_digest for item in report.paired_batch_deltas) == report.pre.batch_digests
    assert report.model_before_digest != report.model_after_digest
    assert report.protocol_digest == "p" * 64
    assert report.code_digest == "c" * 64
    assert report.config_digest == trainer.config_digest
    assert report.model_init_seed == trainer.model_init_seed
    assert report.initial_model_digest == trainer.initial_model_digest
    assert report.report_digest
    assert validate_heldout_report(asdict(report)) == []


def test_heldout_report_refuses_promotion_and_stream_aliasing():
    invalid = {
        "schema": "NLM-STAGE-A-HELDOUT-DEV-EVAL-V1",
        "evidence_level": "EV-E3",
        "decision": "PROMOTE_TO_NEXT_STAGE",
        "training_stream": "augmentation",
        "evaluation_stream": "augmentation",
        "protocol_digest": "p",
        "code_digest": "c",
        "config_digest": "cfg",
        "model_init_seed": 1,
        "initial_model_digest": "i",
        "model_before_digest": "b",
        "model_after_digest": "a",
        "pre": {"batch_digests": ["x"]},
        "post": {"batch_digests": ["x"]},
        "report_digest": "r",
    }
    errors = validate_heldout_report(invalid)
    assert "held-out development report cannot claim EV-E3+" in errors
    assert "held-out development report cannot promote a neural claim" in errors
    assert "training and evaluation RNG streams must be distinct" in errors


def test_heldout_report_detects_digest_tampering_and_paired_lineage_drift():
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    root_seed = "tamper-report"
    trainer = StageAPilotTrainer(
        build_seeded_pilot_model(config, root_seed=root_seed),
        curriculum=StageACurriculum(root_seed=root_seed),
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p",
        code_digest="c",
        model_init_seed=pilot_model_init_seed(root_seed),
    )
    report = asdict(
        run_heldout_development_evaluation(
            trainer,
            train_steps=2,
            train_start_replicate=0,
            eval_start_replicate=50,
            eval_batches=2,
            batch_size=3,
            variables=3,
            constraints=2,
        )
    )
    report["post"]["batch_digests"] = ("tampered",) + tuple(
        report["post"]["batch_digests"][1:]
    )
    errors = validate_heldout_report(report)
    assert "pre/post evaluation must reuse exact held-out batch digests" in errors
    assert "held-out report digest mismatch" in errors
