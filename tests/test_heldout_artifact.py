import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.training.curriculum import StageACurriculum
from nolane_ai.training.evaluation import run_heldout_development_evaluation
from nolane_ai.training.heldout_artifact import build_heldout_artifact, validate_heldout_artifact
from nolane_ai.training.pilot import StageAPilotTrainer, build_seeded_pilot_model, pilot_model_init_seed


def _make_artifact():
    root_seed = "artifact-contract"
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    trainer = StageAPilotTrainer(
        build_seeded_pilot_model(config, root_seed=root_seed),
        curriculum=StageACurriculum(root_seed=root_seed),
        lr=2e-3,
        weight_decay=0.01,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
        model_init_seed=pilot_model_init_seed(root_seed),
    )
    report = run_heldout_development_evaluation(
        trainer,
        train_steps=2,
        train_start_replicate=0,
        eval_start_replicate=100,
        eval_batches=3,
        batch_size=4,
        variables=5,
        constraints=3,
    )
    artifact = build_heldout_artifact(
        report,
        trainer=trainer,
        train_steps=2,
        train_start_replicate=0,
        eval_start_replicate=100,
        eval_batches=3,
        batch_size=4,
        variables=5,
        constraints=3,
    )
    return artifact


def test_heldout_artifact_binds_optimizer_geometry_and_class_support():
    artifact = _make_artifact()
    assert artifact["schema"] == "NLM-STAGE-A-HELDOUT-DEV-ARTIFACT-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    training = artifact["execution_contract"]["training"]
    evaluation = artifact["execution_contract"]["evaluation"]
    assert training["optimizer"]["type"] == "AdamW"
    assert training["optimizer"]["lr"] == pytest.approx(2e-3)
    assert training["optimizer"]["weight_decay"] == pytest.approx(0.01)
    assert training["rng_stream"] == "augmentation"
    assert evaluation["rng_stream"] == "evaluation"
    assert evaluation["batches"] == 3
    support = artifact["class_support"]["aggregate"]
    assert support["belief"]["positive"] + support["belief"]["negative"] == 3 * 4 * 5
    assert support["conflict"]["positive"] + support["conflict"]["negative"] == 3 * 4 * 3
    assert support["fidelity"]["positive"] + support["fidelity"]["negative"] == 3 * 4
    assert tuple(row["batch_digest"] for row in artifact["class_support"]["per_batch"]) == tuple(
        artifact["report"]["pre"]["batch_digests"]
    )
    assert validate_heldout_artifact(artifact) == []


def test_heldout_artifact_rejects_tampering_and_promotion():
    artifact = _make_artifact()
    artifact["decision"] = "PROMOTE_TO_NEXT_STAGE"
    artifact["class_support"]["aggregate"]["belief"]["positive"] += 1
    errors = validate_heldout_artifact(artifact)
    assert "held-out artifact cannot promote a neural claim" in errors
    assert "belief class-support count does not match evaluation geometry" in errors
    assert "held-out artifact digest mismatch" in errors
