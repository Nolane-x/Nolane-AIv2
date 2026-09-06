import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.training.multiseed import run_multiseed_heldout_development, validate_multiseed_heldout


def _run(seed_count: int = 2):
    return run_multiseed_heldout_development(
        config=NLMConfig.stage_a_pilot_tiny_for_tests(),
        root_seed="validator-multi",
        seeds=seed_count,
        train_steps=1,
        eval_batches=1,
        batch_size=2,
        variables=2,
        constraints=2,
        bootstrap_samples=50,
        protocol_digest="p",
        code_digest="c",
    )


def test_multiseed_runner_retains_every_seed_and_reports_uncertainty():
    result = run_multiseed_heldout_development(
        config=NLMConfig.stage_a_pilot_tiny_for_tests(),
        root_seed="multi",
        seeds=3,
        train_steps=2,
        eval_batches=2,
        batch_size=3,
        variables=3,
        constraints=2,
        bootstrap_samples=200,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    assert result["schema"] == "NLM-STAGE-A-MULTISEED-HELDOUT-DEV-V1"
    assert result["evidence_level"] == "EV-E2"
    assert result["decision"] == "UNVERIFIED"
    assert result["seed_count"] == 3
    assert result["successful_seed_count"] == 3
    assert result["failed_seed_count"] == 0
    assert result["aggregate_valid"] is True
    assert len(result["per_seed"]) == 3
    assert len({item["root_seed"] for item in result["per_seed"]}) == 3
    assert len({item["artifact_digest"] for item in result["per_seed"]}) == 3
    for metric, stats in result["metrics"].items():
        assert stats["n"] == 3
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
        assert 0.0 <= stats["positive_sign_fraction"] <= 1.0
        assert metric
    assert validate_multiseed_heldout(result) == []


def test_multiseed_runner_is_deterministic_for_same_root_seed():
    kwargs = dict(
        config=NLMConfig.stage_a_pilot_tiny_for_tests(),
        root_seed="same",
        seeds=2,
        train_steps=1,
        eval_batches=2,
        batch_size=2,
        variables=3,
        constraints=2,
        bootstrap_samples=100,
        protocol_digest="p",
        code_digest="c",
    )
    a = run_multiseed_heldout_development(**kwargs)
    b = run_multiseed_heldout_development(**kwargs)
    assert a["multiseed_digest"] == b["multiseed_digest"]
    assert a["metrics"] == b["metrics"]
    assert [item["artifact_digest"] for item in a["per_seed"]] == [item["artifact_digest"] for item in b["per_seed"]]


def test_multiseed_validator_refuses_seed_dropping_and_promotion():
    result = _run(seed_count=2)
    result["decision"] = "PROMOTE_TO_NEXT_STAGE"
    result["per_seed"].pop()
    errors = validate_multiseed_heldout(result)
    assert "multiseed development artifact cannot promote a neural claim" in errors
    assert "seed_count does not match per_seed outcomes" in errors
    assert "multiseed artifact digest mismatch" in errors


def test_multiseed_validator_rejects_reordered_or_malformed_failed_seed_outcomes():
    payload = _run(seed_count=2)
    payload["per_seed"] = list(reversed(payload["per_seed"]))
    errors = validate_multiseed_heldout(payload)
    assert "per_seed outcomes must retain every seed index in order" in errors

    payload = _run(seed_count=2)
    payload["per_seed"][1]["status"] = "FAILED"
    errors = validate_multiseed_heldout(payload)
    assert "failed per_seed outcome missing error_type" in errors
    assert "aggregate statistics cannot be valid when any seed failed" in errors


def test_multiseed_runner_retains_failed_seed_and_invalidates_all_aggregates(monkeypatch):
    import nolane_ai.training.multiseed as multiseed_module

    original = multiseed_module.run_heldout_development_evaluation

    def fail_one_seed(trainer, **kwargs):
        if trainer.curriculum.root_seed.endswith("seed=1"):
            raise RuntimeError("synthetic architecture-caused divergence")
        return original(trainer, **kwargs)

    monkeypatch.setattr(multiseed_module, "run_heldout_development_evaluation", fail_one_seed)
    payload = _run(seed_count=3)
    assert [item["status"] for item in payload["per_seed"]] == ["SUCCESS", "FAILED", "SUCCESS"]
    assert payload["failed_seed_count"] == 1
    assert payload["successful_seed_count"] == 2
    assert payload["aggregate_valid"] is False
    assert payload["per_seed"][1]["error_type"] == "RuntimeError"
    for stats in payload["metrics"].values():
        assert stats["analysis_status"] == "INVALID_DUE_TO_FAILED_SEED"
        assert stats["mean"] is None
        assert stats["ci_low"] is None
        assert stats["ci_high"] is None
    assert validate_multiseed_heldout(payload) == []
