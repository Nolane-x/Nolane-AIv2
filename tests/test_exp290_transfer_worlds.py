from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def _make_batch(*, replicate: int, rng_stream: str):
    from nolane_ai.experiments.exp290_transfer_worlds import Exp290TransferGenerator

    return Exp290TransferGenerator(
        root_seed="20260913-exp290-transfer-test-root",
    ).make_batch(
        replicate=replicate,
        rng_stream=rng_stream,
        device="cpu",
    )


def test_exp290_geometry_is_frozen_and_digest_stable() -> None:
    from nolane_ai.experiments.exp290_transfer_geometry import (
        EXP290_GEOMETRY,
        exp290_geometry_digest,
        exp290_geometry_manifest,
    )

    assert EXP290_GEOMETRY.canonical_indices == (0, 1, 2, 3)
    assert EXP290_GEOMETRY.train_replicates == 64
    assert EXP290_GEOMETRY.eval_replicates == 32
    assert EXP290_GEOMETRY.eval_start_replicate == 10000
    assert EXP290_GEOMETRY.batch_size == 8
    assert EXP290_GEOMETRY.variables == 8
    assert EXP290_GEOMETRY.restarts == 4
    assert EXP290_GEOMETRY.max_search_steps == 24
    assert EXP290_GEOMETRY.timesteps == 4
    assert EXP290_GEOMETRY.d_model == 64
    assert EXP290_GEOMETRY.hidden_size == 48
    assert EXP290_GEOMETRY.target_parameters == 500000
    assert EXP290_GEOMETRY.noise_std == pytest.approx(0.05)
    assert EXP290_GEOMETRY.lr == pytest.approx(0.002)
    assert EXP290_GEOMETRY.weight_decay == pytest.approx(0.0)
    manifest = exp290_geometry_manifest()
    assert manifest["root_prefix"] == "20260913-exp290-learned-clause-transfer-v1-dev"
    assert manifest["canonical_indices"] == [0, 1, 2, 3]
    assert manifest["authority_scope"] == "DEVELOPMENT_EV_E2_ONLY"
    digest = exp290_geometry_digest()
    assert isinstance(digest, str)
    assert len(digest) == 64
    int(digest, 16)


def test_exp290_evaluation_pair_regenerates_byte_identically() -> None:
    first = _make_batch(replicate=10000, rng_stream="evaluation")
    second = _make_batch(replicate=10000, rng_stream="evaluation")

    assert first.digest == second.digest
    assert first.metadata == second.metadata
    assert torch.equal(first.source_surface_events, second.source_surface_events)
    assert torch.equal(first.target_surface_events, second.target_surface_events)
    assert torch.equal(first.source_variable_states, second.source_variable_states)
    assert torch.equal(first.target_variable_states, second.target_variable_states)


def test_exp290_augmentation_and_evaluation_lineages_are_disjoint() -> None:
    augmentation = _make_batch(replicate=0, rng_stream="augmentation")
    evaluation = _make_batch(replicate=10000, rng_stream="evaluation")

    assert augmentation.digest != evaluation.digest
    assert augmentation.metadata["rng_stream"] == "augmentation"
    assert evaluation.metadata["rng_stream"] == "evaluation"
    assert augmentation.metadata["seed"] != evaluation.metadata["seed"]


def test_exp290_source_target_namespaces_are_disjoint_and_isomorphic() -> None:
    batch = _make_batch(replicate=10000, rng_stream="evaluation")

    assert len(batch.metadata["pairs"]) == 8
    for pair in batch.metadata["pairs"]:
        source_names = pair["source_surface_names"]
        target_names = pair["target_surface_names"]
        mapping = pair["hidden_source_to_target_index"]
        assert len(source_names) == 8
        assert len(target_names) == 8
        assert set(source_names).isdisjoint(target_names)
        assert pair["surface_identity_overlap"] is False
        assert sorted(mapping) == list(range(8))
        assert pair["structurally_isomorphic"] is True
        assert pair["target_local_clause_learning_enabled"] is False


def test_exp290_worlds_freeze_two_literal_transfer_opportunities() -> None:
    batch = _make_batch(replicate=10000, rng_stream="evaluation")

    for pair in batch.metadata["pairs"]:
        source_opportunities = pair["source_dead_end_opportunities"]
        transfer_opportunities = pair["target_transfer_opportunities"]
        assert source_opportunities
        assert transfer_opportunities
        assert all(len(item["source_clause"]) == 2 for item in source_opportunities)
        assert all(len(item["mapped_target_clause"]) == 2 for item in transfer_opportunities)
        assert pair["transfer_opportunity_manifest_frozen_before_mode_execution"] is True


def test_exp290_evaluator_mapping_is_not_delivered_to_learned_mode() -> None:
    batch = _make_batch(replicate=10000, rng_stream="evaluation")

    for pair in batch.metadata["pairs"]:
        assert pair["hidden_correspondence_delivered_to_learned_mode"] is False
        assert pair["target_validity_truth_delivered_to_learned_mode"] is False
        assert pair["transfer_opportunity_manifest_delivered_to_learned_mode"] is False
        assert pair["surface_names_delivered_as_neural_features"] is False


def test_exp290_rejects_unknown_rng_stream() -> None:
    from nolane_ai.experiments.exp290_transfer_worlds import Exp290TransferGenerator

    generator = Exp290TransferGenerator(root_seed="20260913-exp290-transfer-test-root")
    with pytest.raises(ValueError, match="augmentation or evaluation"):
        generator.make_batch(replicate=0, rng_stream="confirmatory", device="cpu")
