from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.protocol.schema import load_and_validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_SHA = ROOT / "protocols" / "stage_a_v1.sha256"
EXP290_MANIFEST = ROOT / "protocols" / "exp290_structural_clause_transfer_v1.json"
EXP290_MANIFEST_SHA = ROOT / "protocols" / "exp290_structural_clause_transfer_v1.sha256"
STAGE_A_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"


def _module(name: str):
    return importlib.import_module(name)


def _stage_a_digest() -> str:
    spec = load_and_validate_protocol(PROTOCOL)
    assert spec.protocol_id == "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert digest == PROTOCOL_SHA.read_text(encoding="utf-8").strip()
    return digest


def test_exp290_geometry_module_and_frozen_manifest_exist() -> None:
    geometry_mod = _module("nolane_ai.experiments.exp290_development_geometry")
    geometry, digest = geometry_mod.load_exp290_development_geometry(
        EXP290_MANIFEST,
        EXP290_MANIFEST_SHA,
        protocol_digest=_stage_a_digest(),
    )
    assert _stage_a_digest() == STAGE_A_DIGEST
    assert geometry == {
        "batch_size": 8,
        "canonical_indices": [0, 1, 2, 3],
        "d_model": 64,
        "decoys": 3,
        "eval_replicates": 32,
        "eval_start_replicate": 40000,
        "hidden_size": 48,
        "lr": 0.002,
        "max_search_steps": 24,
        "noise_std": 0.05,
        "restarts": 4,
        "root_prefix": "20260913-exp290-structural-clause-transfer-v1-dev",
        "target_parameters": 500000,
        "timesteps": 4,
        "train_replicates": 64,
        "variables": 8,
        "weight_decay": 0.0,
    }
    assert hashlib.sha256(EXP290_MANIFEST.read_bytes()).hexdigest() == digest


def test_exp290_manifest_freezes_development_only_authority_and_parent_safety() -> None:
    _module("nolane_ai.experiments.exp290_development_geometry")
    payload = json.loads(EXP290_MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "NLM-EXP-290-STRUCTURAL-CLAUSE-TRANSFER-DEVELOPMENT-V1"
    assert payload["experiment_id"] == "EXP-290"
    assert payload["authority_scope"] == "DEVELOPMENT_EV_E2_ONLY"
    assert payload["protocol_digest"] == STAGE_A_DIGEST
    assert payload["predeclaration"] == "BEFORE_AUTHORITATIVE_DEVELOPMENT_RUN"
    assert payload["scientific_evidence_eligible"] is False
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["stage_a_protocol_modified"] is False
    assert payload["surface_randomization"] == {
        "event_order_randomized": True,
        "value_labels_remapped": False,
        "variable_identity_randomized": True,
        "variable_order_randomized": True,
    }
    assert payload["capture_threshold"] == 0.50
    assert payload["protected_endpoints"] == {
        "valid_state_overprune_rate_ceiling": 0.005,
        "verified_solution_rate_floor_delta": -0.01,
    }
    assert payload["successor_scope_if_recurrent"] == "DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY"
    assert "beacon" not in payload
    assert "challenge_seed" not in payload


def test_exp290_train_and_evaluation_lineages_are_disjoint() -> None:
    geometry_mod = _module("nolane_ai.experiments.exp290_development_geometry")
    geometry, _ = geometry_mod.load_exp290_development_geometry(
        EXP290_MANIFEST,
        EXP290_MANIFEST_SHA,
        protocol_digest=_stage_a_digest(),
    )
    training = set(range(geometry["train_replicates"]))
    evaluation = set(range(geometry["eval_start_replicate"], geometry["eval_start_replicate"] + geometry["eval_replicates"]))
    assert training.isdisjoint(evaluation)


def test_exp290_pair_generator_is_deterministic_and_uses_nonidentity_target_surface() -> None:
    worlds = _module("nolane_ai.experiments.exp290_transfer_worlds")
    generator = worlds.Exp290TransferGenerator(root_seed="exp290-test-root")
    kwargs = dict(replicate=7, batch_size=3, timesteps=4, restarts=4, variables=8, decoys=3, d_model=64, noise_std=0.05, rng_stream="augmentation")
    first = generator.make_pair(**kwargs)
    second = generator.make_pair(**kwargs)
    assert first.digest == second.digest
    assert torch.equal(first.source.surface_events, second.source.surface_events)
    assert torch.equal(first.target.surface_events, second.target.surface_events)
    assert torch.equal(first.source.variable_states, second.source.variable_states)
    assert torch.equal(first.target.variable_states, second.target.variable_states)
    assert first.metadata["latent_problem_digest"] == second.metadata["latent_problem_digest"]
    assert len(first.metadata["episodes"]) == 3
    for episode in first.metadata["episodes"]:
        permutation = episode["evaluator_only_source_to_target_variable_permutation"]
        assert permutation != list(range(8))
        assert sorted(permutation) == list(range(8))
        assert episode["mapping_delivered_to_learned_mode"] is False
        assert episode["value_labels_remapped"] is False
        assert episode["source_world_id"] != episode["target_world_id"]
        assert episode["source_problem_digest"] != episode["target_problem_digest"]
        assert episode["latent_problem_digest"] == first.metadata["latent_problem_digest"]


def test_exp290_pair_generator_requires_permitted_rng_stream() -> None:
    worlds = _module("nolane_ai.experiments.exp290_transfer_worlds")
    generator = worlds.Exp290TransferGenerator(root_seed="exp290-test-root")
    with pytest.raises(ValueError, match="augmentation or evaluation"):
        generator.make_pair(replicate=0, batch_size=1, timesteps=4, restarts=4, variables=8, decoys=3, d_model=64, noise_std=0.05, rng_stream="challenge")


def test_exp290_transfer_model_scores_only_visible_source_and_target_state() -> None:
    arms = _module("nolane_ai.experiments.matched_clause_transfer_arms")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = arms.build_matched_exp290_model(d_model=64, hidden_size=48, target_parameters=500000, device="cpu")
    scores = model.score_transfer(
        source_variable_states=torch.randn(2, 8, 64),
        target_variable_states=torch.randn(2, 8, 64),
        source_variable_index=torch.tensor([1, 6], dtype=torch.long),
        literal_value=torch.tensor([0, 1], dtype=torch.long),
    )
    assert scores.shape == (2, 8)
    assert torch.isfinite(scores).all()


def test_exp290_transfer_top1_tie_break_is_ascending_index() -> None:
    arms = _module("nolane_ai.experiments.matched_clause_transfer_arms")
    scores = torch.tensor([[0.1, 0.7, 0.7, 0.2], [1.0, 1.0, 0.5, 0.1]], dtype=torch.float32)
    assert arms.deterministic_transfer_top1(scores).tolist() == [1, 0]


def test_exp290_model_parameter_and_compute_audit_is_mode_invariant() -> None:
    arms = _module("nolane_ai.experiments.matched_clause_transfer_arms")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = arms.build_matched_exp290_model(d_model=64, hidden_size=48, target_parameters=500000, device="cpu")
    audit = arms.audit_exp290_transfer_model(model, variables=8, source_clause_slots=3)
    assert audit["total_parameters"] == 500000
    assert audit["functional_parameters"] > 0
    assert audit["optimizer_visible_parameters"] == audit["functional_parameters"]
    assert audit["mode_parameter_inventory_equal"] is True
    assert audit["transfer_scorer_executed_in_all_modes"] is True
    assert audit["per_source_clause_transfer_scoring_flops"] > 0
    assert audit["transferred_slot_comparisons_per_target_query"] == 3


def test_exp290_stage_a_protocol_does_not_claim_exp290_confirmatory_authority() -> None:
    spec = load_and_validate_protocol(PROTOCOL)
    assert "EXP-290" not in set(spec.experiment_ids)
    assert _stage_a_digest() == STAGE_A_DIGEST
