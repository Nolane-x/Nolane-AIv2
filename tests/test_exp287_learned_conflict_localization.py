from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp287_development_geometry import (
    CANONICAL_INDICES,
    SCHEMA as GEOMETRY_SCHEMA,
    load_exp287_development_geometry,
)
from nolane_ai.experiments.exp287_conflict_worlds import Exp287ConflictGenerator
from nolane_ai.experiments.matched_conflict_localizer_arms import (
    LEARNED_MODE,
    NULL_MODE,
    ORACLE_MODE,
    audit_exp287_information_modes,
    build_exp287_conflict_localizer,
    learned_top2_mask,
)


PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
ROOT_PREFIX = "20260913-exp287-learned-conflict-localization-v1-dev"


def _write_geometry(tmp_path: Path) -> tuple[Path, Path]:
    payload = {
        "schema": "NLM-EXP-287-DEVELOPMENT-GEOMETRY-V1",
        "experiment_id": "EXP-287",
        "authority_scope": "DEVELOPMENT_ONLY",
        "predeclaration": "BEFORE_HELDOUT_EVALUATION",
        "protocol_digest": PROTOCOL_DIGEST,
        "confirmatory_authority": False,
        "challenge_materialized": False,
        "scientific_evidence_eligible": False,
        "geometry": {
            "root_prefix": ROOT_PREFIX,
            "canonical_indices": [0, 1, 2, 3],
            "train_replicates": 64,
            "eval_replicates": 32,
            "eval_start_replicate": 10000,
            "batch_size": 8,
            "d_model": 64,
            "hidden_size": 48,
            "target_parameters": 500000,
            "timesteps": 4,
            "variables": 8,
            "decoys": 3,
            "max_search_steps": 16,
            "noise_std": 0.05,
            "lr": 0.002,
            "weight_decay": 0.0,
            "top_k": 2,
            "capture_threshold": 0.50,
            "precision_threshold": 0.50,
            "solution_rate_floor_delta": -0.005,
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest = tmp_path / "exp287.json"
    digest = tmp_path / "exp287.sha256"
    manifest.write_bytes(raw)
    digest.write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    return manifest, digest


def test_geometry_loader_freezes_preregistered_contract(tmp_path: Path) -> None:
    manifest, digest = _write_geometry(tmp_path)
    geometry, observed_digest = load_exp287_development_geometry(
        manifest,
        digest,
        protocol_digest=PROTOCOL_DIGEST,
    )

    assert GEOMETRY_SCHEMA == "NLM-EXP-287-DEVELOPMENT-GEOMETRY-V1"
    assert CANONICAL_INDICES == (0, 1, 2, 3)
    assert geometry == {
        "root_prefix": ROOT_PREFIX,
        "canonical_indices": [0, 1, 2, 3],
        "train_replicates": 64,
        "eval_replicates": 32,
        "eval_start_replicate": 10000,
        "batch_size": 8,
        "d_model": 64,
        "hidden_size": 48,
        "target_parameters": 500000,
        "timesteps": 4,
        "variables": 8,
        "decoys": 3,
        "max_search_steps": 16,
        "noise_std": 0.05,
        "lr": 0.002,
        "weight_decay": 0.0,
        "top_k": 2,
        "capture_threshold": 0.50,
        "precision_threshold": 0.50,
        "solution_rate_floor_delta": -0.005,
    }
    assert observed_digest == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert geometry["eval_start_replicate"] >= geometry["train_replicates"]


def test_geometry_loader_rejects_protocol_drift(tmp_path: Path) -> None:
    manifest, digest = _write_geometry(tmp_path)
    with pytest.raises(ValueError, match="protocol"):
        load_exp287_development_geometry(
            manifest,
            digest,
            protocol_digest="0" * 64,
        )


def test_exp287_world_is_deterministic_and_has_exact_two_variable_core() -> None:
    generator = Exp287ConflictGenerator(root_seed=f"{ROOT_PREFIX}::0")
    kwargs = dict(
        replicate=0,
        batch_size=8,
        timesteps=4,
        variables=8,
        decoys=3,
        d_model=64,
        noise_std=0.05,
        rng_stream="augmentation",
    )
    first = generator.make_batch(**kwargs)
    second = generator.make_batch(**kwargs)

    assert first.digest == second.digest
    assert torch.equal(first.surface_events, second.surface_events)
    assert torch.equal(first.variable_states, second.variable_states)
    assert torch.equal(first.core_masks, second.core_masks)
    assert first.metadata["experiment_id"] == "EXP-287"
    assert first.metadata["rng_stream"] == "augmentation"
    assert first.metadata["oracle_delivery_boundary"] == "core_materialized_only_after_current_contradiction"
    assert torch.equal(first.core_masks.sum(dim=1), torch.full((8,), 2.0))
    assert first.solution_targets.shape == (8, 8)
    assert first.surface_events.shape == (8, 4, 64)
    assert first.variable_states.shape == (8, 8, 64)


def test_exp287_training_and_evaluation_seed_domains_are_disjoint() -> None:
    generator = Exp287ConflictGenerator(root_seed=f"{ROOT_PREFIX}::1")
    train = generator.make_batch(
        replicate=63,
        batch_size=2,
        timesteps=4,
        variables=8,
        decoys=3,
        d_model=64,
        noise_std=0.05,
        rng_stream="augmentation",
    )
    evaluation = generator.make_batch(
        replicate=10000,
        batch_size=2,
        timesteps=4,
        variables=8,
        decoys=3,
        d_model=64,
        noise_std=0.05,
        rng_stream="evaluation",
    )
    assert train.metadata["seed"] != evaluation.metadata["seed"]
    assert train.rng_stream == "augmentation"
    assert evaluation.rng_stream == "evaluation"
    assert train.digest != evaluation.digest


def test_top2_mask_is_deterministic_and_ties_break_by_low_index() -> None:
    logits = torch.tensor([[0.5, 1.0, 1.0, -2.0, 0.0, 1.0, -1.0, 0.2]])
    mask = learned_top2_mask(logits)
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def test_same_model_exposes_three_information_modes_with_matched_cost() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = build_exp287_conflict_localizer(
            d_model=64,
            hidden_size=48,
            target_parameters=500000,
        )
    audit = audit_exp287_information_modes(
        model,
        timesteps=4,
        variables=8,
        max_search_steps=16,
    )

    assert audit["modes"] == [NULL_MODE, LEARNED_MODE, ORACLE_MODE]
    assert audit["parameter_inventory_shared"] is True
    assert audit["active_functional_parameters_shared"] is True
    assert audit["optimizer_visible_parameters_shared"] is True
    assert audit["accounted_flops_per_search_step_shared"] is True
    assert audit["localizer_flops_charged_in_all_modes"] is True
    assert audit["oracle_mode_deployable"] is False


def test_precontradiction_modes_use_null_core_and_oracle_leakage_fails_closed() -> None:
    model = build_exp287_conflict_localizer(
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    events = torch.randn(2, 4, 64)
    variables = torch.randn(2, 8, 64)
    oracle_core = torch.zeros(2, 8)
    oracle_core[:, :2] = 1.0

    for mode in (NULL_MODE, LEARNED_MODE, ORACLE_MODE):
        output = model(
            events,
            variables,
            mode=mode,
            contradiction_observed=False,
        )
        assert torch.count_nonzero(output.conflict_token).item() == 0
        assert output.oracle_information_delivered is False
        assert output.localizer_logits.shape == (2, 8)

    with pytest.raises(ValueError, match="before.*contradiction"):
        model(
            events,
            variables,
            mode=ORACLE_MODE,
            contradiction_observed=False,
            oracle_core_mask=oracle_core,
        )


def test_learned_mode_never_accepts_oracle_core_input() -> None:
    model = build_exp287_conflict_localizer(
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    events = torch.randn(1, 4, 64)
    variables = torch.randn(1, 8, 64)
    oracle_core = torch.zeros(1, 8)
    oracle_core[:, :2] = 1.0

    with pytest.raises(ValueError, match="learned.*oracle"):
        model(
            events,
            variables,
            mode=LEARNED_MODE,
            contradiction_observed=True,
            oracle_core_mask=oracle_core,
        )


def test_learned_mode_core_token_is_derived_from_localizer_logits_only() -> None:
    model = build_exp287_conflict_localizer(
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
    )
    events = torch.randn(3, 4, 64)
    variables = torch.randn(3, 8, 64)
    output = model(
        events,
        variables,
        mode=LEARNED_MODE,
        contradiction_observed=True,
    )

    assert torch.equal(output.conflict_token, learned_top2_mask(output.localizer_logits))
    assert torch.equal(output.conflict_token.sum(dim=1), torch.full((3,), 2.0))
    assert output.oracle_information_delivered is False
