from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import _hybrid_stop_decision_logits
from nolane_ai.experiments.exp279_routing_worlds import Exp279RoutingGenerator
from nolane_ai.experiments.exp279_stop_path_ablation import (
    ActiveReclaimedStopArm,
    build_stop_path_ablation_arms,
    run_exp279_stop_path_ablation,
    stop_path_ablation_compute_ledger,
    train_aux_step,
    train_clean_step,
)
from nolane_ai.training.optimizer import build_functional_optimizer


def _batch(*, d_model: int = 16):
    return Exp279RoutingGenerator(root_seed="stop-path-ablation-test").make_batch(
        replicate=0,
        batch_size=4,
        timesteps=3,
        variables=4,
        constraints=2,
        d_model=d_model,
        noise_std=0.05,
        rng_stream="augmentation",
        stratum="MIXED_RESIDUAL",
    )


def _prefix_state(model: torch.nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def _prefix_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    assert left.keys() == right.keys()
    return all(torch.equal(left[name], right[name]) for name in left)


def test_raw_stop_is_exact_hybrid_stop_path() -> None:
    arms = build_stop_path_ablation_arms(
        root_seed="stop-path-exact",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
    )
    raw = arms["raw_stop_clean"]
    batch = _batch()

    output = raw(batch.surface_events, batch.variable_states, batch.incidence)
    expected = _hybrid_stop_decision_logits(
        raw,
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
    )

    assert not bool(output.branch_route_mask.any().item())
    assert torch.allclose(output.decision_logits, expected)
    assert output.representation_semantics == "propagation_then_branch_only_for_routed_residual_uncertainty"


def test_active_reclaimed_stop_executes_one_step_branch_capacity_without_routing() -> None:
    arms = build_stop_path_ablation_arms(
        root_seed="stop-path-active",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
    )
    arm = arms["active_reclaimed_stop_clean"]
    assert isinstance(arm, ActiveReclaimedStopArm)
    batch = _batch()

    calls: list[bool] = []
    handle = arm.branch_gru.register_forward_hook(lambda *_: calls.append(True))
    output = arm(batch.surface_events, batch.variable_states, batch.incidence)
    handle.remove()

    assert calls == [True]
    assert not bool(output.branch_route_mask.any().item())
    assert (
        output.representation_semantics
        == "propagation_stop_with_one_step_active_reclaimed_branch_capacity"
    )


def test_ablation_arms_start_byte_identical() -> None:
    arms = build_stop_path_ablation_arms(
        root_seed="stop-path-identical",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
    )
    states = [arm.state_dict() for arm in arms.values()]
    names = list(states[0])
    assert names
    for state in states[1:]:
        assert list(state) == names
        for name in names:
            assert torch.equal(state[name], states[0][name])


def test_active_reclaimed_stop_matches_propagation_inference_flops() -> None:
    arms = build_stop_path_ablation_arms(
        root_seed="stop-path-ledger",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
    )
    ledger = stop_path_ablation_compute_ledger(
        arms,
        timesteps=3,
        variables=4,
        constraints=2,
    )

    propagation = ledger["propagation_clean"]
    raw = ledger["raw_stop_clean"]
    aux = ledger["raw_stop_aux"]
    reclaimed = ledger["active_reclaimed_stop_clean"]

    assert reclaimed["accounted_flops_per_episode"] == propagation["accounted_flops_per_episode"]
    assert raw["accounted_flops_per_episode"] < propagation["accounted_flops_per_episode"]
    assert aux["accounted_flops_per_episode"] == raw["accounted_flops_per_episode"]
    assert raw["inactive_execution_parameters"] > 0
    assert aux["inactive_execution_parameters"] == raw["inactive_execution_parameters"]
    assert propagation["inactive_execution_parameters"] == 0
    assert reclaimed["inactive_execution_parameters"] == 0
    assert raw["fairness_class"] == "DIAGNOSTIC_INACTIVE_CAPACITY"
    assert reclaimed["fairness_class"] == "FAIR_ACTIVE_CAPACITY_SAME_FLOPS"


def test_aux_training_changes_branch_expert_but_clean_raw_training_does_not() -> None:
    arms = build_stop_path_ablation_arms(
        root_seed="stop-path-training-isolation",
        d_model=16,
        hidden_size=12,
        target_parameters=12_000,
    )
    batch = _batch()
    clean = arms["raw_stop_clean"]
    aux = arms["raw_stop_aux"]

    clean_before = _prefix_state(clean, "branch_gru.")
    aux_before = _prefix_state(aux, "branch_gru.")
    assert clean_before and aux_before

    clean_optimizer = build_functional_optimizer(clean, lr=0.002, weight_decay=0.0)
    aux_optimizer = build_functional_optimizer(aux, lr=0.002, weight_decay=0.0)
    train_clean_step(
        clean,
        clean_optimizer,
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
        targets=batch.targets,
    )
    train_aux_step(
        aux,
        aux_optimizer,
        surface_events=batch.surface_events,
        variable_states=batch.variable_states,
        incidence=batch.incidence,
        targets=batch.targets,
    )

    clean_after = _prefix_state(clean, "branch_gru.")
    aux_after = _prefix_state(aux, "branch_gru.")
    assert _prefix_equal(clean_before, clean_after)
    assert not _prefix_equal(aux_before, aux_after)


def test_runner_is_development_only_and_keeps_stop_paths_stop_only() -> None:
    artifact = run_exp279_stop_path_ablation(
        root_seed="stop-path-runner-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=3,
        eval_replicates=6,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=0.002,
        weight_decay=0.0,
    )

    assert artifact["schema"] == "NLM-EXP-279-STOP-PATH-ABLATION-DEV-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["scientific_evidence_eligible"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["initial_state"]["byte_identical"] is True
    assert artifact["fairness"]["active_pair_same_flops"] is True
    assert artifact["fairness"]["raw_stop_has_inactive_capacity"] is True

    rows = artifact["evaluation"]["per_replicate"]
    assert len(rows) == 6
    for row in rows:
        for arm_id in (
            "propagation_clean",
            "raw_stop_clean",
            "raw_stop_aux",
            "active_reclaimed_stop_clean",
        ):
            assert row[arm_id]["route_fraction"] == 0.0

    contrasts = artifact["evaluation"]["aggregate"]["contrasts"]
    assert set(contrasts) == {
        "raw_clean_vs_propagation",
        "raw_aux_vs_raw_clean",
        "active_reclaimed_vs_propagation",
    }
