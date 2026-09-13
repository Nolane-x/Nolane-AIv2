from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes
from .exp287_conflict_worlds import Exp287ConflictBatch, Exp287ConflictGenerator
from .exp287_development_geometry import (
    CANONICAL_INDICES,
    load_exp287_development_geometry,
)
from .matched_conflict_localizer_arms import (
    LEARNED_MODE,
    NULL_MODE,
    ORACLE_MODE,
    MODES,
    Exp287ConflictLocalizer,
    Exp287ModeOutput,
    audit_exp287_information_modes,
    build_exp287_conflict_localizer,
    learned_top2_mask,
)


SCHEMA = "NLM-EXP-287-LEARNED-CONFLICT-LOCALIZATION-ROOT-V1"
ESTABLISHED = "LEARNED_LOCALIZATION_VALUE_ESTABLISHED"
NOT_ESTABLISHED = "LEARNED_LOCALIZATION_VALUE_NOT_ESTABLISHED"
ORACLE_NOT_REPLICATED = "ORACLE_HEADROOM_NOT_REPLICATED"
CAPTURE_THRESHOLD = 0.50
PRECISION_THRESHOLD = 0.50
SOLUTION_RATE_FLOOR_DELTA = -0.005


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-287-MODEL-STATE-V1\0")
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "byteorder": tensor_byteorder(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(header)
        hasher.update(b"\0")
        hasher.update(tensor_raw_bytes(cpu))
    return hasher.hexdigest()


def build_seeded_exp287_model(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
) -> tuple[Exp287ConflictLocalizer, int]:
    seed = derive_stream_seed(root_seed, "EXP-287", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_exp287_conflict_localizer(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            device="cpu",
        )
    return model, seed


def compute_exp287_training_loss(
    output: Exp287ModeOutput,
    *,
    conflict_targets: torch.Tensor,
    solution_targets: torch.Tensor,
    core_masks: torch.Tensor,
) -> torch.Tensor:
    if output.mode != ORACLE_MODE or not output.contradiction_observed:
        raise ValueError("EXP-287 canonical training requires post-contradiction oracle conditioning")
    rollback_loss = F.cross_entropy(output.rollback_logits, conflict_targets)
    verifier_loss = F.binary_cross_entropy(
        output.verifier_confidence,
        solution_targets.to(dtype=output.verifier_confidence.dtype),
    )
    localizer_loss = F.binary_cross_entropy_with_logits(
        output.localizer_logits,
        core_masks.to(dtype=output.localizer_logits.dtype),
    )
    return rollback_loss + verifier_loss + localizer_loss


def _conflict_targets(batch: Exp287ConflictBatch) -> torch.Tensor:
    return torch.tensor(
        [int(item["conflict_variable"]) for item in batch.metadata["episodes"]],
        dtype=torch.long,
        device=batch.variable_states.device,
    )


def _selected_variables(mask: torch.Tensor, rollback_logits: torch.Tensor) -> list[int]:
    selected = [int(index) for index in torch.nonzero(mask > 0.5, as_tuple=False).flatten().tolist()]
    return sorted(selected, key=lambda value: (-float(rollback_logits[value].item()), value))


def _search_episode(
    *,
    mode: str,
    model: Exp287ConflictLocalizer,
    batch: Exp287ConflictBatch,
    episode_index: int,
    per_step_flops: int,
    ceiling: int,
    max_search_steps: int,
) -> dict[str, Any]:
    surface = batch.surface_events[episode_index : episode_index + 1]
    states = batch.variable_states[episode_index : episode_index + 1]
    oracle_core = batch.core_masks[episode_index : episode_index + 1]
    target = [int(value) for value in batch.solution_targets[episode_index].detach().cpu().tolist()]
    episode = batch.metadata["episodes"][episode_index]
    conflict_variable = int(episode["conflict_variable"])
    decoys = [int(value) for value in episode["decoy_variables"]]
    bad_value = int(episode["bad_branch_value"])
    original_order = [int(value) for value in batch.decoy_order[episode_index].detach().cpu().tolist()]

    queue = list(original_order)
    assignment: dict[int, int] = {}
    visited: list[int] = []
    step_receipts: list[dict[str, Any]] = []
    bad_attempted = False
    solved = False
    oracle_deliveries = 0
    learned_oracle_deliveries = 0

    for step_index in range(1, max_search_steps + 1):
        if not queue:
            break
        variable = int(queue.pop(0))
        visited.append(variable)
        contradiction = variable == conflict_variable and not bad_attempted
        selected_conflict_variables: list[int] = []

        if contradiction:
            bad_attempted = True
            assigned_value = bad_value
            output = model(
                surface,
                states,
                mode=mode,
                contradiction_observed=True,
                oracle_core_mask=oracle_core if mode == ORACLE_MODE else None,
            )
            if output.oracle_information_delivered:
                oracle_deliveries += 1
                if mode == LEARNED_MODE:
                    learned_oracle_deliveries += 1
            rollback_logits = output.rollback_logits[0].detach().cpu()

            if mode == NULL_MODE:
                replay_decoys = [value for value in reversed(decoys) if value in assignment]
                for value in replay_decoys:
                    assignment.pop(value, None)
                remaining = [
                    value
                    for value in original_order
                    if value not in replay_decoys
                    and value != conflict_variable
                    and value not in assignment
                ]
                queue = replay_decoys + [conflict_variable] + remaining
            else:
                selected_conflict_variables = _selected_variables(
                    output.conflict_token[0].detach().cpu(), rollback_logits
                )
                for value in selected_conflict_variables:
                    assignment.pop(value, None)
                remaining = [
                    value
                    for value in original_order
                    if value not in selected_conflict_variables and value not in assignment
                ]
                queue = selected_conflict_variables + remaining
        else:
            assigned_value = target[variable]
            assignment[variable] = assigned_value
            output = model(
                surface,
                states,
                mode=mode,
                contradiction_observed=False,
            )
            rollback_logits = output.rollback_logits[0].detach().cpu()

        step_receipts.append(
            {
                "step_index": int(step_index),
                "variable": int(variable),
                "assigned_value": int(assigned_value),
                "contradiction_observed": bool(contradiction),
                "selected_conflict_variables": selected_conflict_variables,
                "oracle_information_delivered": bool(output.oracle_information_delivered),
                "charged_accounted_flops": int(per_step_flops),
                "rollback_score_at_visited_variable": float(rollback_logits[variable].item()),
            }
        )

        solved = len(assignment) == len(target) and all(
            assignment.get(index) == expected for index, expected in enumerate(target)
        )
        if solved:
            break

    actual_cost = len(step_receipts) * int(per_step_flops)
    if actual_cost > ceiling:
        raise RuntimeError("EXP-287 search path exceeded frozen accounted-FLOP ceiling")
    censored = not solved
    charged_cost = int(ceiling if censored else actual_cost)
    return {
        "episode_index": int(episode_index),
        "verified_solution": bool(solved),
        "accounted_reasoning_flops_to_verified_solution": charged_cost,
        "actual_executed_flops_before_censoring": int(actual_cost),
        "censored_at_max_flops": bool(censored),
        "search_steps": int(len(step_receipts)),
        "visited_variables": visited,
        "oracle_information_delivery_count": int(oracle_deliveries),
        "learned_oracle_information_delivery_count": int(learned_oracle_deliveries),
        "precontradiction_conflict_delivery": False,
        "step_receipts": step_receipts,
    }


def _search_batch(
    *,
    mode: str,
    model: Exp287ConflictLocalizer,
    batch: Exp287ConflictBatch,
    per_step_flops: int,
    ceiling: int,
    max_search_steps: int,
) -> dict[str, Any]:
    episodes = [
        _search_episode(
            mode=mode,
            model=model,
            batch=batch,
            episode_index=index,
            per_step_flops=per_step_flops,
            ceiling=ceiling,
            max_search_steps=max_search_steps,
        )
        for index in range(batch.solution_targets.shape[0])
    ]
    return {
        "mean_accounted_cost": sum(
            float(item["accounted_reasoning_flops_to_verified_solution"]) for item in episodes
        )
        / len(episodes),
        "verified_solution_rate": sum(float(item["verified_solution"]) for item in episodes)
        / len(episodes),
        "oracle_information_delivered": any(
            int(item["oracle_information_delivery_count"]) > 0 for item in episodes
        ),
        "learned_oracle_information_delivered": any(
            int(item["learned_oracle_information_delivery_count"]) > 0 for item in episodes
        ),
        "censored_episode_count": sum(bool(item["censored_at_max_flops"]) for item in episodes),
        "episodes": episodes,
    }


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    flat_scores = scores.detach().cpu().reshape(-1)
    flat_labels = labels.detach().cpu().reshape(-1).to(dtype=torch.bool)
    positives = int(flat_labels.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(flat_scores, descending=True, stable=True)
    ranked = flat_labels[order]
    cumulative = torch.cumsum(ranked.to(dtype=torch.float64), dim=0)
    ranks = torch.arange(1, ranked.numel() + 1, dtype=torch.float64)
    precision = cumulative / ranks
    return float(precision[ranked].sum().item() / positives)


def _localization_receipt(
    model: Exp287ConflictLocalizer,
    batch: Exp287ConflictBatch,
) -> dict[str, Any]:
    output = model(
        batch.surface_events,
        batch.variable_states,
        mode=LEARNED_MODE,
        contradiction_observed=True,
    )
    selected = learned_top2_mask(output.localizer_logits)
    labels = batch.core_masks.to(dtype=selected.dtype)
    true_selected = float((selected * labels).sum().item())
    selected_total = float(selected.sum().item())
    exact = torch.all(selected == labels, dim=1)
    return {
        "evaluated_variables": int(labels.numel()),
        "episodes": int(labels.shape[0]),
        "selected_true_core_members": int(true_selected),
        "selected_core_members": int(selected_total),
        "top2_core_precision": true_selected / selected_total,
        "top2_core_recall": true_selected / float(labels.sum().item()),
        "exact_core_recovery_rate": float(exact.to(dtype=torch.float64).mean().item()),
        "off_core_selection_rate": 1.0 - (true_selected / selected_total),
        "variable_average_precision": _average_precision(output.localizer_logits, labels),
        "raw_core_prevalence": float(labels.mean().item()),
    }


def evaluate_exp287_batch_same_weights(
    model: Exp287ConflictLocalizer,
    batch: Exp287ConflictBatch,
    *,
    per_step_flops: int,
    ceiling: int,
    max_search_steps: int,
) -> dict[str, Any]:
    if min(per_step_flops, ceiling, max_search_steps) <= 0:
        raise ValueError("EXP-287 evaluation compute geometry must be positive")
    before = model_state_digest(model)
    model.eval()
    with torch.no_grad():
        localization = _localization_receipt(model, batch)
        modes = {
            mode: _search_batch(
                mode=mode,
                model=model,
                batch=batch,
                per_step_flops=per_step_flops,
                ceiling=ceiling,
                max_search_steps=max_search_steps,
            )
            for mode in MODES
        }
    after = model_state_digest(model)
    if after != before:
        raise RuntimeError("EXP-287 canonical model changed during same-weights evaluation")
    if modes[LEARNED_MODE]["learned_oracle_information_delivered"]:
        raise RuntimeError("EXP-287 learned mode consumed oracle information")
    precontradiction = any(
        bool(episode["precontradiction_conflict_delivery"])
        for mode in MODES
        for episode in modes[mode]["episodes"]
    )
    return {
        "schema": "NLM-EXP-287-SAME-WEIGHTS-BATCH-EVAL-V1",
        "experiment_id": "EXP-287",
        "replicate": int(batch.replicate),
        "rng_stream": batch.rng_stream,
        "batch_digest": batch.digest,
        "model_state_digest": before,
        "modes": modes,
        "localization": localization,
        "precontradiction_conflict_delivery": bool(precontradiction),
    }


def classify_exp287_root_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    c0 = float(metrics["mean_control_cost"])
    cl = float(metrics["mean_learned_cost"])
    co = float(metrics["mean_oracle_cost"])
    control_solution = float(metrics["control_solution_rate"])
    learned_solution = float(metrics["learned_solution_rate"])
    precision = float(metrics["top2_core_precision"])
    if min(c0, cl, co) < 0.0:
        raise ValueError("EXP-287 accounted costs must be non-negative")
    if not (0.0 <= control_solution <= 1.0 and 0.0 <= learned_solution <= 1.0):
        raise ValueError("EXP-287 solution rates must be in [0,1]")
    if not 0.0 <= precision <= 1.0:
        raise ValueError("EXP-287 top2 precision must be in [0,1]")

    oracle_headroom = c0 - co
    learned_headroom = c0 - cl
    if oracle_headroom <= 0.0:
        return {
            "classification": ORACLE_NOT_REPLICATED,
            "oracle_headroom": oracle_headroom,
            "learned_headroom": learned_headroom,
            "capture": None,
            "predicates": {
                "oracle_headroom_positive": False,
                "learned_headroom_positive": learned_headroom > 0.0,
                "capture_at_least_half": False,
                "solution_floor_preserved": learned_solution >= control_solution + SOLUTION_RATE_FLOOR_DELTA,
                "top2_precision_above_half": precision > PRECISION_THRESHOLD,
                "learned_oracle_boundary_closed": not bool(
                    metrics["learned_oracle_information_delivered"]
                ),
                "precontradiction_boundary_closed": not bool(
                    metrics["precontradiction_conflict_delivery"]
                ),
            },
        }

    capture = learned_headroom / oracle_headroom
    predicates = {
        "oracle_headroom_positive": True,
        "learned_headroom_positive": learned_headroom > 0.0,
        "capture_at_least_half": capture >= CAPTURE_THRESHOLD,
        "solution_floor_preserved": learned_solution >= control_solution + SOLUTION_RATE_FLOOR_DELTA,
        "top2_precision_above_half": precision > PRECISION_THRESHOLD,
        "learned_oracle_boundary_closed": not bool(
            metrics["learned_oracle_information_delivered"]
        ),
        "precontradiction_boundary_closed": not bool(
            metrics["precontradiction_conflict_delivery"]
        ),
    }
    return {
        "classification": ESTABLISHED if all(predicates.values()) else NOT_ESTABLISHED,
        "oracle_headroom": oracle_headroom,
        "learned_headroom": learned_headroom,
        "capture": capture,
        "predicates": predicates,
    }


def _train_canonical_model(
    *,
    root_seed: str,
    geometry: dict[str, Any],
) -> tuple[Exp287ConflictLocalizer, int, list[float]]:
    model, model_seed = build_seeded_exp287_model(
        root_seed=root_seed,
        d_model=int(geometry["d_model"]),
        hidden_size=int(geometry["hidden_size"]),
        target_parameters=int(geometry["target_parameters"]),
    )
    optimizer = build_functional_optimizer(
        model,
        lr=float(geometry["lr"]),
        weight_decay=float(geometry["weight_decay"]),
    )
    generator = Exp287ConflictGenerator(root_seed=root_seed)
    losses: list[float] = []
    for replicate in range(int(geometry["train_replicates"])):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=int(geometry["batch_size"]),
            timesteps=int(geometry["timesteps"]),
            variables=int(geometry["variables"]),
            decoys=int(geometry["decoys"]),
            d_model=int(geometry["d_model"]),
            noise_std=float(geometry["noise_std"]),
            rng_stream="augmentation",
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.surface_events,
            batch.variable_states,
            mode=ORACLE_MODE,
            contradiction_observed=True,
            oracle_core_mask=batch.core_masks,
        )
        loss = compute_exp287_training_loss(
            output,
            conflict_targets=_conflict_targets(batch),
            solution_targets=batch.solution_targets,
            core_masks=batch.core_masks,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return model, model_seed, losses


def run_exp287_root_development(
    canonical_index: int,
    *,
    manifest_path: str | Path | None = None,
    digest_path: str | Path | None = None,
    protocol_digest: str,
) -> dict[str, Any]:
    if canonical_index not in CANONICAL_INDICES:
        raise ValueError("EXP-287 canonical_index must be one of 0,1,2,3")
    root = _repo_root()
    manifest = Path(manifest_path) if manifest_path is not None else root / "protocols/exp287_development_v1.json"
    digest = Path(digest_path) if digest_path is not None else root / "protocols/exp287_development_v1.sha256"
    geometry, geometry_digest = load_exp287_development_geometry(
        manifest,
        digest,
        protocol_digest=protocol_digest,
    )
    root_seed = f"{geometry['root_prefix']}::{canonical_index}"
    model, model_seed, training_losses = _train_canonical_model(
        root_seed=root_seed,
        geometry=geometry,
    )
    frozen_digest = model_state_digest(model)
    audit = audit_exp287_information_modes(
        model,
        timesteps=int(geometry["timesteps"]),
        variables=int(geometry["variables"]),
        max_search_steps=int(geometry["max_search_steps"]),
    )
    per_step_flops = int(audit["compute_ledger"]["accounted_flops_per_search_step"])
    ceiling = int(audit["compute_ledger"]["max_accounted_flops_per_episode"])
    generator = Exp287ConflictGenerator(root_seed=root_seed)

    batch_receipts: list[dict[str, Any]] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    selected_true = 0
    selected_total = 0
    exact_recoveries = 0
    total_episodes = 0
    for offset in range(int(geometry["eval_replicates"])):
        replicate = int(geometry["eval_start_replicate"]) + offset
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=int(geometry["batch_size"]),
            timesteps=int(geometry["timesteps"]),
            variables=int(geometry["variables"]),
            decoys=int(geometry["decoys"]),
            d_model=int(geometry["d_model"]),
            noise_std=float(geometry["noise_std"]),
            rng_stream="evaluation",
        )
        receipt = evaluate_exp287_batch_same_weights(
            model,
            batch,
            per_step_flops=per_step_flops,
            ceiling=ceiling,
            max_search_steps=int(geometry["max_search_steps"]),
        )
        batch_receipts.append(receipt)
        with torch.no_grad():
            localizer = model(
                batch.surface_events,
                batch.variable_states,
                mode=LEARNED_MODE,
                contradiction_observed=True,
            )
            selected = learned_top2_mask(localizer.localizer_logits)
        labels = batch.core_masks.to(dtype=selected.dtype)
        all_scores.append(localizer.localizer_logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        selected_true += int((selected * labels).sum().item())
        selected_total += int(selected.sum().item())
        exact_recoveries += int(torch.all(selected == labels, dim=1).sum().item())
        total_episodes += int(labels.shape[0])

    if model_state_digest(model) != frozen_digest:
        raise RuntimeError("EXP-287 canonical model changed after held-out evaluation")

    def mean_mode(mode: str, key: str) -> float:
        return sum(float(row["modes"][mode][key]) for row in batch_receipts) / len(batch_receipts)

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    precision = selected_true / selected_total
    localization = {
        "evaluated_variables": int(labels.numel()),
        "episodes": int(total_episodes),
        "selected_true_core_members": int(selected_true),
        "selected_core_members": int(selected_total),
        "top2_core_precision": precision,
        "top2_core_recall": selected_true / float(labels.sum().item()),
        "exact_core_recovery_rate": exact_recoveries / total_episodes,
        "off_core_selection_rate": 1.0 - precision,
        "variable_average_precision": _average_precision(scores, labels),
        "raw_core_prevalence": float(labels.mean().item()),
    }
    primitive_metrics = {
        "mean_control_cost": mean_mode(NULL_MODE, "mean_accounted_cost"),
        "mean_learned_cost": mean_mode(LEARNED_MODE, "mean_accounted_cost"),
        "mean_oracle_cost": mean_mode(ORACLE_MODE, "mean_accounted_cost"),
        "control_solution_rate": mean_mode(NULL_MODE, "verified_solution_rate"),
        "learned_solution_rate": mean_mode(LEARNED_MODE, "verified_solution_rate"),
        "oracle_solution_rate": mean_mode(ORACLE_MODE, "verified_solution_rate"),
        "top2_core_precision": precision,
        "learned_oracle_information_delivered": any(
            bool(row["modes"][LEARNED_MODE]["learned_oracle_information_delivered"])
            for row in batch_receipts
        ),
        "precontradiction_conflict_delivery": any(
            bool(row["precontradiction_conflict_delivery"]) for row in batch_receipts
        ),
    }
    decision = classify_exp287_root_metrics(primitive_metrics)
    return {
        "schema": SCHEMA,
        "experiment_id": "EXP-287",
        "canonical_index": int(canonical_index),
        "root_seed": root_seed,
        "geometry": geometry,
        "geometry_digest": geometry_digest,
        "protocol_digest": protocol_digest,
        "model_init_seed": int(model_seed),
        "model_state_digest": frozen_digest,
        "training": {
            "rng_stream": "augmentation",
            "replicates": int(geometry["train_replicates"]),
            "loss_formula": "rollback_ce + verifier_bce + localizer_bce",
            "loss_coefficients": [1.0, 1.0, 1.0],
            "first_loss": training_losses[0],
            "last_loss": training_losses[-1],
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "replicates": int(geometry["eval_replicates"]),
            "start_replicate": int(geometry["eval_start_replicate"]),
            "batch_receipts": batch_receipts,
        },
        "mode_audit": audit,
        "localization": localization,
        "primitive_metrics": primitive_metrics,
        "decision": decision,
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "evaluation_labels_used_for_training": False,
        "evaluation_core_used_by_learned_mode": False,
        "oracle_mode_deployable": False,
    }
