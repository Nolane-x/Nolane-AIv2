from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.experiments.exp290_transfer_worlds import (
    Exp290TransferGenerator,
    Exp290TransferPairBatch,
)
from nolane_ai.experiments.matched_clause_transfer_arms import (
    TARGET_MODES,
    Exp290ClauseTransferModel,
    audit_exp290_transfer_model,
    build_matched_exp290_model,
    deterministic_transfer_top1,
)
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import build_functional_optimizer
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


EXP290_ROOT_SCHEMA = "NLM-EXP-290-STRUCTURAL-CLAUSE-TRANSFER-ROOT-V1"
PRIMARY_ENDPOINT = "source_equivalent_target_dead_end_rate"
ROOT_ESTABLISHED = "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"
ROOT_NOT_ESTABLISHED = "LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED"
ORACLE_NOT_REPLICATED = "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-290-MODEL-STATE-V1\0")
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(
            _canonical_bytes(
                {
                    "shape": list(cpu.shape),
                    "dtype": str(cpu.dtype),
                    "byteorder": tensor_byteorder(),
                }
            )
        )
        hasher.update(b"\0")
        hasher.update(tensor_raw_bytes(cpu))
    return hasher.hexdigest()


def _mapping_tensor(pair: Exp290TransferPairBatch) -> torch.Tensor:
    rows = [
        episode["evaluator_only_source_to_target_variable_permutation"]
        for episode in pair.metadata["episodes"]
    ]
    return torch.tensor(rows, dtype=torch.long)


def _nonidentity_episode_count(pair: Exp290TransferPairBatch, variables: int) -> int:
    identity = list(range(variables))
    return sum(
        episode["evaluator_only_source_to_target_variable_permutation"] != identity
        for episode in pair.metadata["episodes"]
    )


def training_contract_receipt() -> dict[str, Any]:
    return {
        "rng_stream": "augmentation",
        "evaluation_mapping_used_for_training": False,
        "evaluation_lineage_consumed": False,
        "loss_weights": {
            "branch_cross_entropy": 1.0,
            "verifier_binary_cross_entropy": 1.0,
            "transfer_mapping_cross_entropy": 1.0,
        },
        "class_weights_used": False,
        "calibration_used": False,
        "early_stopping": False,
        "hard_negative_mining": False,
    }


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc


def _validate_geometry(geometry: Mapping[str, object], canonical_index: int) -> None:
    for key in (
        "batch_size",
        "d_model",
        "eval_replicates",
        "hidden_size",
        "max_search_steps",
        "restarts",
        "target_parameters",
        "timesteps",
        "train_replicates",
        "variables",
    ):
        if int(geometry[key]) <= 0:
            raise ValueError(f"EXP-290 geometry {key} must be positive")
    if canonical_index not in [int(value) for value in geometry["canonical_indices"]]:
        raise ValueError("canonical_index is outside frozen EXP-290 geometry")
    if int(geometry["eval_start_replicate"]) < int(geometry["train_replicates"]):
        raise ValueError("EXP-290 training and evaluation replicate lineages must be disjoint")
    variables = int(geometry["variables"])
    decoys = int(geometry["decoys"])
    if decoys <= 0 or decoys >= variables:
        raise ValueError("EXP-290 decoys must be positive and leave productive variables")
    if float(geometry["noise_std"]) < 0.0:
        raise ValueError("noise_std must be non-negative")
    if float(geometry["lr"]) <= 0.0:
        raise ValueError("lr must be positive")
    if float(geometry["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must be non-negative")


def _root_seed(geometry: Mapping[str, object], canonical_index: int) -> str:
    return f"{geometry['root_prefix']}-root-{canonical_index}"


def _build_model(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
) -> tuple[Exp290ClauseTransferModel, int]:
    seed = derive_stream_seed(root_seed, "EXP-290", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_matched_exp290_model(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            device="cpu",
        )
    return model, int(seed)


def _train_model(
    *,
    model: Exp290ClauseTransferModel,
    generator: Exp290TransferGenerator,
    geometry: Mapping[str, object],
) -> dict[str, Any]:
    optimizer = build_functional_optimizer(
        model,
        lr=float(geometry["lr"]),
        weight_decay=float(geometry["weight_decay"]),
    )
    batch_size = int(geometry["batch_size"])
    variables = int(geometry["variables"])
    rows: list[dict[str, Any]] = []
    pair_digests: list[str] = []

    model.train()
    for replicate in range(int(geometry["train_replicates"])):
        pair = generator.make_pair(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=int(geometry["timesteps"]),
            restarts=int(geometry["restarts"]),
            variables=variables,
            decoys=int(geometry["decoys"]),
            d_model=int(geometry["d_model"]),
            noise_std=float(geometry["noise_std"]),
            rng_stream="augmentation",
            device="cpu",
        )
        pair_digests.append(pair.digest)

        source_index = pair.source.restart_orders[:, 0, 0].to(torch.long)
        gather_index = source_index.view(batch_size, 1)
        source_truth = pair.source.solution_targets.gather(1, gather_index).squeeze(1)
        source_literal = 1 - source_truth
        target_index = _mapping_tensor(pair).gather(1, gather_index).squeeze(1)

        optimizer.zero_grad(set_to_none=True)
        reasoner = model.encode_reasoner_state(
            pair.source.surface_events,
            pair.source.variable_states,
            local_memory_hit=torch.zeros(batch_size, variables),
        )
        transfer_scores = model.score_transfer(
            source_variable_states=pair.source.variable_states,
            target_variable_states=pair.target.variable_states,
            source_variable_index=source_index,
            literal_value=source_literal,
        )
        branch_loss = F.cross_entropy(reasoner.branch_logits, source_index)
        verifier_loss = F.binary_cross_entropy(
            reasoner.verifier_confidence,
            pair.source.solution_targets.to(torch.float32),
        )
        transfer_loss = F.cross_entropy(transfer_scores, target_index)
        loss = branch_loss + verifier_loss + transfer_loss
        loss.backward()
        optimizer.step()

        rows.append(
            {
                "replicate": int(replicate),
                "pair_digest": pair.digest,
                "branch_cross_entropy": float(branch_loss.detach().item()),
                "verifier_binary_cross_entropy": float(verifier_loss.detach().item()),
                "transfer_mapping_cross_entropy": float(transfer_loss.detach().item()),
                "total_loss": float(loss.detach().item()),
                "evaluation_mapping_used_for_training": False,
                "evaluator_truth_gated_action": False,
            }
        )

    return {
        **training_contract_receipt(),
        "start_replicate": 0,
        "replicates": int(geometry["train_replicates"]),
        "batch_size": batch_size,
        "pair_digests": pair_digests,
        "per_replicate": rows,
        "post_training_model_digest": _state_digest(model),
    }


def _literal(variable_index: int, value: int) -> dict[str, int]:
    return {"variable_index": int(variable_index), "value": int(value)}


def build_common_source_phase(
    pair: Exp290TransferPairBatch,
    *,
    max_search_steps: int,
) -> dict[str, Any]:
    if max_search_steps <= 0:
        raise ValueError("max_search_steps must be positive")
    source = pair.source
    batch_size = int(source.solution_targets.shape[0])
    variables = int(source.solution_targets.shape[1])
    restarts = int(source.restart_orders.shape[1])
    episodes: list[dict[str, Any]] = []

    for episode_index in range(batch_size):
        local_nogoods: set[tuple[int, int]] = set()
        clauses: list[dict[str, int]] = []
        insertions: list[dict[str, Any]] = []
        search_steps = 0
        verified_solution = False

        for restart_index in range(restarts):
            assignment: dict[int, int] = {}
            contradicted = False
            for order_offset in range(variables):
                if search_steps >= max_search_steps:
                    break
                variable_index = int(
                    source.restart_orders[episode_index, restart_index, order_offset].item()
                )
                search_steps += 1
                chosen: int | None = None
                for candidate_tensor in source.restart_value_orders[
                    episode_index, restart_index, variable_index
                ]:
                    candidate = int(candidate_tensor.item())
                    if (variable_index, candidate) in local_nogoods:
                        continue
                    chosen = candidate
                    break
                if chosen is None:
                    continue

                public_allowed_value = int(
                    source.solution_targets[episode_index, variable_index].item()
                )
                assignment[variable_index] = chosen
                if chosen != public_allowed_value:
                    key = (variable_index, chosen)
                    if key not in local_nogoods:
                        local_nogoods.add(key)
                        clause = _literal(variable_index, chosen)
                        clauses.append(clause)
                        insertions.append(
                            {
                                "restart_index": int(restart_index),
                                "search_step": int(search_steps),
                                "partial_assignment_reached": True,
                                "public_contradiction_observed_before_insertion": True,
                                "evaluator_truth_used_for_insertion": False,
                                "future_target_truth_used": False,
                                "clause": [dict(clause)],
                            }
                        )
                    contradicted = True
                    break
            if not contradicted and len(assignment) == variables:
                verified_solution = True
                break
            if search_steps >= max_search_steps:
                break

        episodes.append(
            {
                "episode_index": int(episode_index),
                "source_clauses": clauses,
                "insertion_receipts": insertions,
                "source_search_steps": int(search_steps),
                "source_verified_solution": bool(verified_solution),
            }
        )

    return {
        "schema": "NLM-EXP-290-COMMON-SOURCE-PHASE-V1",
        "pair_digest": pair.digest,
        "source_batch_digest": _json_digest(source.metadata),
        "source_clause_set_sealed_before_target": True,
        "evaluator_truth_gated_insertion": False,
        "oracle_mapping_used": False,
        "cross_episode_reuse": False,
        "exact_one_literal_nogoods": True,
        "episodes": episodes,
    }


def _translate_learned_source_clauses(
    model: Exp290ClauseTransferModel,
    *,
    source_variable_states: torch.Tensor,
    target_variable_states: torch.Tensor,
    source_clauses: list[dict[str, int]],
) -> list[tuple[int, int]]:
    translated: list[tuple[int, int]] = []
    for clause in source_clauses:
        source_index = torch.tensor([int(clause["variable_index"])], dtype=torch.long)
        literal_value = torch.tensor([int(clause["value"])], dtype=torch.long)
        scores = model.score_transfer(
            source_variable_states=source_variable_states,
            target_variable_states=target_variable_states,
            source_variable_index=source_index,
            literal_value=literal_value,
        )
        target_index = int(deterministic_transfer_top1(scores).item())
        translated.append((target_index, int(clause["value"])))
    return translated


def _translate_isomorphic_source_clauses(
    pair: Exp290TransferPairBatch,
    *,
    episode_index: int,
    source_clauses: list[dict[str, int]],
) -> list[tuple[int, int]]:
    mapping = pair.metadata["episodes"][episode_index][
        "evaluator_only_source_to_target_variable_permutation"
    ]
    return [
        (int(mapping[int(clause["variable_index"])]), int(clause["value"]))
        for clause in source_clauses
    ]


def _evaluate_target_episode(
    *,
    mode: str,
    pair: Exp290TransferPairBatch,
    source_episode: Mapping[str, Any],
    episode_index: int,
    model: Exp290ClauseTransferModel,
    max_search_steps: int,
    transferred_slot_capacity: int,
) -> dict[str, Any]:
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown EXP-290 target mode: {mode}")
    target = pair.target
    variables = int(target.solution_targets.shape[1])
    restarts = int(target.restart_orders.shape[1])
    source_clauses = [dict(row) for row in source_episode["source_clauses"]]
    if len(source_clauses) > transferred_slot_capacity:
        raise ValueError("source clause count exceeds frozen transferred-slot capacity")

    source_states = pair.source.variable_states[episode_index : episode_index + 1]
    target_states = pair.target.variable_states[episode_index : episode_index + 1]
    learned_slots = _translate_learned_source_clauses(
        model,
        source_variable_states=source_states,
        target_variable_states=target_states,
        source_clauses=source_clauses,
    )
    isomorphic_slots = _translate_isomorphic_source_clauses(
        pair,
        episode_index=episode_index,
        source_clauses=source_clauses,
    )

    if mode == "LOCAL_ONLY_CONTROL":
        active_slots: set[tuple[int, int]] = set()
    elif mode == "LEARNED_STRUCTURAL_TRANSFER":
        active_slots = set(learned_slots)
    else:
        active_slots = set(isomorphic_slots)

    opportunity_literals = set(isomorphic_slots)
    locally_observed: set[tuple[int, int]] = set()
    first_opportunity_seen: set[tuple[int, int]] = set()
    source_equivalent_dead_ends = 0
    transferred_prunes = 0
    transferred_valid_state_prunes = 0
    local_prunes = 0
    target_local_insertions = 0
    search_steps = 0
    transferred_slot_comparisons = 0
    verified_solution = False

    for restart_index in range(restarts):
        assignment: dict[int, int] = {}
        contradicted = False
        for order_offset in range(variables):
            if search_steps >= max_search_steps:
                break
            variable_index = int(
                target.restart_orders[episode_index, restart_index, order_offset].item()
            )
            search_steps += 1
            transferred_slot_comparisons += transferred_slot_capacity
            chosen: int | None = None
            for candidate_tensor in target.restart_value_orders[
                episode_index, restart_index, variable_index
            ]:
                candidate = int(candidate_tensor.item())
                key = (variable_index, candidate)
                if key in active_slots:
                    transferred_prunes += 1
                    if candidate == int(
                        target.solution_targets[episode_index, variable_index].item()
                    ):
                        transferred_valid_state_prunes += 1
                    continue
                if key in locally_observed:
                    local_prunes += 1
                    continue
                chosen = candidate
                break
            if chosen is None:
                contradicted = True
                break

            key = (variable_index, chosen)
            public_allowed_value = int(
                target.solution_targets[episode_index, variable_index].item()
            )
            assignment[variable_index] = chosen
            if chosen != public_allowed_value:
                if key in opportunity_literals and key not in first_opportunity_seen:
                    first_opportunity_seen.add(key)
                    source_equivalent_dead_ends += 1
                if key not in locally_observed:
                    locally_observed.add(key)
                    target_local_insertions += 1
                contradicted = True
                break
        if not contradicted and len(assignment) == variables:
            verified_solution = True
            break
        if search_steps >= max_search_steps:
            break

    return {
        "episode_index": int(episode_index),
        "mode": mode,
        "source_clause_count": len(source_clauses),
        "source_equivalent_opportunities": len(opportunity_literals),
        "source_equivalent_target_dead_ends": int(source_equivalent_dead_ends),
        "transferred_prune_count": int(transferred_prunes),
        "transferred_valid_state_prune_count": int(transferred_valid_state_prunes),
        "target_local_prune_count": int(local_prunes),
        "target_local_insertion_count": int(target_local_insertions),
        "target_local_memory_enabled": True,
        "target_local_insertion_requires_observed_contradiction": True,
        "transfer_scorer_executed": True,
        "fixed_transferred_slot_accounting": True,
        "transferred_slot_capacity": int(transferred_slot_capacity),
        "transferred_slot_comparisons_charged": int(transferred_slot_comparisons),
        "oracle_mapping_delivered": mode == "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
        "evaluation_mapping_delivered": mode == "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
        "evaluator_truth_gates_target_action": False,
        "verified_solution": bool(verified_solution),
        "search_steps": int(search_steps),
    }


def _aggregate_mode(
    *,
    mode: str,
    rows: list[dict[str, Any]],
    model_digest: str,
    transferred_slot_capacity: int,
) -> dict[str, Any]:
    opportunities = sum(int(row["source_equivalent_opportunities"]) for row in rows)
    dead_ends = sum(int(row["source_equivalent_target_dead_ends"]) for row in rows)
    transferred_prunes = sum(int(row["transferred_prune_count"]) for row in rows)
    valid_prunes = sum(int(row["transferred_valid_state_prune_count"]) for row in rows)
    solutions = sum(bool(row["verified_solution"]) for row in rows)
    rate = float(dead_ends / opportunities) if opportunities else 0.0
    overprune = float(valid_prunes / transferred_prunes) if transferred_prunes else 0.0
    return {
        "mode": mode,
        "model_digest": model_digest,
        "episodes": len(rows),
        "source_equivalent_opportunities": int(opportunities),
        "source_equivalent_target_dead_ends": int(dead_ends),
        "source_equivalent_target_dead_end_rate": rate,
        "transferred_prune_count": int(transferred_prunes),
        "transferred_valid_state_prune_count": int(valid_prunes),
        "valid_state_overprune_rate": overprune,
        "verified_solution_rate": float(solutions / len(rows)) if rows else 0.0,
        "target_local_memory_enabled": True,
        "target_local_insertion_requires_observed_contradiction": True,
        "transfer_scorer_executed": True,
        "fixed_transferred_slot_accounting": True,
        "transferred_slot_capacity": int(transferred_slot_capacity),
        "transferred_slot_comparisons_charged": int(
            sum(int(row["transferred_slot_comparisons_charged"]) for row in rows)
        ),
        "oracle_mapping_delivered": mode == "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
        "evaluation_mapping_delivered": mode == "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
        "evaluator_truth_gates_target_action": False,
        "per_episode": rows,
    }


def classify_exp290_root(metrics: Mapping[str, Any]) -> str:
    oracle_headroom = float(metrics["oracle_headroom"])
    if oracle_headroom <= 0.0:
        return ORACLE_NOT_REPLICATED
    learned_solution_floor = float(metrics["control_verified_solution_rate"]) - 0.01
    success = (
        float(metrics["learned_headroom"]) > 0.0
        and float(metrics["learned_oracle_value_capture"]) >= 0.50
        and float(metrics["learned_valid_state_overprune_rate"]) <= 0.005
        and float(metrics["learned_verified_solution_rate"]) >= learned_solution_floor
        and int(metrics["learned_transferred_prune_count"]) > 0
        and float(metrics["heldout_nonidentity_permutation_rate"]) == 1.0
        and metrics["evaluation_mapping_used_by_learned_mode"] is False
        and metrics["oracle_mapping_used_by_learned_mode"] is False
    )
    return ROOT_ESTABLISHED if success else ROOT_NOT_ESTABLISHED


def run_exp290_root(
    *,
    canonical_index: int,
    geometry: Mapping[str, object],
    protocol_digest: str,
    geometry_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    _validate_geometry(geometry, canonical_index)
    _require_digest("protocol_digest", protocol_digest)
    _require_digest("geometry_digest", geometry_digest)
    _require_digest("code_digest", code_digest)

    root_seed = _root_seed(geometry, canonical_index)
    generator = Exp290TransferGenerator(root_seed=root_seed)
    model, model_seed = _build_model(
        root_seed=root_seed,
        d_model=int(geometry["d_model"]),
        hidden_size=int(geometry["hidden_size"]),
        target_parameters=int(geometry["target_parameters"]),
    )
    training = _train_model(model=model, generator=generator, geometry=geometry)
    post_training_digest = str(training["post_training_model_digest"])
    model.eval()

    variables = int(geometry["variables"])
    transferred_slot_capacity = variables
    audit = audit_exp290_transfer_model(
        model,
        variables=variables,
        source_clause_slots=transferred_slot_capacity,
    )
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in TARGET_MODES}
    pair_digests: list[str] = []
    source_receipts: list[dict[str, Any]] = []
    nonidentity_count = 0

    with torch.no_grad():
        for offset in range(int(geometry["eval_replicates"])):
            replicate = int(geometry["eval_start_replicate"]) + offset
            pair = generator.make_pair(
                replicate=replicate,
                batch_size=int(geometry["batch_size"]),
                timesteps=int(geometry["timesteps"]),
                restarts=int(geometry["restarts"]),
                variables=variables,
                decoys=int(geometry["decoys"]),
                d_model=int(geometry["d_model"]),
                noise_std=float(geometry["noise_std"]),
                rng_stream="evaluation",
                device="cpu",
            )
            pair_digests.append(pair.digest)
            nonidentity_count += _nonidentity_episode_count(pair, variables)
            source_phase = build_common_source_phase(
                pair,
                max_search_steps=int(geometry["max_search_steps"]),
            )
            source_receipts.append(source_phase)
            for mode in TARGET_MODES:
                for episode_index, source_episode in enumerate(source_phase["episodes"]):
                    rows_by_mode[mode].append(
                        _evaluate_target_episode(
                            mode=mode,
                            pair=pair,
                            source_episode=source_episode,
                            episode_index=episode_index,
                            model=model,
                            max_search_steps=int(geometry["max_search_steps"]),
                            transferred_slot_capacity=transferred_slot_capacity,
                        )
                    )

    post_evaluation_digest = _state_digest(model)
    modes = {
        mode: _aggregate_mode(
            mode=mode,
            rows=rows_by_mode[mode],
            model_digest=post_evaluation_digest,
            transferred_slot_capacity=transferred_slot_capacity,
        )
        for mode in TARGET_MODES
    }
    control = modes["LOCAL_ONLY_CONTROL"]
    learned = modes["LEARNED_STRUCTURAL_TRANSFER"]
    oracle = modes["ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND"]
    control_rate = float(control["source_equivalent_target_dead_end_rate"])
    learned_rate = float(learned["source_equivalent_target_dead_end_rate"])
    oracle_rate = float(oracle["source_equivalent_target_dead_end_rate"])
    oracle_headroom = control_rate - oracle_rate
    learned_headroom = control_rate - learned_rate
    capture = learned_headroom / oracle_headroom if oracle_headroom > 0.0 else 0.0
    evaluation_episodes = int(geometry["eval_replicates"]) * int(geometry["batch_size"])

    root_metrics = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "control_source_equivalent_target_dead_end_rate": control_rate,
        "learned_source_equivalent_target_dead_end_rate": learned_rate,
        "oracle_source_equivalent_target_dead_end_rate": oracle_rate,
        "oracle_headroom": float(oracle_headroom),
        "learned_headroom": float(learned_headroom),
        "learned_oracle_value_capture": float(capture),
        "learned_valid_state_overprune_rate": float(learned["valid_state_overprune_rate"]),
        "control_verified_solution_rate": float(control["verified_solution_rate"]),
        "learned_verified_solution_rate": float(learned["verified_solution_rate"]),
        "oracle_verified_solution_rate": float(oracle["verified_solution_rate"]),
        "learned_transferred_prune_count": int(learned["transferred_prune_count"]),
        "heldout_nonidentity_permutation_rate": float(
            nonidentity_count / evaluation_episodes if evaluation_episodes else 0.0
        ),
        "evaluation_mapping_used_by_learned_mode": False,
        "oracle_mapping_used_by_learned_mode": False,
        "local_only_transfer_activated": False,
        "oracle_transfer_mode_deployable": False,
        "epsilon_denominator_rescue_used": False,
    }
    decision = classify_exp290_root(root_metrics)

    return {
        "schema": EXP290_ROOT_SCHEMA,
        "experiment_id": "EXP-290",
        "canonical_index": int(canonical_index),
        "root_seed": root_seed,
        "model_seed": int(model_seed),
        "evidence_level": "EV-E2",
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "evaluation_mapping_used_for_training": False,
        "evaluation_mapping_used_by_learned_mode": False,
        "oracle_mapping_used_by_learned_mode": False,
        "oracle_transfer_mode_deployable": False,
        "arbitrary_value_symbol_remapping_claimed": False,
        "cross_domain_transfer_claimed": False,
        "lemma_generation_claimed": False,
        "protocol_digest": protocol_digest,
        "geometry_digest": geometry_digest,
        "code_digest": code_digest,
        "geometry": dict(geometry),
        "training": training,
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": int(geometry["eval_start_replicate"]),
            "replicates": int(geometry["eval_replicates"]),
            "evaluation_episode_count": evaluation_episodes,
            "heldout_nonidentity_episode_count": int(nonidentity_count),
            "pair_digests": pair_digests,
            "common_source_phases": source_receipts,
            "modes": modes,
        },
        "model_audit": audit,
        "post_training_model_digest": post_training_digest,
        "post_evaluation_model_digest": post_evaluation_digest,
        "model_state_unchanged_during_evaluation": post_training_digest
        == post_evaluation_digest,
        "root_metrics": root_metrics,
        "decision": decision,
        "successor_design_authorized": False,
        "authorization_scope": "NONE",
        "mechanism_successor_authorized": False,
        "descriptive_notes": {
            "value_label_remapping_claimed": False,
            "cross_domain_transfer_claimed": False,
            "target_local_memory_matches_parent_exp289": True,
        },
    }
