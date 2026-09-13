from __future__ import annotations

from itertools import product
from statistics import mean
from typing import Any

import torch

from nolane_ai.experiments.exp290_correspondence import (
    Exp290CorrespondenceModel,
    exp290_correspondence_loss,
    exp290_model_state_digest,
    learned_row_top1_mapping,
)
from nolane_ai.experiments.exp290_transfer_geometry import EXP290_GEOMETRY
from nolane_ai.experiments.exp290_transfer_worlds import Exp290TransferGenerator
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.training.optimizer import (
    build_functional_optimizer,
    functional_trainable_named_parameters,
)


NULL_TRANSFER_CONTROL = "NULL_TRANSFER_CONTROL"
RAW_SURFACE_TRANSFER_CONTROL = "RAW_SURFACE_TRANSFER_CONTROL"
LEARNED_STRUCTURAL_TRANSFER = "LEARNED_STRUCTURAL_TRANSFER"
ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND = "ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND"

_EXP290_MODE_ORDER = (
    NULL_TRANSFER_CONTROL,
    RAW_SURFACE_TRANSFER_CONTROL,
    LEARNED_STRUCTURAL_TRANSFER,
    ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND,
)
_EXP290_MODES = set(_EXP290_MODE_ORDER)


def _canonical_clause(clause: list[list[Any]]) -> list[list[Any]]:
    return [
        [str(name), int(value)]
        for name, value in sorted(
            ((str(name), int(value)) for name, value in clause),
            key=lambda item: item[0],
        )
    ]


def _clause_key(clause: list[list[Any]]) -> tuple[tuple[str, int], ...]:
    return tuple((str(name), int(value)) for name, value in _canonical_clause(clause))


def _public_contradiction(problem: dict[str, Any], assignment: dict[str, int]) -> bool:
    for constraint in problem["constraints"]:
        scope = [str(name) for name in constraint["scope"]]
        if not all(name in assignment for name in scope):
            continue
        values = tuple(int(assignment[name]) for name in scope)
        allowed = {
            tuple(int(value) for value in row)
            for row in constraint["allowed"]
        }
        if values not in allowed:
            return True
    return False


def _assignment_is_complete_and_valid(
    problem: dict[str, Any],
    assignment: dict[str, int],
) -> bool:
    names = [str(variable["name"]) for variable in problem["variables"]]
    return set(assignment) == set(names) and not _public_contradiction(problem, assignment)


def _has_valid_completion(
    problem: dict[str, Any],
    partial_assignment: dict[str, int],
) -> bool:
    variables = [str(variable["name"]) for variable in problem["variables"]]
    remaining = [name for name in variables if name not in partial_assignment]
    if _public_contradiction(problem, partial_assignment):
        return False
    for values in product((0, 1), repeat=len(remaining)):
        candidate = dict(partial_assignment)
        candidate.update(dict(zip(remaining, values, strict=True)))
        if _assignment_is_complete_and_valid(problem, candidate):
            return True
    return False


def _matching_clause(
    clauses: list[list[list[Any]]],
    assignment: dict[str, int],
) -> list[list[Any]] | None:
    for clause in clauses:
        if all(assignment.get(str(name)) == int(value) for name, value in clause):
            return clause
    return None


def map_source_clause_with_learned_indices(
    source_clause: list[list[Any]],
    *,
    source_names: list[str],
    target_names: list[str],
    learned_source_to_target: list[int],
) -> tuple[list[list[Any]] | None, str]:
    if len(learned_source_to_target) != len(source_names):
        raise ValueError("learned mapping length must equal source variable count")
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        raise ValueError("source and target surface names must be unique")
    source_index = {str(name): index for index, name in enumerate(source_names)}
    mapped: list[list[Any]] = []
    used_target_indices: set[int] = set()
    for name, value in source_clause:
        key = str(name)
        if key not in source_index:
            raise ValueError("source clause references unknown source variable")
        target_index = int(learned_source_to_target[source_index[key]])
        if target_index < 0 or target_index >= len(target_names):
            raise ValueError("learned mapping target index out of range")
        if target_index in used_target_indices:
            return None, "mapping_collision"
        used_target_indices.add(target_index)
        mapped.append([str(target_names[target_index]), int(value)])
    return _canonical_clause(mapped), "accepted"


def _build_learned_transferred_clauses(
    *,
    source_clauses: list[list[list[Any]]],
    source_names: list[str],
    target_names: list[str],
    learned_source_to_target: list[int],
) -> tuple[list[list[list[Any]]], list[dict[str, Any]]]:
    accepted: list[list[list[Any]]] = []
    receipts: list[dict[str, Any]] = []
    for source_clause in source_clauses:
        mapped, reason = map_source_clause_with_learned_indices(
            source_clause,
            source_names=source_names,
            target_names=target_names,
            learned_source_to_target=learned_source_to_target,
        )
        receipts.append(
            {
                "source_clause": _canonical_clause(source_clause),
                "mapped_target_clause": mapped,
                "status": reason,
                "oracle_fallback_used": False,
                "evaluator_truth_used": False,
            }
        )
        if mapped is not None:
            accepted.append(mapped)
    return accepted, receipts


def _build_oracle_transferred_clauses(
    *,
    pair: dict[str, Any],
    source_clauses: list[list[list[Any]]],
) -> list[list[list[Any]]]:
    source_names = [str(name) for name in pair["source_surface_names"]]
    target_names = [str(name) for name in pair["target_surface_names"]]
    mapping = [int(index) for index in pair["hidden_source_to_target_index"]]
    transferred: list[list[list[Any]]] = []
    for source_clause in source_clauses:
        mapped, reason = map_source_clause_with_learned_indices(
            source_clause,
            source_names=source_names,
            target_names=target_names,
            learned_source_to_target=mapping,
        )
        if mapped is None or reason != "accepted":
            raise RuntimeError("oracle source-to-target bijection produced an invalid clause")
        transferred.append(mapped)
    return transferred


def acquire_source_clauses(
    *,
    pair: dict[str, Any],
    restart_orders: Any,
    restart_value_orders: Any,
    max_search_steps: int,
) -> dict[str, Any]:
    if max_search_steps <= 0:
        raise ValueError("max_search_steps must be positive")
    problem = pair["source_problem"]
    source_names = [str(name) for name in pair["source_surface_names"]]
    clauses: list[list[list[Any]]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    receipts: list[dict[str, Any]] = []
    steps = 0

    for restart_index in range(int(restart_orders.shape[0])):
        assignment: dict[str, int] = {}
        restart_terminated = False
        for variable_tensor in restart_orders[restart_index]:
            variable_index = int(variable_tensor.item())
            variable_name = source_names[variable_index]
            accepted = False
            for value_tensor in restart_value_orders[restart_index, variable_index]:
                if steps >= max_search_steps:
                    restart_terminated = True
                    break
                steps += 1
                value = int(value_tensor.item())
                candidate = dict(assignment)
                candidate[variable_name] = value
                if _public_contradiction(problem, candidate):
                    if len(candidate) == 2:
                        clause = _canonical_clause(
                            [[name, literal_value] for name, literal_value in candidate.items()]
                        )
                        key = _clause_key(clause)
                        inserted = key not in seen
                        if inserted:
                            seen.add(key)
                            clauses.append(clause)
                        receipts.append(
                            {
                                "restart_index": restart_index,
                                "source_clause": clause,
                                "partial_assignment_reached": True,
                                "dead_end_observed_before_insertion": True,
                                "inserted": inserted,
                                "evaluator_truth_used_for_insertion": False,
                                "oracle_conflict_core_used": False,
                            }
                        )
                    restart_terminated = True
                    break
                assignment = candidate
                accepted = True
                break
            if restart_terminated or steps >= max_search_steps or not accepted:
                break

    return {
        "clauses": clauses,
        "receipts": receipts,
        "source_search_steps": int(steps),
        "source_clause_count": len(clauses),
        "evaluator_truth_used_for_insertion": False,
    }


def _run_target_search(
    *,
    problem: dict[str, Any],
    target_names: list[str],
    transferred_clauses: list[list[list[Any]]],
    restart_orders: Any,
    restart_value_orders: Any,
    max_search_steps: int,
) -> dict[str, Any]:
    steps = 0
    clause_comparisons = 0
    candidate_evaluation_operations = 0
    transferred_clause_hits = 0
    observed_dead_ends: list[list[list[Any]]] = []
    prune_receipts: list[dict[str, Any]] = []
    completed_valid_restarts = 0
    candidate_step_cost = len(problem["variables"]) + len(problem["constraints"])

    for restart_index in range(int(restart_orders.shape[0])):
        assignment: dict[str, int] = {}
        restart_terminated = False
        for variable_tensor in restart_orders[restart_index]:
            variable_index = int(variable_tensor.item())
            variable_name = target_names[variable_index]
            accepted = False
            for value_tensor in restart_value_orders[restart_index, variable_index]:
                if steps >= max_search_steps:
                    restart_terminated = True
                    break
                steps += 1
                candidate_evaluation_operations += candidate_step_cost
                value = int(value_tensor.item())
                candidate = dict(assignment)
                candidate[variable_name] = value

                clause_comparisons += len(transferred_clauses)
                matched = _matching_clause(transferred_clauses, candidate)
                if matched is not None:
                    transferred_clause_hits += 1
                    prune_receipts.append(
                        {
                            "restart_index": restart_index,
                            "candidate_assignment": _canonical_clause(
                                [[name, literal] for name, literal in candidate.items()]
                            ),
                            "matched_clause": _canonical_clause(matched),
                        }
                    )
                    continue

                if _public_contradiction(problem, candidate):
                    if len(candidate) == 2:
                        observed_dead_ends.append(
                            _canonical_clause(
                                [[name, literal] for name, literal in candidate.items()]
                            )
                        )
                    restart_terminated = True
                    break

                assignment = candidate
                accepted = True
                break
            if restart_terminated or steps >= max_search_steps or not accepted:
                break

        if _assignment_is_complete_and_valid(problem, assignment):
            completed_valid_restarts += 1
            break

    return {
        "search_steps": int(steps),
        "candidate_step_cost": int(candidate_step_cost),
        "candidate_evaluation_operations": int(candidate_evaluation_operations),
        "clause_comparisons": int(clause_comparisons),
        "transferred_clause_hit_count": int(transferred_clause_hits),
        "observed_dead_ends": observed_dead_ends,
        "prune_receipts": prune_receipts,
        "completed_valid_restarts": int(completed_valid_restarts),
        "verified_solution": completed_valid_restarts > 0,
    }


def run_target_mode(
    *,
    mode: str,
    pair: dict[str, Any],
    source_clauses: list[list[list[Any]]],
    learned_source_to_target: list[int],
    restart_orders: Any,
    restart_value_orders: Any,
    max_search_steps: int,
) -> dict[str, Any]:
    if mode not in _EXP290_MODES:
        raise ValueError(f"unknown EXP-290 transfer mode: {mode}")
    if max_search_steps <= 0:
        raise ValueError("max_search_steps must be positive")

    source_names = [str(name) for name in pair["source_surface_names"]]
    target_names = [str(name) for name in pair["target_surface_names"]]
    if set(source_names) & set(target_names):
        raise ValueError("EXP-290 source and target surface namespaces must be disjoint")

    mapping_receipts: list[dict[str, Any]] = []
    oracle_correspondence_delivered = False
    if mode == NULL_TRANSFER_CONTROL:
        transferred: list[list[list[Any]]] = []
    elif mode == RAW_SURFACE_TRANSFER_CONTROL:
        transferred = [_canonical_clause(clause) for clause in source_clauses]
    elif mode == LEARNED_STRUCTURAL_TRANSFER:
        transferred, mapping_receipts = _build_learned_transferred_clauses(
            source_clauses=source_clauses,
            source_names=source_names,
            target_names=target_names,
            learned_source_to_target=learned_source_to_target,
        )
    else:
        transferred = _build_oracle_transferred_clauses(
            pair=pair,
            source_clauses=source_clauses,
        )
        oracle_correspondence_delivered = True

    search = _run_target_search(
        problem=pair["target_problem"],
        target_names=target_names,
        transferred_clauses=transferred,
        restart_orders=restart_orders,
        restart_value_orders=restart_value_orders,
        max_search_steps=max_search_steps,
    )

    manifest_keys = {
        _clause_key(item["mapped_target_clause"])
        for item in pair["target_transfer_opportunities"]
    }
    structural_reentries = sum(
        1
        for clause in search["observed_dead_ends"]
        if _clause_key(clause) in manifest_keys
    )
    invalid_prunes = 0
    for prune in search["prune_receipts"]:
        partial = {
            str(name): int(value)
            for name, value in prune["candidate_assignment"]
        }
        if _has_valid_completion(pair["target_problem"], partial):
            invalid_prunes += 1
    prune_count = len(search["prune_receipts"])
    overprune_rate = float(invalid_prunes / prune_count) if prune_count else 0.0

    return {
        "mode": mode,
        "transferred_clause_count": len(transferred),
        "mapping_receipts": mapping_receipts,
        "mapping_collision_count": sum(
            1 for row in mapping_receipts if row["status"] == "mapping_collision"
        ),
        "transferred_clause_hit_count": int(search["transferred_clause_hit_count"]),
        "structural_repeat_dead_end_reentries": int(structural_reentries),
        "predeclared_transfer_opportunities": len(manifest_keys),
        "structural_repeat_dead_end_rate": (
            float(structural_reentries / len(manifest_keys)) if manifest_keys else None
        ),
        "invalid_transferred_prune_count": int(invalid_prunes),
        "transferred_prune_event_count": int(prune_count),
        "valid_state_overprune_rate": overprune_rate,
        "verified_solution": bool(search["verified_solution"]),
        "verified_solution_rate": 1.0 if search["verified_solution"] else 0.0,
        "search_steps": int(search["search_steps"]),
        "candidate_step_cost": int(search["candidate_step_cost"]),
        "candidate_evaluation_operations": int(search["candidate_evaluation_operations"]),
        "clause_comparisons": int(search["clause_comparisons"]),
        "search_accounted_operations": int(
            search["candidate_evaluation_operations"] + search["clause_comparisons"]
        ),
        "target_local_clause_learning_enabled": False,
        "oracle_correspondence_delivered": oracle_correspondence_delivered,
        "evaluator_validity_truth_delivered": False,
        "target_clause_truth_delivered": False,
        "surface_namespaces_disjoint": True,
        "raw_surface_transfer_hit_count": (
            int(search["transferred_clause_hit_count"])
            if mode == RAW_SURFACE_TRANSFER_CONTROL
            else 0
        ),
        "posthoc_safety_audit_only": True,
    }


def classify_exp290_root(metrics: dict[str, object]) -> str:
    oracle_headroom = float(metrics["oracle_headroom"])
    oracle_repeat_reduction = float(metrics["oracle_structural_repeat_relative_reduction"])
    if oracle_headroom <= 0.0 or oracle_repeat_reduction < 0.25:
        return "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"

    null_repeat = float(metrics["null_structural_repeat_dead_end_rate"])
    learned_repeat = float(metrics["learned_structural_repeat_dead_end_rate"])
    null_solution = float(metrics["null_verified_solution_rate"])
    learned_solution = float(metrics["learned_verified_solution_rate"])
    learned_ok = all(
        (
            float(metrics["learned_headroom"]) > 0.0,
            float(metrics["learned_oracle_value_capture"]) >= 0.50,
            float(metrics["learned_source_to_target_correspondence_accuracy"]) > 0.50,
            float(metrics["learned_exact_transferred_clause_recovery_rate"]) > 0.50,
            learned_repeat < null_repeat,
            learned_solution >= null_solution - 0.01,
            float(metrics["learned_valid_state_overprune_rate"]) <= 0.005,
            not bool(metrics["learned_oracle_correspondence_delivered"]),
            not bool(metrics["learned_evaluator_validity_truth_delivered"]),
            not bool(metrics["learned_target_clause_truth_delivered"]),
            int(metrics["raw_surface_transfer_hit_count"]) == 0,
            bool(metrics["surface_namespaces_disjoint"]),
        )
    )
    return (
        "LEARNED_CLAUSE_TRANSFER_ESTABLISHED"
        if learned_ok
        else "LEARNED_CLAUSE_TRANSFER_NOT_ESTABLISHED"
    )


def _inverse_bijection(source_to_target: list[int]) -> list[int]:
    inverse = [-1] * len(source_to_target)
    for source_index, target_index in enumerate(source_to_target):
        if target_index < 0 or target_index >= len(source_to_target):
            raise ValueError("correspondence label is outside target range")
        if inverse[target_index] != -1:
            raise ValueError("correspondence labels must define a bijection")
        inverse[target_index] = source_index
    if any(index < 0 for index in inverse):
        raise ValueError("correspondence labels must define a complete bijection")
    return inverse


def _functional_parameter_count(model: torch.nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for _, parameter in functional_trainable_named_parameters(model)
        )
    )


def _aggregate_rate(rows: list[dict[str, Any]], *, numerator: str, denominator: str) -> float:
    den = sum(int(row[denominator]) for row in rows)
    num = sum(int(row[numerator]) for row in rows)
    return float(num / den) if den > 0 else 0.0


def _run_exp290_root(
    *,
    root_seed: str,
    train_replicates: int,
    eval_replicates: int,
    eval_start_replicate: int,
) -> dict[str, Any]:
    """Run one frozen EXP-290 DEVELOPMENT root.

    This helper deliberately accepts only lineage counts needed by tiny tests. The
    public canonical wrapper introduced with the receipt/CLI layer supplies the
    frozen V1 geometry from EXP290_GEOMETRY rather than exposing tuning knobs.
    """

    if not root_seed:
        raise ValueError("root_seed must be non-empty")
    if min(train_replicates, eval_replicates) <= 0:
        raise ValueError("train_replicates and eval_replicates must be positive")
    if eval_start_replicate < 0:
        raise ValueError("eval_start_replicate must be non-negative")
    train_lineage = set(range(train_replicates))
    eval_lineage = set(range(eval_start_replicate, eval_start_replicate + eval_replicates))
    if train_lineage & eval_lineage:
        raise ValueError("training and evaluation replicate lineages must be disjoint")

    geometry = EXP290_GEOMETRY
    generator = Exp290TransferGenerator(root_seed=root_seed)
    model_seed = derive_stream_seed(root_seed, "EXP-290", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = Exp290CorrespondenceModel(
            geometry.d_model,
            geometry.hidden_size,
            geometry.target_parameters,
            device="cpu",
        )

    optimizer = build_functional_optimizer(
        model,
        lr=geometry.lr,
        weight_decay=geometry.weight_decay,
    )
    train_rows: list[dict[str, Any]] = []
    model.train()
    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            rng_stream="augmentation",
            device="cpu",
        )
        source_labels = torch.tensor(
            [pair["hidden_source_to_target_index"] for pair in batch.metadata["pairs"]],
            dtype=torch.long,
        )
        target_labels = torch.tensor(
            [
                _inverse_bijection([int(index) for index in pair["hidden_source_to_target_index"]])
                for pair in batch.metadata["pairs"]
            ],
            dtype=torch.long,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.source_surface_events,
            batch.target_surface_events,
            batch.source_variable_states,
            batch.target_variable_states,
        )
        loss, loss_audit = exp290_correspondence_loss(
            output.source_to_target_similarity,
            output.target_to_source_similarity,
            source_to_target_labels=source_labels,
            target_to_source_labels=target_labels,
        )
        loss.backward()
        optimizer.step()
        train_rows.append(
            {
                "replicate": int(replicate),
                "batch_digest": batch.digest,
                "loss": float(loss.detach().item()),
            }
        )

    post_training_digest = exp290_model_state_digest(model)
    functional_parameters = _functional_parameter_count(model)
    correspondence_inference_operations_per_pair = int(2 * functional_parameters)
    candidate_step_cost = int(geometry.variables + 3)

    mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in _EXP290_MODE_ORDER}
    mode_costs: dict[str, list[float]] = {mode: [] for mode in _EXP290_MODE_ORDER}
    mode_model_digests: dict[str, str] = {}
    correspondence_correct = 0
    correspondence_total = 0
    clause_recovery_correct = 0
    clause_recovery_total = 0
    pair_receipts: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for replicate in range(eval_start_replicate, eval_start_replicate + eval_replicates):
            batch = generator.make_batch(
                replicate=replicate,
                rng_stream="evaluation",
                device="cpu",
            )
            per_mode_outputs: dict[str, Any] = {}
            for mode in _EXP290_MODE_ORDER:
                per_mode_outputs[mode] = model(
                    batch.source_surface_events,
                    batch.target_surface_events,
                    batch.source_variable_states,
                    batch.target_variable_states,
                )
                digest = exp290_model_state_digest(model)
                previous = mode_model_digests.get(mode)
                if previous is not None and previous != digest:
                    raise RuntimeError("EXP-290 model state drifted within an evaluation mode")
                mode_model_digests[mode] = digest
                if digest != post_training_digest:
                    raise RuntimeError("EXP-290 model state drifted after training freeze")

            learned_mapping_batch = learned_row_top1_mapping(
                per_mode_outputs[LEARNED_STRUCTURAL_TRANSFER].source_to_target_similarity
            )

            for pair_index, pair in enumerate(batch.metadata["pairs"]):
                source = acquire_source_clauses(
                    pair=pair,
                    restart_orders=batch.source_restart_orders[pair_index],
                    restart_value_orders=batch.source_restart_value_orders[pair_index],
                    max_search_steps=geometry.max_search_steps,
                )
                source_clauses = source["clauses"]
                learned_mapping = [
                    int(index)
                    for index in learned_mapping_batch[pair_index].tolist()
                ]
                hidden_mapping = [
                    int(index) for index in pair["hidden_source_to_target_index"]
                ]
                correspondence_correct += sum(
                    int(predicted == truth)
                    for predicted, truth in zip(learned_mapping, hidden_mapping, strict=True)
                )
                correspondence_total += len(hidden_mapping)

                learned_clauses, _ = _build_learned_transferred_clauses(
                    source_clauses=source_clauses,
                    source_names=[str(name) for name in pair["source_surface_names"]],
                    target_names=[str(name) for name in pair["target_surface_names"]],
                    learned_source_to_target=learned_mapping,
                )
                oracle_clauses = _build_oracle_transferred_clauses(
                    pair=pair,
                    source_clauses=source_clauses,
                )
                learned_keys = {_clause_key(clause) for clause in learned_clauses}
                oracle_keys = {_clause_key(clause) for clause in oracle_clauses}
                clause_recovery_correct += sum(
                    int(_clause_key(clause) in learned_keys)
                    for clause in oracle_clauses
                )
                clause_recovery_total += len(oracle_keys)

                source_cost = int(
                    source["source_search_steps"] * candidate_step_cost
                    + sum(len(clause) for clause in source_clauses)
                )
                pair_modes: dict[str, dict[str, Any]] = {}
                for mode in _EXP290_MODE_ORDER:
                    mode_result = run_target_mode(
                        mode=mode,
                        pair=pair,
                        source_clauses=source_clauses,
                        learned_source_to_target=learned_mapping,
                        restart_orders=batch.target_restart_orders[pair_index],
                        restart_value_orders=batch.target_restart_value_orders[pair_index],
                        max_search_steps=geometry.max_search_steps,
                    )
                    # Correspondence is executed and charged in every arm. Mapping
                    # selection is likewise charged equally so only transferred
                    # clause/search behavior can create economic headroom.
                    matched_mapping_operations = geometry.variables
                    total_cost = float(
                        source_cost
                        + correspondence_inference_operations_per_pair
                        + matched_mapping_operations
                        + int(mode_result["search_accounted_operations"])
                    )
                    row = {
                        **mode_result,
                        "replicate": int(replicate),
                        "pair_index": int(pair_index),
                        "batch_digest": batch.digest,
                        "source_clause_count": int(len(source_clauses)),
                        "source_cost": int(source_cost),
                        "correspondence_inference_operations": correspondence_inference_operations_per_pair,
                        "matched_mapping_operations": int(matched_mapping_operations),
                        "accounted_cost": total_cost,
                        "model_digest": post_training_digest,
                    }
                    mode_rows[mode].append(row)
                    mode_costs[mode].append(total_cost)
                    pair_modes[mode] = row

                pair_receipts.append(
                    {
                        "replicate": int(replicate),
                        "pair_index": int(pair_index),
                        "batch_digest": batch.digest,
                        "source_clause_receipts": source["receipts"],
                        "source_clause_count": int(len(source_clauses)),
                        "learned_mapping": learned_mapping,
                        "modes": pair_modes,
                    }
                )

    post_evaluation_digest = exp290_model_state_digest(model)
    if post_evaluation_digest != post_training_digest:
        raise RuntimeError("EXP-290 model state changed during frozen evaluation")

    def _mean_cost(mode: str) -> float:
        return float(mean(mode_costs[mode]))

    c0 = _mean_cost(NULL_TRANSFER_CONTROL)
    cr = _mean_cost(RAW_SURFACE_TRANSFER_CONTROL)
    cl = _mean_cost(LEARNED_STRUCTURAL_TRANSFER)
    co = _mean_cost(ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND)
    oracle_headroom = float(c0 - co)
    learned_headroom = float(c0 - cl)
    capture = float(learned_headroom / oracle_headroom) if oracle_headroom > 0.0 else 0.0

    null_rate = _aggregate_rate(
        mode_rows[NULL_TRANSFER_CONTROL],
        numerator="structural_repeat_dead_end_reentries",
        denominator="predeclared_transfer_opportunities",
    )
    learned_rate = _aggregate_rate(
        mode_rows[LEARNED_STRUCTURAL_TRANSFER],
        numerator="structural_repeat_dead_end_reentries",
        denominator="predeclared_transfer_opportunities",
    )
    oracle_rate = _aggregate_rate(
        mode_rows[ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND],
        numerator="structural_repeat_dead_end_reentries",
        denominator="predeclared_transfer_opportunities",
    )
    oracle_repeat_reduction = (
        float((null_rate - oracle_rate) / null_rate) if null_rate > 0.0 else 0.0
    )
    learned_prunes = sum(
        int(row["transferred_prune_event_count"])
        for row in mode_rows[LEARNED_STRUCTURAL_TRANSFER]
    )
    learned_invalid_prunes = sum(
        int(row["invalid_transferred_prune_count"])
        for row in mode_rows[LEARNED_STRUCTURAL_TRANSFER]
    )
    learned_overprune = (
        float(learned_invalid_prunes / learned_prunes) if learned_prunes > 0 else 0.0
    )

    root_metrics: dict[str, object] = {
        "null_cost": c0,
        "raw_cost": cr,
        "learned_cost": cl,
        "oracle_cost": co,
        "oracle_headroom": oracle_headroom,
        "learned_headroom": learned_headroom,
        "learned_oracle_value_capture": capture,
        "null_structural_repeat_dead_end_rate": null_rate,
        "oracle_structural_repeat_dead_end_rate": oracle_rate,
        "oracle_structural_repeat_relative_reduction": oracle_repeat_reduction,
        "learned_structural_repeat_dead_end_rate": learned_rate,
        "learned_source_to_target_correspondence_accuracy": (
            float(correspondence_correct / correspondence_total)
            if correspondence_total > 0
            else 0.0
        ),
        "learned_exact_transferred_clause_recovery_rate": (
            float(clause_recovery_correct / clause_recovery_total)
            if clause_recovery_total > 0
            else 0.0
        ),
        "null_verified_solution_rate": float(
            mean(float(row["verified_solution_rate"]) for row in mode_rows[NULL_TRANSFER_CONTROL])
        ),
        "learned_verified_solution_rate": float(
            mean(float(row["verified_solution_rate"]) for row in mode_rows[LEARNED_STRUCTURAL_TRANSFER])
        ),
        "learned_valid_state_overprune_rate": learned_overprune,
        "learned_oracle_correspondence_delivered": False,
        "learned_evaluator_validity_truth_delivered": False,
        "learned_target_clause_truth_delivered": False,
        "raw_surface_transfer_hit_count": int(
            sum(int(row["raw_surface_transfer_hit_count"]) for row in mode_rows[RAW_SURFACE_TRANSFER_CONTROL])
        ),
        "surface_namespaces_disjoint": all(
            bool(row["surface_namespaces_disjoint"])
            for mode in _EXP290_MODE_ORDER
            for row in mode_rows[mode]
        ),
    }
    decision = classify_exp290_root(root_metrics)

    return {
        "schema": "NLM-EXP-290-LEARNED-CLAUSE-TRANSFER-ROOT-V1",
        "experiment_id": "EXP-290",
        "root_seed": root_seed,
        "model_init_seed": int(model_seed),
        "training": {
            "rng_stream": "augmentation",
            "replicates": int(train_replicates),
            "per_replicate": train_rows,
            "evaluation_lineage_consumed": False,
            "evaluation_correspondence_used_for_training": False,
            "loss_weights": {
                "source_to_target_ce": 0.5,
                "target_to_source_ce": 0.5,
            },
            "calibration_used": False,
            "early_stopping": False,
            "hard_negative_mining": False,
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": int(eval_start_replicate),
            "replicates": int(eval_replicates),
            "pair_count": int(eval_replicates * geometry.batch_size),
            "mode_model_digests": mode_model_digests,
            "pair_receipts": pair_receipts,
            "raw_surface_transfer_hit_count": int(root_metrics["raw_surface_transfer_hit_count"]),
            "surface_namespaces_disjoint": bool(root_metrics["surface_namespaces_disjoint"]),
            "learned_oracle_correspondence_delivered": False,
            "learned_evaluator_validity_truth_delivered": False,
            "learned_target_clause_truth_delivered": False,
        },
        "post_training_model_digest": post_training_digest,
        "post_evaluation_model_digest": post_evaluation_digest,
        "functional_parameter_count": functional_parameters,
        "correspondence_inference_operations_per_pair": correspondence_inference_operations_per_pair,
        "candidate_step_cost": candidate_step_cost,
        "root_metrics": root_metrics,
        "decision": decision,
        "target_local_clause_learning_enabled": False,
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "oracle_mode_deployable": False,
        "lifelong_clause_reuse_claimed": False,
        "semantic_authority_claimed": False,
    }
