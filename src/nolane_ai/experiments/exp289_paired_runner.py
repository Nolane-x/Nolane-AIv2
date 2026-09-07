from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from statistics import mean
from typing import Any

import torch
from torch.nn import functional as F

from nolane_ai.experiments.exp289_nogood_worlds import (
    Exp289NogoodBatch,
    Exp289NogoodGenerator,
    problem_from_exp289_episode,
)
from nolane_ai.experiments.matched_nogood_arms import (
    EpisodeScopedNogoodStore,
    LocalNogoodArm,
    NoNogoodArm,
    audit_matched_exp289_arm_pair,
    build_matched_exp289_arm_pair,
)
from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.reasoning.cps import CanonicalProblemState
from nolane_ai.training.optimizer import (
    build_functional_optimizer,
    functional_trainable_named_parameters,
)
from nolane_ai.training.tensor_bytes import tensor_byteorder, tensor_raw_bytes


EXP289_ARTIFACT_SCHEMA = "NLM-EXP-289-PAIRED-DEV-EVAL-V1"
_PROTOCOL_ID = "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
_ARM_ORDER = ("no_nogood", "local_nogood")
_PRIMARY_ENDPOINT = {
    "metric": "repeat_dead_end_rate",
    "direction": "lower",
    "mesi_relative_reduction": 0.25,
}
_PROTECTED_ENDPOINTS = {
    "valid_state_overprune_rate_ceiling": 0.005,
    "verified_solution_rate_floor": "local_nogood >= no_nogood - 0.01",
}
_SCOPE_POLICY = {
    "scope": "episode_local",
    "cross_episode_reuse": False,
    "cross_problem_reuse": False,
    "exact_subset_matching_only": True,
    "oracle_conflict_core_used": False,
    "ground_truth_gates_arm_action": False,
    "learned_clause_generalization_claimed": False,
}
_ANALYSIS_BOUNDARY = (
    "descriptive DEVELOPMENT statistics only; frozen confirmatory bootstrap inference not executed"
)
_SEARCH_PATH_PAIRING_POLICY = (
    "same_initial_world_restart_lineage_with_causal_post_memory_divergence_preserved"
)
_MULTIPLICITY_FAMILY = "LOCAL_NOGOOD"
_ZERO_OPPORTUNITY_POLICY = "retain_raw_episode_exclude_from_rder_denominator"

_EXPECTED_ARTIFACT_CACHE: dict[str, dict[str, Any]] = {}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _artifact_digest(artifact: dict[str, Any]) -> str:
    payload = deepcopy(artifact)
    payload.pop("artifact_digest", None)
    return _json_digest(payload)


def _functional_state_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"NLM-EXP-289-FUNCTIONAL-STATE-V1\0")
    for name, parameter in functional_trainable_named_parameters(model):
        cpu = parameter.detach().cpu().contiguous()
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


def _build_seeded_pair(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
) -> tuple[NoNogoodArm, LocalNogoodArm, int]:
    model_seed = derive_stream_seed(root_seed, "EXP-289", 0, "model_init")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        no_nogood, local_nogood = build_matched_exp289_arm_pair(
            d_model=d_model,
            hidden_size=hidden_size,
            target_parameters=target_parameters,
            device="cpu",
        )
    return no_nogood, local_nogood, int(model_seed)


def _canonical_key_pairs(assignment: dict[str, int]) -> list[list[Any]]:
    return [[name, int(value)] for name, value in sorted(assignment.items())]


def _canonical_key_string(assignment: dict[str, int]) -> str:
    return json.dumps(_canonical_key_pairs(assignment), separators=(",", ":"))


def _pairs_to_assignment(pairs: list[list[Any]]) -> dict[str, int]:
    return {str(name): int(value) for name, value in pairs}


def _has_valid_completion(
    canonical_key: list[list[Any]],
    valid_solutions: list[list[list[Any]]],
) -> bool:
    partial = _pairs_to_assignment(canonical_key)
    for solution_pairs in valid_solutions:
        solution = _pairs_to_assignment(solution_pairs)
        if all(solution.get(name) == value for name, value in partial.items()):
            return True
    return False


def _mean_optional(values: list[float | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return float(mean(concrete)) if concrete else None


def _validate_run_args(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    train_replicates: int,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    restarts: int,
    variables: int,
    decoys: int,
    max_search_steps: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
) -> None:
    if not root_seed:
        raise ValueError("root_seed must be non-empty")
    if min(
        d_model,
        hidden_size,
        target_parameters,
        train_replicates,
        eval_replicates,
        batch_size,
        timesteps,
        restarts,
        variables,
        max_search_steps,
    ) <= 0:
        raise ValueError("EXP-289 runner geometry and replicate counts must be positive")
    if eval_start_replicate < 0:
        raise ValueError("eval_start_replicate must be non-negative")
    if decoys < 0 or decoys > variables - 1:
        raise ValueError("decoys must leave at least one productive variable")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if not protocol_digest or not code_digest:
        raise ValueError("protocol_digest and code_digest must be non-empty")

    training = set(range(train_replicates))
    evaluation = set(range(eval_start_replicate, eval_start_replicate + eval_replicates))
    if training & evaluation:
        raise ValueError("training and evaluation replicate lineages must be disjoint")


def _train_pair(
    *,
    no_nogood: NoNogoodArm,
    local_nogood: LocalNogoodArm,
    generator: Exp289NogoodGenerator,
    train_replicates: int,
    batch_size: int,
    timesteps: int,
    restarts: int,
    variables: int,
    decoys: int,
    d_model: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    no_optimizer = build_functional_optimizer(
        no_nogood,
        lr=lr,
        weight_decay=weight_decay,
    )
    local_optimizer = build_functional_optimizer(
        local_nogood,
        lr=lr,
        weight_decay=weight_decay,
    )
    paired_batch_digests: list[str] = []
    rows: list[dict[str, Any]] = []

    no_nogood.train()
    local_nogood.train()
    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            restarts=restarts,
            variables=variables,
            decoys=decoys,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="augmentation",
        )
        paired_batch_digests.append(batch.digest)
        target_variable = batch.restart_orders[:, 0, 0]
        verifier_target = batch.solution_targets.to(torch.float32)

        no_optimizer.zero_grad(set_to_none=True)
        no_output = no_nogood(batch.surface_events, batch.variable_states)
        no_loss = (
            F.cross_entropy(no_output.branch_logits, target_variable)
            + F.binary_cross_entropy(no_output.verifier_confidence, verifier_target)
            + 0.05
            * F.binary_cross_entropy_with_logits(
                no_output.memory_query_logit,
                torch.zeros_like(no_output.memory_query_logit),
            )
        )
        no_loss.backward()
        no_optimizer.step()

        local_optimizer.zero_grad(set_to_none=True)
        local_output = local_nogood(
            batch.surface_events,
            batch.variable_states,
            memory_hit=torch.zeros(batch_size),
        )
        local_loss = (
            F.cross_entropy(local_output.branch_logits, target_variable)
            + F.binary_cross_entropy(local_output.verifier_confidence, verifier_target)
            + 0.05
            * F.binary_cross_entropy_with_logits(
                local_output.memory_query_logit,
                torch.zeros_like(local_output.memory_query_logit),
            )
        )
        local_loss.backward()
        local_optimizer.step()

        rows.append(
            {
                "replicate": replicate,
                "paired_batch_digest": batch.digest,
                "no_nogood_loss": float(no_loss.detach().item()),
                "local_nogood_loss": float(local_loss.detach().item()),
                "memory_hit_training_signal": False,
                "evaluator_metadata_delivered_to_arm": False,
            }
        )

    no_digest = _functional_state_digest(no_nogood)
    local_digest = _functional_state_digest(local_nogood)
    return {
        "rng_stream": "augmentation",
        "start_replicate": 0,
        "replicates": int(train_replicates),
        "paired_batch_digests": paired_batch_digests,
        "per_replicate": rows,
        "post_training_no_nogood_digest": no_digest,
        "post_training_local_nogood_digest": local_digest,
        "post_training_functional_digest_match": no_digest == local_digest,
        "optimizer_family": "AdamW",
        "optimizer_hyperparameters": {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "decision_rule_executed": False,
    }


def _step_surface(batch: Exp289NogoodBatch, episode_index: int, step_index: int) -> torch.Tensor:
    timestep = step_index % batch.surface_events.shape[1]
    return batch.surface_events[
        episode_index : episode_index + 1,
        timestep : timestep + 1,
        :,
    ]


def _observed_public_contradiction(
    problem: CanonicalProblemState,
    assignment: dict[str, int],
) -> bool:
    """Return whether the currently reached partial assignment violates a public constraint.

    TableConstraint.is_satisfied deliberately treats incomplete scopes as not yet
    contradictory. This keeps evaluator-only valid-solution metadata out of the
    causal action/insertion path while still allowing the environment's public CSP
    semantics to report a contradiction once its scope has actually been reached.
    """

    return any(
        not constraint.is_satisfied(assignment)
        for constraint in problem.constraints
    )


def _run_episode(
    *,
    arm_name: str,
    model: NoNogoodArm | LocalNogoodArm,
    batch: Exp289NogoodBatch,
    episode_index: int,
    episode_metadata: dict[str, Any],
    max_search_steps: int,
    neural_flops_per_step: int,
    common_ceiling: int,
) -> dict[str, Any]:
    local_enabled = arm_name == "local_nogood"
    store = (
        EpisodeScopedNogoodStore(
            episode_digest=str(episode_metadata["episode_digest"]),
            problem_digest=str(episode_metadata["problem_digest"]),
        )
        if local_enabled
        else None
    )
    valid_solutions = deepcopy(episode_metadata["valid_solutions"])
    manifest = deepcopy(episode_metadata["repeat_opportunities"])
    manifest_by_key = {
        str(item["canonical_dead_end_key"]): item
        for item in manifest
    }
    problem = problem_from_exp289_episode(episode_metadata)
    repeated_dead_end_reentries = 0
    prevented_repeat_count = 0
    unique_dead_ends: set[str] = set()
    raw_dead_end_signatures: list[str] = []
    insertion_receipts: list[dict[str, Any]] = []
    prune_receipts: list[dict[str, Any]] = []
    step_receipts: list[dict[str, Any]] = []
    per_restart_dead_end_lineage: list[dict[str, Any]] = []
    invalid_prune_events = 0
    store_soundness_violations = 0
    neural_steps = 0
    censored = False
    candidate_solution: dict[str, int] | None = None

    with torch.no_grad():
        model.eval()
        for restart_index in range(batch.restart_orders.shape[1]):
            assignment: dict[str, int] = {}
            restart_dead_ends: list[str] = []
            restart_prevented: list[str] = []
            steps_this_restart = 0
            for variable_tensor in batch.restart_orders[episode_index, restart_index]:
                variable_index = int(variable_tensor.item())
                variable_name = f"v{variable_index}"
                accepted = False
                for value_tensor in batch.restart_value_orders[
                    episode_index,
                    restart_index,
                    variable_index,
                ]:
                    if steps_this_restart >= max_search_steps:
                        censored = True
                        break
                    value = int(value_tensor.item())
                    literal = {variable_name: value}
                    canonical_pairs = _canonical_key_pairs(literal)
                    canonical_string = _canonical_key_string(literal)

                    before_query = store.snapshot_counters() if store is not None else None
                    memory_hit = (
                        store.matches(
                            literal,
                            episode_digest=str(episode_metadata["episode_digest"]),
                            problem_digest=str(episode_metadata["problem_digest"]),
                        )
                        if store is not None
                        else False
                    )
                    after_query = store.snapshot_counters() if store is not None else None

                    surface = _step_surface(batch, episode_index, neural_steps)
                    variables = batch.variable_states[episode_index : episode_index + 1]
                    if local_enabled:
                        assert isinstance(model, LocalNogoodArm)
                        output = model(
                            surface,
                            variables,
                            memory_hit=torch.tensor([1.0 if memory_hit else 0.0]),
                        )
                    else:
                        assert isinstance(model, NoNogoodArm)
                        output = model(surface, variables)
                    neural_steps += 1
                    steps_this_restart += 1

                    step_receipt = {
                        "restart_index": restart_index,
                        "step_index_within_restart": steps_this_restart - 1,
                        "variable": variable_name,
                        "value": value,
                        "canonical_key": canonical_pairs,
                        "memory_hit": bool(memory_hit),
                        "memory_conditioning_used": bool(output.memory_conditioning_used),
                        "branch_score": float(output.branch_logits[0, variable_index].item()),
                        "verifier_confidence": float(
                            output.verifier_confidence[0, variable_index].item()
                        ),
                        "memory_query_logit": float(
                            output.memory_query_logit[0, variable_index].item()
                        ),
                        "charged_neural_flops": int(neural_flops_per_step),
                        "evaluator_metadata_delivered_to_arm": False,
                    }
                    if before_query is not None and after_query is not None:
                        step_receipt["charged_query_canonicalization_operations"] = int(
                            after_query["canonicalization_operations"]
                            - before_query["canonicalization_operations"]
                        )
                        step_receipt["charged_subset_comparison_operations"] = int(
                            after_query["comparison_count"]
                            - before_query["comparison_count"]
                        )
                    else:
                        step_receipt["charged_query_canonicalization_operations"] = 0
                        step_receipt["charged_subset_comparison_operations"] = 0
                    step_receipts.append(step_receipt)

                    opportunity = manifest_by_key.get(canonical_string)
                    is_later_opportunity = bool(
                        opportunity
                        and restart_index in opportunity["later_restart_indices"]
                    )
                    if memory_hit:
                        posthoc_has_valid_completion = _has_valid_completion(
                            canonical_pairs,
                            valid_solutions,
                        )
                        if posthoc_has_valid_completion:
                            invalid_prune_events += 1
                        if is_later_opportunity:
                            prevented_repeat_count += 1
                            restart_prevented.append(canonical_string)
                        prune_receipts.append(
                            {
                                "restart_index": restart_index,
                                "canonical_key": canonical_pairs,
                                "episode_digest": episode_metadata["episode_digest"],
                                "problem_digest": episode_metadata["problem_digest"],
                                "exact_subset_match": True,
                                "posthoc_has_valid_completion": bool(
                                    posthoc_has_valid_completion
                                ),
                                "invalid_valid_state_prune": bool(
                                    posthoc_has_valid_completion
                                ),
                                "predeclared_repeat_opportunity": bool(opportunity),
                                "prevented_repeat": bool(is_later_opportunity),
                                "evaluator_result_delivered_to_arm": False,
                            }
                        )
                        continue

                    candidate_assignment = dict(assignment)
                    candidate_assignment[variable_name] = value
                    dead_end_observed = _observed_public_contradiction(
                        problem,
                        candidate_assignment,
                    )
                    if dead_end_observed:
                        raw_dead_end_signatures.append(canonical_string)
                        unique_dead_ends.add(canonical_string)
                        restart_dead_ends.append(canonical_string)
                        if is_later_opportunity:
                            repeated_dead_end_reentries += 1
                        if store is not None:
                            before_insert = store.snapshot_counters()
                            inserted = store.add(
                                literal,
                                dead_end_observed=True,
                                episode_digest=str(episode_metadata["episode_digest"]),
                                problem_digest=str(episode_metadata["problem_digest"]),
                            )
                            after_insert = store.snapshot_counters()
                            posthoc_has_valid_completion = _has_valid_completion(
                                canonical_pairs,
                                valid_solutions,
                            )
                            if posthoc_has_valid_completion:
                                store_soundness_violations += 1
                            insertion_receipts.append(
                                {
                                    "restart_index": restart_index,
                                    "canonical_key": canonical_pairs,
                                    "partial_assignment_reached": True,
                                    "dead_end_observed_before_insertion": True,
                                    "inserted": bool(inserted),
                                    "episode_digest": episode_metadata["episode_digest"],
                                    "problem_digest": episode_metadata["problem_digest"],
                                    "oracle_conflict_core_used": False,
                                    "ground_truth_safety_gate_used": False,
                                    "future_path_used": False,
                                    "evaluator_result_delivered_to_arm": False,
                                    "posthoc_has_valid_completion": bool(
                                        posthoc_has_valid_completion
                                    ),
                                    "charged_canonicalization_operations": int(
                                        after_insert["canonicalization_operations"]
                                        - before_insert["canonicalization_operations"]
                                    ),
                                    "charged_insertion_operations": int(
                                        after_insert["insertion_count"]
                                        - before_insert["insertion_count"]
                                    ),
                                }
                            )
                        continue

                    assignment[variable_name] = value
                    accepted = True
                    break
                if censored or not accepted:
                    if censored:
                        break
                    # A productive value must exist in the frozen value schedule.
                    censored = True
                    break
            per_restart_dead_end_lineage.append(
                {
                    "restart_index": restart_index,
                    "observed_dead_end_signatures": restart_dead_ends,
                    "prevented_repeat_signatures": restart_prevented,
                    "assignment_after_restart": _canonical_key_pairs(assignment),
                }
            )
            if censored:
                break
            if restart_index == batch.restart_orders.shape[1] - 1:
                candidate_solution = dict(assignment)

    external_solution_verified = bool(
        not censored
        and candidate_solution is not None
        and problem.is_solution(candidate_solution)
    )

    if store is not None:
        counters = store.snapshot_counters()
    else:
        counters = {
            "insertion_count": 0,
            "query_count": 0,
            "comparison_count": 0,
            "hit_count": 0,
            "canonicalization_operations": 0,
        }
    neural_cost = int(neural_steps * neural_flops_per_step)
    memory_cost = int(
        counters["canonicalization_operations"]
        + counters["insertion_count"]
        + counters["comparison_count"]
    )
    raw_cost = int(neural_cost + memory_cost)
    if censored or raw_cost > common_ceiling:
        censored = True
        accounted_cost = int(common_ceiling)
        external_solution_verified = False
    else:
        accounted_cost = raw_cost

    denominator = int(episode_metadata["predeclared_repeat_opportunities"])
    rder = (
        float(repeated_dead_end_reentries / denominator)
        if denominator > 0
        else None
    )
    prune_count = len(prune_receipts)
    overprune_rate = (
        float(invalid_prune_events / prune_count)
        if prune_count > 0
        else 0.0
    )

    return {
        "episode_index": int(episode_index),
        "episode_digest": episode_metadata["episode_digest"],
        "problem_digest": episode_metadata["problem_digest"],
        "scope_receipt": {
            "episode_digest": episode_metadata["episode_digest"],
            "problem_digest": episode_metadata["problem_digest"],
            "scope": "episode_local",
            "cross_episode_reuse": False,
            "cross_problem_reuse": False,
        },
        "evaluator_metadata_delivered_to_arm": False,
        "predeclared_repeat_opportunities": denominator,
        "rder_defined": denominator > 0,
        "repeat_dead_end_rate": rder,
        "zero_opportunity_policy": _ZERO_OPPORTUNITY_POLICY,
        "repeated_dead_end_reentries": int(repeated_dead_end_reentries),
        "prevented_repeat_count": int(prevented_repeat_count),
        "unique_dead_end_count": int(len(unique_dead_ends)),
        "raw_dead_end_signatures": raw_dead_end_signatures,
        "per_restart_dead_end_lineage": per_restart_dead_end_lineage,
        "store_insertion_receipts": insertion_receipts,
        "prune_receipts": prune_receipts,
        "invalid_valid_state_prune_count": int(invalid_prune_events),
        "store_soundness_violation_count": int(store_soundness_violations),
        "valid_state_overprune_rate": overprune_rate,
        "memory_insertion_count": int(counters["insertion_count"]),
        "memory_query_count": int(counters["query_count"]),
        "memory_comparison_count": int(counters["comparison_count"]),
        "memory_canonicalization_operations": int(
            counters["canonicalization_operations"]
        ),
        "nogood_hits": int(counters["hit_count"]),
        "step_receipts": step_receipts,
        "cost_receipt": {
            "neural_steps": int(neural_steps),
            "neural_flops_per_step": int(neural_flops_per_step),
            "neural_accounted_flops": neural_cost,
            "memory_canonicalization_operations": int(
                counters["canonicalization_operations"]
            ),
            "memory_insertion_operations": int(counters["insertion_count"]),
            "memory_subset_comparison_operations": int(counters["comparison_count"]),
            "memory_accounted_operations": memory_cost,
            "raw_accounted_reasoning_cost": raw_cost,
            "common_ceiling": int(common_ceiling),
        },
        "accounted_reasoning_cost": int(accounted_cost),
        "censored_at_max_cost": bool(censored),
        "candidate_solution": (
            _canonical_key_pairs(candidate_solution)
            if candidate_solution is not None
            else None
        ),
        "external_solution_verified": bool(external_solution_verified),
        "verified_solution_rate": 1.0 if external_solution_verified else 0.0,
    }


def _summarize_arm(
    *,
    arm_name: str,
    episodes: list[dict[str, Any]],
    common_ceiling: int,
) -> dict[str, Any]:
    denominator = sum(
        int(episode["predeclared_repeat_opportunities"])
        for episode in episodes
    )
    numerator = sum(
        int(episode["repeated_dead_end_reentries"])
        for episode in episodes
    )
    prune_count = sum(len(episode["prune_receipts"]) for episode in episodes)
    invalid_prunes = sum(
        int(episode["invalid_valid_state_prune_count"])
        for episode in episodes
    )
    any_censored = any(bool(episode["censored_at_max_cost"]) for episode in episodes)
    if any_censored:
        accounted_cost: float | int = int(common_ceiling)
    else:
        accounted_cost = float(
            mean(float(episode["accounted_reasoning_cost"]) for episode in episodes)
        )
    return {
        "arm": arm_name,
        "predeclared_repeat_opportunities": int(denominator),
        "repeated_dead_end_reentries": int(numerator),
        "repeat_dead_end_rate": (
            float(numerator / denominator) if denominator > 0 else None
        ),
        "prevented_repeat_count": int(
            sum(int(episode["prevented_repeat_count"]) for episode in episodes)
        ),
        "unique_dead_end_count": int(
            sum(int(episode["unique_dead_end_count"]) for episode in episodes)
        ),
        "memory_insertion_count": int(
            sum(int(episode["memory_insertion_count"]) for episode in episodes)
        ),
        "memory_query_count": int(
            sum(int(episode["memory_query_count"]) for episode in episodes)
        ),
        "memory_comparison_count": int(
            sum(int(episode["memory_comparison_count"]) for episode in episodes)
        ),
        "memory_canonicalization_operations": int(
            sum(
                int(episode["memory_canonicalization_operations"])
                for episode in episodes
            )
        ),
        "nogood_hits": int(sum(int(episode["nogood_hits"]) for episode in episodes)),
        "valid_state_overprune_rate": (
            float(invalid_prunes / prune_count) if prune_count > 0 else 0.0
        ),
        "invalid_valid_state_prune_count": int(invalid_prunes),
        "nogood_prune_event_count": int(prune_count),
        "store_soundness_violation_count": int(
            sum(
                int(episode["store_soundness_violation_count"])
                for episode in episodes
            )
        ),
        "verified_solution_rate": float(
            mean(float(episode["verified_solution_rate"]) for episode in episodes)
        ),
        "accounted_reasoning_cost": accounted_cost,
        "censored_at_max_cost": bool(any_censored),
        "episodes": episodes,
    }


def _evaluate_pair(
    *,
    no_nogood: NoNogoodArm,
    local_nogood: LocalNogoodArm,
    generator: Exp289NogoodGenerator,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    restarts: int,
    variables: int,
    decoys: int,
    d_model: int,
    noise_std: float,
    max_search_steps: int,
    pair_audit: dict[str, Any],
) -> dict[str, Any]:
    common_ceiling = int(pair_audit["declared_max_accounted_cost_per_episode"])
    no_neural_per_step = int(
        pair_audit["compute_ledger"]["no_nogood"][
            "accounted_neural_flops_per_search_step"
        ]
    )
    local_neural_per_step = int(
        pair_audit["compute_ledger"]["local_nogood"][
            "accounted_neural_flops_per_search_step"
        ]
    )
    rows: list[dict[str, Any]] = []

    for offset in range(eval_replicates):
        replicate = eval_start_replicate + offset
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            restarts=restarts,
            variables=variables,
            decoys=decoys,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="evaluation",
        )
        no_episodes: list[dict[str, Any]] = []
        local_episodes: list[dict[str, Any]] = []
        world_receipts: list[dict[str, Any]] = []

        for episode_index, episode_metadata in enumerate(batch.metadata["episodes"]):
            no_episodes.append(
                _run_episode(
                    arm_name="no_nogood",
                    model=no_nogood,
                    batch=batch,
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    max_search_steps=max_search_steps,
                    neural_flops_per_step=no_neural_per_step,
                    common_ceiling=common_ceiling,
                )
            )
            local_episodes.append(
                _run_episode(
                    arm_name="local_nogood",
                    model=local_nogood,
                    batch=batch,
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    max_search_steps=max_search_steps,
                    neural_flops_per_step=local_neural_per_step,
                    common_ceiling=common_ceiling,
                )
            )
            manifest = deepcopy(episode_metadata["repeat_opportunities"])
            world_receipts.append(
                {
                    "episode_index": int(episode_index),
                    "episode_digest": episode_metadata["episode_digest"],
                    "problem_digest": episode_metadata["problem_digest"],
                    "valid_solutions": deepcopy(episode_metadata["valid_solutions"]),
                    "repeat_opportunities": manifest,
                    "predeclared_repeat_opportunities": int(
                        episode_metadata["predeclared_repeat_opportunities"]
                    ),
                    "opportunity_manifest_digest": _json_digest(manifest),
                    "evaluator_metadata_delivered_to_arm": False,
                }
            )

        rows.append(
            {
                "replicate": int(replicate),
                "paired_batch_digest": batch.digest,
                "initial_world_pairing_closed": True,
                "opportunity_manifest_pairing_closed": True,
                "evaluator_metadata_delivered_to_arm": False,
                "search_path_pairing_policy": _SEARCH_PATH_PAIRING_POLICY,
                "world_receipts": world_receipts,
                "no_nogood": _summarize_arm(
                    arm_name="no_nogood",
                    episodes=no_episodes,
                    common_ceiling=common_ceiling,
                ),
                "local_nogood": _summarize_arm(
                    arm_name="local_nogood",
                    episodes=local_episodes,
                    common_ceiling=common_ceiling,
                ),
            }
        )

    no_rates = [row["no_nogood"]["repeat_dead_end_rate"] for row in rows]
    local_rates = [row["local_nogood"]["repeat_dead_end_rate"] for row in rows]
    relative_reductions: list[float] = []
    for no_rate, local_rate in zip(no_rates, local_rates, strict=True):
        if no_rate is not None and local_rate is not None and float(no_rate) > 0.0:
            relative_reductions.append(
                float((float(no_rate) - float(local_rate)) / float(no_rate))
            )
    zero_opportunity_episode_count = sum(
        1
        for row in rows
        for receipt in row["world_receipts"]
        if int(receipt["predeclared_repeat_opportunities"]) == 0
    )
    aggregate = {
        "n": int(eval_replicates),
        "analysis_boundary": _ANALYSIS_BOUNDARY,
        "zero_opportunity_episode_count": int(zero_opportunity_episode_count),
        "mean_no_nogood_repeat_dead_end_rate": _mean_optional(no_rates),
        "mean_local_repeat_dead_end_rate": _mean_optional(local_rates),
        "mean_descriptive_relative_rder_reduction": (
            float(mean(relative_reductions)) if relative_reductions else None
        ),
        "mean_no_nogood_verified_solution_rate": float(
            mean(float(row["no_nogood"]["verified_solution_rate"]) for row in rows)
        ),
        "mean_local_verified_solution_rate": float(
            mean(float(row["local_nogood"]["verified_solution_rate"]) for row in rows)
        ),
        "mean_local_valid_state_overprune_rate": float(
            mean(float(row["local_nogood"]["valid_state_overprune_rate"]) for row in rows)
        ),
        "mean_no_nogood_accounted_reasoning_cost": float(
            mean(float(row["no_nogood"]["accounted_reasoning_cost"]) for row in rows)
        ),
        "mean_local_accounted_reasoning_cost": float(
            mean(float(row["local_nogood"]["accounted_reasoning_cost"]) for row in rows)
        ),
        "confirmatory_bootstrap_ci_computed": False,
        "scientific_decision_executed": False,
    }
    return {
        "rng_stream": "evaluation",
        "start_replicate": int(eval_start_replicate),
        "replicates": int(eval_replicates),
        "per_replicate": rows,
        "aggregate": aggregate,
    }


def _configuration_key(
    configuration: dict[str, Any],
    protocol_digest: str,
    code_digest: str,
) -> str:
    return _json_digest(
        {
            "configuration": configuration,
            "protocol_digest": protocol_digest,
            "code_digest": code_digest,
        }
    )


def _build_artifact(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    train_replicates: int,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    restarts: int,
    variables: int,
    decoys: int,
    max_search_steps: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
    cache_result: bool,
) -> dict[str, Any]:
    _validate_run_args(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        train_replicates=train_replicates,
        eval_replicates=eval_replicates,
        eval_start_replicate=eval_start_replicate,
        batch_size=batch_size,
        timesteps=timesteps,
        restarts=restarts,
        variables=variables,
        decoys=decoys,
        max_search_steps=max_search_steps,
        noise_std=noise_std,
        lr=lr,
        weight_decay=weight_decay,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
    )
    no_nogood, local_nogood, model_seed = _build_seeded_pair(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
    )
    no_initial_digest = _functional_state_digest(no_nogood)
    local_initial_digest = _functional_state_digest(local_nogood)
    pair_audit = audit_matched_exp289_arm_pair(
        no_nogood,
        local_nogood,
        restarts=restarts,
        variables=variables,
        max_search_steps=max_search_steps,
    )
    generator = Exp289NogoodGenerator(root_seed=root_seed)
    training = _train_pair(
        no_nogood=no_nogood,
        local_nogood=local_nogood,
        generator=generator,
        train_replicates=train_replicates,
        batch_size=batch_size,
        timesteps=timesteps,
        restarts=restarts,
        variables=variables,
        decoys=decoys,
        d_model=d_model,
        noise_std=noise_std,
        lr=lr,
        weight_decay=weight_decay,
    )
    evaluation = _evaluate_pair(
        no_nogood=no_nogood,
        local_nogood=local_nogood,
        generator=generator,
        eval_replicates=eval_replicates,
        eval_start_replicate=eval_start_replicate,
        batch_size=batch_size,
        timesteps=timesteps,
        restarts=restarts,
        variables=variables,
        decoys=decoys,
        d_model=d_model,
        noise_std=noise_std,
        max_search_steps=max_search_steps,
        pair_audit=pair_audit,
    )
    configuration = {
        "root_seed": root_seed,
        "d_model": int(d_model),
        "hidden_size": int(hidden_size),
        "target_parameters": int(target_parameters),
        "train_replicates": int(train_replicates),
        "eval_replicates": int(eval_replicates),
        "eval_start_replicate": int(eval_start_replicate),
        "batch_size": int(batch_size),
        "timesteps": int(timesteps),
        "restarts": int(restarts),
        "variables": int(variables),
        "decoys": int(decoys),
        "max_search_steps": int(max_search_steps),
        "noise_std": float(noise_std),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }
    artifact: dict[str, Any] = {
        "schema": EXP289_ARTIFACT_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "protocol_id": _PROTOCOL_ID,
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "experiment_id": "EXP-289",
        "hypothesis_id": "H-CONFLICT-01",
        "arm_order": list(_ARM_ORDER),
        "primary_endpoint": deepcopy(_PRIMARY_ENDPOINT),
        "protected_endpoints": deepcopy(_PROTECTED_ENDPOINTS),
        "multiplicity_family": _MULTIPLICITY_FAMILY,
        "scope_policy": deepcopy(_SCOPE_POLICY),
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_seed_materialized": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "learned_clause_generalization_validated": False,
        "root_seed": root_seed,
        "model_init_rng_stream": "model_init",
        "model_init_seed": int(model_seed),
        "initial_state": {
            "no_nogood_digest": no_initial_digest,
            "local_nogood_digest": local_initial_digest,
            "functional_digest_match": no_initial_digest == local_initial_digest,
        },
        "resource_match": {
            "pair_audit": pair_audit,
            "pair_audit_digest": _json_digest(pair_audit),
            "parameter_match": bool(pair_audit["parameter_match"]),
            "functional_parameter_match": bool(
                pair_audit["functional_parameter_match"]
            ),
            "active_functional_parameter_match": bool(
                pair_audit["active_functional_parameter_match"]
            ),
            "optimizer_visible_parameter_match": bool(
                pair_audit["optimizer_visible_parameter_match"]
            ),
            "same_initial_world_lineage": True,
            "memory_scope_closed": bool(pair_audit["memory_scope_closed"]),
            "compute_budget_closed": bool(pair_audit["compute_budget_closed"]),
            "declared_max_accounted_cost_per_episode": int(
                pair_audit["declared_max_accounted_cost_per_episode"]
            ),
            "hardware_profiler_flops_claimed": False,
        },
        "training": training,
        "evaluation": evaluation,
        "analysis_method_boundary": _ANALYSIS_BOUNDARY,
        "configuration": configuration,
        "remaining_blockers": [
            "confirmatory data remain unopened",
            "challenge seed remains unmaterialized",
            "frozen paired bootstrap decision rule remains unexecuted",
        ],
    }
    artifact["artifact_digest"] = _artifact_digest(artifact)
    if cache_result:
        key = _configuration_key(configuration, protocol_digest, code_digest)
        _EXPECTED_ARTIFACT_CACHE[key] = deepcopy(artifact)
    return artifact


def run_exp289_paired_development(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    train_replicates: int,
    eval_replicates: int,
    eval_start_replicate: int,
    batch_size: int,
    timesteps: int,
    restarts: int,
    variables: int,
    decoys: int,
    max_search_steps: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    """Run the paired EXP-289 DEVELOPMENT lane without scientific promotion."""

    return _build_artifact(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        train_replicates=train_replicates,
        eval_replicates=eval_replicates,
        eval_start_replicate=eval_start_replicate,
        batch_size=batch_size,
        timesteps=timesteps,
        restarts=restarts,
        variables=variables,
        decoys=decoys,
        max_search_steps=max_search_steps,
        noise_std=noise_std,
        lr=lr,
        weight_decay=weight_decay,
        protocol_digest=protocol_digest,
        code_digest=code_digest,
        cache_result=True,
    )


def _direct_semantic_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_constants = {
        "schema": EXP289_ARTIFACT_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "protocol_id": _PROTOCOL_ID,
        "experiment_id": "EXP-289",
        "arm_order": list(_ARM_ORDER),
        "primary_endpoint": _PRIMARY_ENDPOINT,
        "protected_endpoints": _PROTECTED_ENDPOINTS,
        "multiplicity_family": _MULTIPLICITY_FAMILY,
        "scope_policy": _SCOPE_POLICY,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_seed_materialized": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
    }
    for key, expected in expected_constants.items():
        if artifact.get(key) != expected:
            errors.append(f"EXP-289 semantic contract drift: {key}")

    if artifact.get("artifact_digest") != _artifact_digest(artifact):
        errors.append("EXP-289 artifact self-hash mismatch")

    resource = artifact.get("resource_match")
    if not isinstance(resource, dict):
        errors.append("EXP-289 resource_match missing")
    else:
        for key in (
            "parameter_match",
            "functional_parameter_match",
            "active_functional_parameter_match",
            "optimizer_visible_parameter_match",
            "same_initial_world_lineage",
            "memory_scope_closed",
            "compute_budget_closed",
        ):
            if resource.get(key) is not True:
                errors.append(f"EXP-289 resource court failed: {key}")
        if resource.get("hardware_profiler_flops_claimed") is not False:
            errors.append("EXP-289 hardware-profiler FLOP claim is forbidden")

    configuration = artifact.get("configuration")
    if not isinstance(configuration, dict):
        errors.append("EXP-289 configuration missing")
        return errors
    try:
        train_replicates = int(configuration["train_replicates"])
        eval_start = int(configuration["eval_start_replicate"])
        eval_replicates = int(configuration["eval_replicates"])
        train_set = set(range(train_replicates))
        eval_set = set(range(eval_start, eval_start + eval_replicates))
        if train_set & eval_set:
            errors.append("EXP-289 training/evaluation lineage is not disjoint")
        evaluation = artifact["evaluation"]
        expected_replicates = list(range(eval_start, eval_start + eval_replicates))
        actual_replicates = [
            int(row["replicate"]) for row in evaluation["per_replicate"]
        ]
        if actual_replicates != expected_replicates:
            errors.append("EXP-289 evaluation replicate ordering drift")
        if int(artifact["training"]["replicates"]) != train_replicates:
            errors.append("EXP-289 training replicate receipt drift")
    except (KeyError, TypeError, ValueError):
        errors.append("EXP-289 lineage receipts are malformed")
        return errors

    try:
        ceiling = int(resource["declared_max_accounted_cost_per_episode"])
        for row in artifact["evaluation"]["per_replicate"]:
            world_by_index = {
                int(item["episode_index"]): item
                for item in row["world_receipts"]
            }
            if row.get("evaluator_metadata_delivered_to_arm") is not False:
                errors.append("EXP-289 evaluator metadata leakage")
            no_arm = row["no_nogood"]
            local_arm = row["local_nogood"]
            if no_arm["predeclared_repeat_opportunities"] != local_arm[
                "predeclared_repeat_opportunities"
            ]:
                errors.append("EXP-289 arm-controlled RDER denominator drift")
            for arm_name, arm in (("no_nogood", no_arm), ("local_nogood", local_arm)):
                denominator = int(arm["predeclared_repeat_opportunities"])
                numerator = int(arm["repeated_dead_end_reentries"])
                expected_rate = float(numerator / denominator) if denominator else None
                if arm["repeat_dead_end_rate"] != expected_rate:
                    errors.append(f"EXP-289 {arm_name} RDER arithmetic drift")
                if float(arm["accounted_reasoning_cost"]) > ceiling:
                    errors.append(f"EXP-289 {arm_name} cost exceeds common ceiling")
                for episode in arm["episodes"]:
                    ep_denominator = int(episode["predeclared_repeat_opportunities"])
                    ep_numerator = int(episode["repeated_dead_end_reentries"])
                    expected_ep_rate = (
                        float(ep_numerator / ep_denominator)
                        if ep_denominator
                        else None
                    )
                    if episode["repeat_dead_end_rate"] != expected_ep_rate:
                        errors.append(f"EXP-289 {arm_name} episode RDER drift")
                    cost = episode["cost_receipt"]
                    raw_cost = int(cost["neural_accounted_flops"]) + int(
                        cost["memory_accounted_operations"]
                    )
                    if raw_cost != int(cost["raw_accounted_reasoning_cost"]):
                        errors.append(f"EXP-289 {arm_name} raw cost receipt drift")
                    expected_accounted = (
                        ceiling
                        if bool(episode["censored_at_max_cost"])
                        or raw_cost > ceiling
                        else raw_cost
                    )
                    if int(episode["accounted_reasoning_cost"]) != expected_accounted:
                        errors.append(f"EXP-289 {arm_name} episode cost drift")
                    if arm_name == "local_nogood":
                        world = world_by_index[int(episode["episode_index"])]
                        valid_solutions = world["valid_solutions"]
                        for insertion in episode["store_insertion_receipts"]:
                            if not insertion.get("canonical_key"):
                                errors.append("EXP-289 empty/root nogood insertion")
                                continue
                            recomputed_truth = _has_valid_completion(
                                insertion["canonical_key"],
                                valid_solutions,
                            )
                            if bool(insertion["posthoc_has_valid_completion"]) != bool(
                                recomputed_truth
                            ):
                                errors.append(
                                    "EXP-289 post-hoc nogood safety reconstruction drift"
                                )
                            if insertion.get("ground_truth_safety_gate_used") is not False:
                                errors.append("EXP-289 ground truth illegally gated insertion")
                            if insertion.get("evaluator_result_delivered_to_arm") is not False:
                                errors.append("EXP-289 evaluator result leaked to arm")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("EXP-289 raw evaluation receipts are malformed")

    return errors


def validate_exp289_paired_development(artifact: dict[str, Any]) -> list[str]:
    """Return fail-closed semantic validation errors for an EXP-289 artifact.

    The court checks frozen semantics directly and then compares against a
    deterministic reconstruction from the recorded DEVELOPMENT configuration.
    A recomputed self-hash therefore cannot legalize semantic drift.
    """

    if not isinstance(artifact, dict):
        return ["EXP-289 artifact must be a dictionary"]
    errors = _direct_semantic_errors(artifact)
    configuration = artifact.get("configuration")
    protocol_digest = artifact.get("protocol_digest")
    code_digest = artifact.get("code_digest")
    if not isinstance(configuration, dict) or not isinstance(protocol_digest, str) or not isinstance(code_digest, str):
        if "EXP-289 reconstruction inputs missing" not in errors:
            errors.append("EXP-289 reconstruction inputs missing")
        return errors

    key = _configuration_key(configuration, protocol_digest, code_digest)
    expected = _EXPECTED_ARTIFACT_CACHE.get(key)
    if expected is None:
        try:
            expected = _build_artifact(
                **configuration,
                protocol_digest=protocol_digest,
                code_digest=code_digest,
                cache_result=False,
            )
            _EXPECTED_ARTIFACT_CACHE[key] = deepcopy(expected)
        except Exception as exc:  # fail closed on malformed reconstruction inputs
            errors.append(f"EXP-289 deterministic reconstruction failed: {type(exc).__name__}")
            return errors

    if _artifact_digest(expected) != _artifact_digest(artifact):
        errors.append("EXP-289 deterministic semantic reconstruction mismatch")
    return errors


def validate_exp289_execution_artifact(
    artifact: dict[str, Any],
    protocol: dict[str, Any] | None = None,
) -> None:
    errors = validate_exp289_paired_development(artifact)
    if protocol is not None:
        if protocol.get("protocol_id") not in {None, artifact.get("protocol_id")}:
            errors.append("EXP-289 supplied protocol id mismatch")
    if errors:
        raise ValueError("; ".join(errors))
    return None
