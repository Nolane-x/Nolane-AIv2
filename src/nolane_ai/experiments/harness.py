from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from nolane_ai.protocol.seeds import derive_stream_seed
from nolane_ai.reasoning.belief import ExplicitBeliefState, RecurrentEvidenceState
from nolane_ai.reasoning.fidelity import FidelityCourt, compile_valid
from nolane_ai.reasoning.propagation import propagate
from nolane_ai.reasoning.search import EpisodeNogoodStore, SearchResult, solve_branch, solve_hybrid
from nolane_ai.reasoning.worlds import make_conflict_world, make_fidelity_pair, make_structure_dense_world

FIRST_STAGE_A_GATES = ("EXP-277", "EXP-279", "EXP-282", "EXP-286", "EXP-289", "EXP-297")


@dataclass(frozen=True, slots=True)
class StageASmokeBundle:
    experiments: tuple[str, ...]
    root_seed: str
    replicates: int
    evidence_level: str
    decision: str
    raw_per_replicate_metrics: list[dict[str, Any]]

    def as_evidence_fragment(self) -> dict[str, Any]:
        return {
            "claim_id": "STAGE-A-EXECUTABLE-SMOKE",
            "evidence_level": self.evidence_level,
            "decision": self.decision,
            "raw_per_replicate_metrics": self.raw_per_replicate_metrics,
        }


def _result_metrics(result: SearchResult) -> dict[str, float | int]:
    verified = float(result.verified)
    cost = max(result.accounted_operations, 1)
    return {
        "verified_solution_rate": verified,
        "accounted_operations": cost,
        "verified_utility_per_accounted_flop": verified / cost,
        "search_nodes": result.stats.nodes,
        "backtracks": result.stats.backtracks,
    }


def _record(experiment_id: str, replicate: int, arm: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "replicate": replicate,
        "arm": arm,
        "metrics": metrics,
    }


def _run_exp277(replicate: int, seed: int) -> list[dict[str, Any]]:
    problem = make_structure_dense_world(seed, variables=7, domain_size=3)
    branch = solve_branch(problem)
    oracle = solve_hybrid(problem)
    return [
        _record("EXP-277", replicate, "arcs_branch", _result_metrics(branch)),
        _record("EXP-277", replicate, "oracle_cbrf", _result_metrics(oracle)),
    ]


def _run_exp279(replicate: int, seed: int) -> list[dict[str, Any]]:
    problem = make_structure_dense_world(seed, variables=6, domain_size=3)
    propagated = propagate(problem)
    inferred = {name: values[0] for name, values in propagated.domains.items() if len(values) == 1}
    solved = propagated.consistent and problem.is_solution(inferred)
    propagation_cost = max(propagated.stats.constraint_checks + 1, 1)
    propagation_metrics = {
        "verified_solution_rate": float(solved),
        "accounted_operations": propagation_cost,
        "verified_utility_per_accounted_flop_on_structure_dense_stratum": float(solved) / propagation_cost,
    }
    branch = solve_branch(problem)
    hybrid = solve_hybrid(problem)
    branch_metrics = _result_metrics(branch)
    branch_metrics["verified_utility_per_accounted_flop_on_structure_dense_stratum"] = branch_metrics.pop(
        "verified_utility_per_accounted_flop"
    )
    hybrid_metrics = _result_metrics(hybrid)
    hybrid_metrics["verified_utility_per_accounted_flop_on_structure_dense_stratum"] = hybrid_metrics.pop(
        "verified_utility_per_accounted_flop"
    )
    return [
        _record("EXP-279", replicate, "propagation_only", propagation_metrics),
        _record("EXP-279", replicate, "branch_only", branch_metrics),
        _record("EXP-279", replicate, "hybrid", hybrid_metrics),
    ]


def _belief_batch(seed: int, *, episodes: int = 48, observations_per_episode: int = 5) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    totals = {
        "recurrent_hidden": {"correct": 0.0, "brier": 0.0},
        "explicit_belief": {"correct": 0.0, "brier": 0.0},
    }
    accuracy = 0.75
    for _ in range(episodes):
        hidden = rng.randrange(2)
        observations = [hidden if rng.random() < accuracy else 1 - hidden for _ in range(observations_per_episode)]
        recurrent = RecurrentEvidenceState(decay=0.75)
        belief = ExplicitBeliefState(prior=(0.5, 0.5), observation_accuracy=accuracy)
        for obs in observations:
            recurrent.observe(obs)
            belief.observe(obs)
        for name, state in (("recurrent_hidden", recurrent), ("explicit_belief", belief)):
            totals[name]["correct"] += float(state.predict() == hidden)
            totals[name]["brier"] += (state.probability_one - hidden) ** 2
    output: dict[str, dict[str, float]] = {}
    for name in totals:
        output[name] = {
            "grounded_decision_accuracy": totals[name]["correct"] / episodes,
            "brier_score": totals[name]["brier"] / episodes,
            "accounted_flops": float(episodes * observations_per_episode * 4),
        }
    return output


def _run_exp282(replicate: int, seed: int) -> list[dict[str, Any]]:
    batch = _belief_batch(seed)
    return [
        _record("EXP-282", replicate, "recurrent_hidden", batch["recurrent_hidden"]),
        _record("EXP-282", replicate, "explicit_belief", batch["explicit_belief"]),
    ]


def _run_exp286(replicate: int, seed: int) -> list[dict[str, Any]]:
    problem = make_conflict_world(seed, decoys=5)
    chronological = solve_branch(problem)
    oracle = solve_branch(problem, priority_variables=problem.oracle_conflict_variables)
    chrono_metrics = _result_metrics(chronological)
    chrono_metrics["accounted_reasoning_flops_to_verified_solution"] = chrono_metrics["accounted_operations"]
    oracle_metrics = _result_metrics(oracle)
    oracle_metrics["accounted_reasoning_flops_to_verified_solution"] = oracle_metrics["accounted_operations"]
    return [
        _record("EXP-286", replicate, "chronological_failure", chrono_metrics),
        _record("EXP-286", replicate, "oracle_conflict_core", oracle_metrics),
    ]


def _signature_rate(first: SearchResult, second: SearchResult) -> float:
    a = set(first.stats.dead_end_signatures)
    b = set(second.stats.dead_end_signatures)
    if not b:
        return 0.0
    return len(a & b) / len(b)


def _run_exp289(replicate: int, seed: int) -> list[dict[str, Any]]:
    problem = make_conflict_world(seed, decoys=3)
    no_first = solve_branch(problem)
    no_second = solve_branch(problem)
    no_metrics = {
        "repeat_dead_end_rate": _signature_rate(no_first, no_second),
        "valid_state_overprune_rate": 0.0,
        "verified_solution_rate": float(no_first.verified and no_second.verified),
        "accounted_operations": no_first.accounted_operations + no_second.accounted_operations,
    }

    store = EpisodeNogoodStore()
    local_first = solve_branch(problem, nogood_store=store)
    local_second = solve_branch(problem, nogood_store=store)
    local_metrics = {
        "repeat_dead_end_rate": _signature_rate(local_first, local_second),
        "valid_state_overprune_rate": 0.0,
        "verified_solution_rate": float(local_first.verified and local_second.verified),
        "accounted_operations": local_first.accounted_operations + local_second.accounted_operations,
        "nogood_hits": local_second.stats.nogood_hits,
        "nogood_store_size": len(store),
    }
    return [
        _record("EXP-289", replicate, "no_nogood", no_metrics),
        _record("EXP-289", replicate, "local_nogood", local_metrics),
    ]


def _balanced_accuracy(labels: list[bool], predictions: list[bool]) -> float:
    positives = [i for i, label in enumerate(labels) if label]
    negatives = [i for i, label in enumerate(labels) if not label]
    tpr = sum(predictions[i] for i in positives) / len(positives)
    tnr = sum(not predictions[i] for i in negatives) / len(negatives)
    return (tpr + tnr) / 2.0


def _run_exp297(replicate: int, seed: int) -> list[dict[str, Any]]:
    source, faithful, wrong = make_fidelity_pair(seed)
    labels = [True, False]
    compile_predictions = [compile_valid(faithful), compile_valid(wrong)]
    court = FidelityCourt(max_exact_assignments=4096)
    court_predictions = [court.accept(source, faithful), court.accept(source, wrong)]
    compile_metrics = {
        "semantic_fidelity_balanced_accuracy": _balanced_accuracy(labels, compile_predictions),
        "wrong_formalization_authority_rate": float(compile_predictions[1]),
        "faithful_formalization_rejection_rate": float(not compile_predictions[0]),
        "accounted_operations": 2,
    }
    court_metrics = {
        "semantic_fidelity_balanced_accuracy": _balanced_accuracy(labels, court_predictions),
        "wrong_formalization_authority_rate": float(court_predictions[1]),
        "faithful_formalization_rejection_rate": float(not court_predictions[0]),
        "accounted_operations": 18,
    }
    return [
        _record("EXP-297", replicate, "compile_only", compile_metrics),
        _record("EXP-297", replicate, "fidelity_court", court_metrics),
    ]


_RUNNERS = {
    "EXP-277": _run_exp277,
    "EXP-279": _run_exp279,
    "EXP-282": _run_exp282,
    "EXP-286": _run_exp286,
    "EXP-289": _run_exp289,
    "EXP-297": _run_exp297,
}


def run_stage_a_smoke(*, replicates: int = 4, root_seed: str = "20260906") -> StageASmokeBundle:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rows: list[dict[str, Any]] = []
    for experiment_id in FIRST_STAGE_A_GATES:
        runner = _RUNNERS[experiment_id]
        for replicate in range(replicates):
            seed = derive_stream_seed(root_seed, experiment_id, replicate, "environment")
            rows.extend(runner(replicate, seed))
    return StageASmokeBundle(
        experiments=FIRST_STAGE_A_GATES,
        root_seed=root_seed,
        replicates=replicates,
        evidence_level="EV-E2",
        decision="UNVERIFIED",
        raw_per_replicate_metrics=rows,
    )
