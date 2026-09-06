from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from statistics import fmean
from typing import Any

from .harness import StageASmokeBundle


@dataclass(frozen=True, slots=True)
class EffectSummary:
    experiment_id: str
    metric: str
    baseline_arm: str
    candidate_arm: str
    observed_effect: float
    ci_low: float
    ci_high: float
    replicates: int
    effect_type: str


_COMPARISONS: dict[str, tuple[str, str, str, str]] = {
    "EXP-277": ("verified_utility_per_accounted_flop", "arcs_branch", "oracle_cbrf", "relative_gain"),
    "EXP-282": ("grounded_decision_accuracy", "recurrent_hidden", "explicit_belief", "absolute_gain"),
    "EXP-286": ("accounted_reasoning_flops_to_verified_solution", "chronological_failure", "oracle_conflict_core", "relative_reduction"),
    "EXP-289": ("repeat_dead_end_rate", "no_nogood", "local_nogood", "relative_reduction"),
    "EXP-297": ("semantic_fidelity_balanced_accuracy", "compile_only", "fidelity_court", "absolute_gain"),
}


def _rows_for(bundle: StageASmokeBundle, experiment_id: str) -> dict[int, dict[str, dict[str, Any]]]:
    paired: dict[int, dict[str, dict[str, Any]]] = {}
    for row in bundle.raw_per_replicate_metrics:
        if row["experiment_id"] != experiment_id:
            continue
        paired.setdefault(int(row["replicate"]), {})[str(row["arm"])] = row["metrics"]
    return paired


def _effect(base: float, candidate: float, effect_type: str) -> float:
    if effect_type == "absolute_gain":
        return candidate - base
    denominator = max(abs(base), 1e-12)
    if effect_type == "relative_gain":
        return (candidate - base) / denominator
    if effect_type == "relative_reduction":
        return (base - candidate) / denominator
    raise ValueError(effect_type)


def _exp279_effects(bundle: StageASmokeBundle) -> tuple[list[float], str, str, str]:
    paired = _rows_for(bundle, "EXP-279")
    metric = "verified_utility_per_accounted_flop_on_structure_dense_stratum"
    effects: list[float] = []
    for replicate in sorted(paired):
        arms = paired[replicate]
        best_simple = max(float(arms["propagation_only"][metric]), float(arms["branch_only"][metric]))
        candidate = float(arms["hybrid"][metric])
        effects.append(_effect(best_simple, candidate, "relative_gain"))
    return effects, metric, "best_simple", "hybrid"


def _bootstrap_mean_ci(values: list[float], *, samples: int, seed_material: str) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty paired effect vector")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if len(values) == 1:
        return values[0], values[0]
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[max(0, int(0.025 * samples) - 1)]
    hi = means[min(samples - 1, int(0.975 * samples))]
    observed = fmean(values)
    return min(lo, observed), max(hi, observed)


def summarize_bundle(bundle: StageASmokeBundle, *, bootstrap_samples: int = 1000) -> dict[str, EffectSummary]:
    summaries: dict[str, EffectSummary] = {}
    for experiment_id in bundle.experiments:
        if experiment_id == "EXP-279":
            effects, metric, baseline, candidate = _exp279_effects(bundle)
            effect_type = "relative_gain"
        else:
            metric, baseline, candidate, effect_type = _COMPARISONS[experiment_id]
            paired = _rows_for(bundle, experiment_id)
            effects = [
                _effect(
                    float(paired[replicate][baseline][metric]),
                    float(paired[replicate][candidate][metric]),
                    effect_type,
                )
                for replicate in sorted(paired)
            ]
        observed = fmean(effects)
        lo, hi = _bootstrap_mean_ci(
            effects,
            samples=bootstrap_samples,
            seed_material=f"{bundle.root_seed}|{experiment_id}|analysis-v1",
        )
        summaries[experiment_id] = EffectSummary(
            experiment_id=experiment_id,
            metric=metric,
            baseline_arm=baseline,
            candidate_arm=candidate,
            observed_effect=observed,
            ci_low=lo,
            ci_high=hi,
            replicates=len(effects),
            effect_type=effect_type,
        )
    return summaries


def build_smoke_evidence_packet(
    bundle: StageASmokeBundle,
    *,
    protocol_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    if bundle.evidence_level != "EV-E2" or bundle.decision != "UNVERIFIED":
        raise ValueError("smoke bundles are structurally capped at EV-E2/UNVERIFIED")
    return {
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "claim_id": "STAGE-A-EXECUTABLE-SMOKE",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "experiment_ids": list(bundle.experiments),
        "root_seed": bundle.root_seed,
        "replicates": bundle.replicates,
        "raw_per_replicate_metrics": bundle.raw_per_replicate_metrics,
    }
