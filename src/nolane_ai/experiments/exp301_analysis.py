from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

from nolane_ai.experiments.exp301_evaluation import (
    EXP301_ARMS,
    EXP301_EFFORTS,
    PRIMARY_TRAINED_EFFORTS,
    Exp301EvaluationRow,
    evaluate_protected_floors,
)


DECISION_PROMOTE = "PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY"
DECISION_KILL = "KILL_H_RD_01"
DECISION_STABILITY = "RECURRENCE_STABILITY_DIAGNOSTIC_ONLY"
DECISION_INVALID = "INVALID_COURT"

REASONING_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
)
LANGUAGE_FAMILY = "language-sequence-control"
ALL_FAMILIES = (*REASONING_FAMILIES, LANGUAGE_FAMILY)
EXPECTED_ROOTS = (0, 1, 2, 3)

RCG_MESI = 0.05
FAMILY_GAIN_MESI = 0.05
MIN_QUALIFYING_REASONING_FAMILIES = 2
MIN_POSITIVE_ROOTS = 3


@dataclass(frozen=True, slots=True)
class Exp301AnalysisResult:
    decision: str
    aggregate_rcg: float
    rcg_ci_low: float
    rcg_ci_high: float
    secondary_simple_recurrent_rcg: float
    qualifying_reasoning_family_count: int
    qualifying_effort: int | None
    positive_root_count: int
    protected_floors: dict[str, bool]
    valid_efforts: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class _PairUnit:
    root: int
    family: str
    effort: int
    content_id: str
    candidate_success: float
    rival_success: float
    candidate_flops: int
    rival_flops: int


def _row_key(row: Exp301EvaluationRow) -> tuple[int, str, int, str]:
    return (row.root, row.family, row.effort_multiplier, row.content_id)


def _pair_units(
    rows: Iterable[Exp301EvaluationRow],
    *,
    candidate_arm: str,
    rival_arm: str,
    require_unique: bool = True,
) -> list[_PairUnit]:
    if candidate_arm == rival_arm:
        raise ValueError("candidate and rival arms must differ")
    if candidate_arm not in EXP301_ARMS or rival_arm not in EXP301_ARMS:
        raise ValueError("unknown EXP-301 arm")

    by_arm: dict[str, dict[tuple[int, str, int, str], Exp301EvaluationRow]] = {
        candidate_arm: {},
        rival_arm: {},
    }
    for row in rows:
        if row.arm_id not in by_arm:
            continue
        key = _row_key(row)
        if require_unique and key in by_arm[row.arm_id]:
            raise ValueError(f"duplicate paired observation for {row.arm_id}: {key}")
        by_arm[row.arm_id][key] = row

    candidate_keys = set(by_arm[candidate_arm])
    rival_keys = set(by_arm[rival_arm])
    if candidate_keys != rival_keys or not candidate_keys:
        raise ValueError("candidate/rival rows do not form a complete paired ledger")

    units: list[_PairUnit] = []
    for key in sorted(candidate_keys):
        candidate = by_arm[candidate_arm][key]
        rival = by_arm[rival_arm][key]
        units.append(
            _PairUnit(
                root=candidate.root,
                family=candidate.family,
                effort=candidate.effort_multiplier,
                content_id=candidate.content_id,
                candidate_success=float(candidate.verified_success),
                rival_success=float(rival.verified_success),
                candidate_flops=candidate.accounted_flops,
                rival_flops=rival.accounted_flops,
            )
        )
    return units


def _normalized_rcg_from_units(units: Sequence[_PairUnit]) -> float:
    if not units:
        raise ValueError("RCG requires paired observations")

    by_effort: dict[int, list[_PairUnit]] = {}
    for unit in units:
        by_effort.setdefault(unit.effort, []).append(unit)

    points: list[tuple[float, float]] = []
    for effort in sorted(by_effort):
        group = by_effort[effort]
        if not group:
            continue
        gap = sum(unit.candidate_success - unit.rival_success for unit in group) / len(group)
        # One x coordinate per common operating point. Geometric-mean FLOPs
        # treats a small symmetric ledger mismatch without privileging an arm.
        log_flops = sum(
            0.5 * (math.log2(unit.candidate_flops) + math.log2(unit.rival_flops))
            for unit in group
        ) / len(group)
        points.append((log_flops, gap))

    if not points:
        raise ValueError("RCG has no valid operating points")
    if len(points) == 1:
        return points[0][1]

    points.sort(key=lambda point: point[0])
    span = points[-1][0] - points[0][0]
    if span <= 0.0:
        raise ValueError("RCG compute span must be positive")

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) * 0.5
    return area / span


def normalized_rcg(
    rows: Iterable[Exp301EvaluationRow],
    *,
    candidate_arm: str = "C_NRS_CORE",
    rival_arm: str = "A_FIXED",
) -> float:
    return _normalized_rcg_from_units(
        _pair_units(rows, candidate_arm=candidate_arm, rival_arm=rival_arm)
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_hierarchical_bootstrap_rcg(
    rows: Iterable[Exp301EvaluationRow],
    *,
    samples: int = 2_000,
    seed: int = 301,
    candidate_arm: str = "C_NRS_CORE",
    rival_arm: str = "A_FIXED",
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    units = _pair_units(rows, candidate_arm=candidate_arm, rival_arm=rival_arm)
    by_root_cell: dict[int, dict[tuple[str, int], list[_PairUnit]]] = {}
    for unit in units:
        by_root_cell.setdefault(unit.root, {}).setdefault((unit.family, unit.effort), []).append(unit)

    roots = sorted(by_root_cell)
    if not roots:
        raise ValueError("bootstrap requires roots")
    rng = random.Random(seed)
    estimates: list[float] = []

    for _ in range(samples):
        sampled_units: list[_PairUnit] = []
        # Root is the replication stratum. Sample roots with replacement, then
        # paired held-out instances within each family/effort cell.
        sampled_roots = [rng.choice(roots) for _ in roots]
        for synthetic_root, source_root in enumerate(sampled_roots):
            cells = by_root_cell[source_root]
            for (family, effort), cell in sorted(cells.items()):
                if not cell:
                    continue
                for sample_index in range(len(cell)):
                    chosen = rng.choice(cell)
                    sampled_units.append(
                        _PairUnit(
                            root=synthetic_root,
                            family=family,
                            effort=effort,
                            content_id=f"bootstrap-{synthetic_root}-{family}-{effort}-{sample_index}",
                            candidate_success=chosen.candidate_success,
                            rival_success=chosen.rival_success,
                            candidate_flops=chosen.candidate_flops,
                            rival_flops=chosen.rival_flops,
                        )
                    )
        estimates.append(_normalized_rcg_from_units(sampled_units))

    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _validate_complete_scientific_ledger(rows: Sequence[Exp301EvaluationRow]) -> str | None:
    if not rows:
        return "scientific ledger is empty"

    if {row.arm_id for row in rows} != set(EXP301_ARMS):
        return "scientific ledger does not contain all three registered arms"
    if {row.root for row in rows} != set(EXPECTED_ROOTS):
        return "scientific ledger does not contain exactly roots 0,1,2,3"
    if {row.family for row in rows} != set(ALL_FAMILIES):
        return "scientific ledger does not contain all registered task families"
    if {row.effort_multiplier for row in rows} != set(EXP301_EFFORTS):
        return "scientific ledger does not contain all registered effort points"

    key_sets: dict[str, set[tuple[int, str, int, str]]] = {}
    for arm in EXP301_ARMS:
        arm_rows = [row for row in rows if row.arm_id == arm]
        keys = [_row_key(row) for row in arm_rows]
        if len(keys) != len(set(keys)):
            return f"scientific ledger contains duplicate pair keys for {arm}"
        key_sets[arm] = set(keys)
    first = key_sets[EXP301_ARMS[0]]
    if any(key_sets[arm] != first for arm in EXP301_ARMS[1:]):
        return "scientific ledger does not form complete three-arm pairs"
    return None


def _reasoning_family_gain(rows: Sequence[Exp301EvaluationRow]) -> tuple[int, int | None]:
    best_count = 0
    best_effort: int | None = None
    for effort in PRIMARY_TRAINED_EFFORTS:
        qualifying = 0
        for family in REASONING_FAMILIES:
            subset = [
                row for row in rows
                if row.family == family and row.effort_multiplier == effort
            ]
            units = _pair_units(
                subset,
                candidate_arm="C_NRS_CORE",
                rival_arm="A_FIXED",
            )
            gain = sum(unit.candidate_success - unit.rival_success for unit in units) / len(units)
            if gain >= FAMILY_GAIN_MESI - 1e-12:
                qualifying += 1
        if qualifying > best_count:
            best_count = qualifying
            best_effort = effort
    return best_count, best_effort


def _positive_root_count(rows: Sequence[Exp301EvaluationRow]) -> int:
    positive = 0
    for root in EXPECTED_ROOTS:
        subset = [row for row in rows if row.root == root]
        if normalized_rcg(subset, candidate_arm="C_NRS_CORE", rival_arm="A_FIXED") > 0.0:
            positive += 1
    return positive


def reduce_exp301(
    rows: Iterable[Exp301EvaluationRow],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 301,
    challenge_leakage_clear: bool,
    hidden_scaffold_clear: bool,
    representation_stability_ok: bool,
) -> Exp301AnalysisResult:
    materialized = list(rows)
    floors = evaluate_protected_floors(materialized)
    floors = {
        **floors,
        "challenge_leakage": bool(challenge_leakage_clear),
        "hidden_scaffold": bool(hidden_scaffold_clear),
    }
    floors["all_clear"] = all(
        value for key, value in floors.items() if key != "all_clear"
    )

    completeness_error = _validate_complete_scientific_ledger(materialized)
    infrastructure_valid = (
        completeness_error is None
        and floors["resident_parameter_identity"]
        and floors["compute_match"]
        and floors["challenge_leakage"]
        and floors["hidden_scaffold"]
    )

    if not infrastructure_valid:
        reason = completeness_error or "scientific court failed an infrastructure/evidence-boundary floor"
        return Exp301AnalysisResult(
            decision=DECISION_INVALID,
            aggregate_rcg=float("nan"),
            rcg_ci_low=float("nan"),
            rcg_ci_high=float("nan"),
            secondary_simple_recurrent_rcg=float("nan"),
            qualifying_reasoning_family_count=0,
            qualifying_effort=None,
            positive_root_count=0,
            protected_floors=floors,
            valid_efforts=tuple(),
            reason=reason,
        )

    aggregate_rcg = normalized_rcg(
        materialized,
        candidate_arm="C_NRS_CORE",
        rival_arm="A_FIXED",
    )
    simple_rcg = normalized_rcg(
        materialized,
        candidate_arm="C_NRS_CORE",
        rival_arm="B_LOOP_SIMPLE",
    )
    ci_low, ci_high = paired_hierarchical_bootstrap_rcg(
        materialized,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        candidate_arm="C_NRS_CORE",
        rival_arm="A_FIXED",
    )
    family_count, qualifying_effort = _reasoning_family_gain(materialized)
    positive_roots = _positive_root_count(materialized)

    scientific_performance_floors = floors["language_control"] and floors["invalid_output"]
    promote = (
        scientific_performance_floors
        and aggregate_rcg >= RCG_MESI - 1e-12
        and family_count >= MIN_QUALIFYING_REASONING_FAMILIES
        and positive_roots >= MIN_POSITIVE_ROOTS
        and ci_low > 0.0
    )

    if promote:
        decision = DECISION_PROMOTE
        reason = "all preregistered EXP-301 promotion conjuncts passed"
    elif (
        not representation_stability_ok
        and aggregate_rcg > 0.0
        and family_count < MIN_QUALIFYING_REASONING_FAMILIES
    ):
        decision = DECISION_STABILITY
        reason = "positive aggregate recurrence signal lacked trained-depth family qualification and representation stability"
    else:
        decision = DECISION_KILL
        reason = "valid EXP-301 court did not clear the full preregistered promotion conjunction"

    return Exp301AnalysisResult(
        decision=decision,
        aggregate_rcg=aggregate_rcg,
        rcg_ci_low=ci_low,
        rcg_ci_high=ci_high,
        secondary_simple_recurrent_rcg=simple_rcg,
        qualifying_reasoning_family_count=family_count,
        qualifying_effort=qualifying_effort,
        positive_root_count=positive_roots,
        protected_floors=floors,
        valid_efforts=tuple(EXP301_EFFORTS),
        reason=reason,
    )
