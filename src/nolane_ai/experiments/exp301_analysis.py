from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Iterable

from .exp301_evaluation import EXP301_ARMS, PRIMARY_TRAINED_EFFORTS, RESIDENT_PARAMETERS, Exp301EvaluationRow, evaluate_protected_floors

BOOTSTRAP_SEED = 301_170_017
ROOTS = (0, 1, 2, 3)
REASONING = ("iterative-grid-and-maze", "algorithmic-sequence-transform", "generator-heldout-abstract-transformation")
LANGUAGE = "language-sequence-control"
FAMILIES = (*REASONING, LANGUAGE)


@dataclass(frozen=True, slots=True)
class RCGResult:
    aggregate_rcg: float
    fixed_normalized_auc: float
    nrs_normalized_auc: float
    valid_efforts: tuple[int, ...]
    axis: str = "log2(accounted_flops)"
    normalized: bool = True


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    samples: int
    seed: int


@dataclass(frozen=True, slots=True)
class Exp301AnalysisSummary:
    decision: str
    aggregate_rcg: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    positive_roots: int
    reasoning_families_ge_5pp: int
    language_regression: float
    protected_floors_all_clear: bool
    infrastructure_valid: bool
    valid_primary_efforts: tuple[int, ...]
    bootstrap_seed: int
    exp302_implementation_authorized: bool
    scale_authorized: bool


def _primary(rows):
    return [r for r in rows if r.effort_multiplier in PRIMARY_TRAINED_EFFORTS]


def _mean(rows):
    if not rows:
        raise ValueError("empty cell")
    return sum(float(r.verified_success) for r in rows) / len(rows)


def _balanced(rows, arm, effort):
    values = []
    for family in FAMILIES:
        cell = [r for r in rows if r.arm_id == arm and r.family == family and r.effort_multiplier == effort]
        if not cell:
            raise ValueError("missing family cell")
        values.append(_mean(cell))
    return sum(values) / len(values)


def _axis(rows, effort):
    values = []
    for arm in ("A_FIXED", "C_NRS_CORE"):
        flops = [float(r.accounted_flops) for r in rows if r.arm_id == arm and r.effort_multiplier == effort]
        if not flops or any(v <= 0 or not math.isfinite(v) for v in flops):
            raise ValueError("invalid FLOPs")
        values.append(math.log2(sum(flops) / len(flops)))
    return sum(values) / 2


def _auc(xs, ys):
    if len(xs) < 2 or any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("invalid AUC axis")
    area = sum((b - a) * (y0 + y1) / 2 for a, b, y0, y1 in zip(xs, xs[1:], ys, ys[1:]))
    return area / (xs[-1] - xs[0])


def compute_primary_rcg(rows: Iterable[Exp301EvaluationRow]) -> RCGResult:
    rows = _primary(list(rows))
    efforts = []
    for effort in PRIMARY_TRAINED_EFFORTS:
        cell = [r for r in rows if r.effort_multiplier == effort]
        if not cell or any(r.compute_match_status != "VALID_COMPUTE_MATCH" for r in cell):
            continue
        try:
            _balanced(cell, "A_FIXED", effort); _balanced(cell, "C_NRS_CORE", effort)
        except ValueError:
            continue
        efforts.append(effort)
    if len(efforts) < 2:
        raise ValueError("fewer than two common budgets")
    xs = [_axis(rows, e) for e in efforts]
    a = [_balanced(rows, "A_FIXED", e) for e in efforts]
    c = [_balanced(rows, "C_NRS_CORE", e) for e in efforts]
    aa, ca = _auc(xs, a), _auc(xs, c)
    return RCGResult(ca - aa, aa, ca, tuple(efforts))


def _pair_cells(rows):
    out = {}
    for root in ROOTS:
        for family in FAMILIES:
            for effort in PRIMARY_TRAINED_EFFORTS:
                a = {r.content_id: r for r in rows if r.root == root and r.family == family and r.effort_multiplier == effort and r.arm_id == "A_FIXED"}
                c = {r.content_id: r for r in rows if r.root == root and r.family == family and r.effort_multiplier == effort and r.arm_id == "C_NRS_CORE"}
                ids = sorted(set(a) & set(c))
                if not ids:
                    raise ValueError("missing paired cell")
                out[(root, family, effort)] = [(a[i], c[i]) for i in ids]
    return out


def _pct(values, q):
    values = sorted(values); p = (len(values) - 1) * q; lo = math.floor(p); hi = math.ceil(p)
    if lo == hi:
        return values[lo]
    w = p - lo
    return values[lo] * (1 - w) + values[hi] * w


def paired_hierarchical_bootstrap(rows: Iterable[Exp301EvaluationRow], *, samples: int, seed: int = BOOTSTRAP_SEED) -> BootstrapResult:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rows = list(rows); estimate = compute_primary_rcg(rows).aggregate_rcg; pairs = _pair_cells(_primary(rows)); rng = random.Random(seed); draws = []
    for sample in range(samples):
        boot = []
        for new_root in ROOTS:
            source_root = rng.choice(ROOTS)
            for family in FAMILIES:
                for effort in PRIMARY_TRAINED_EFFORTS:
                    cell = pairs[(source_root, family, effort)]
                    for index in range(len(cell)):
                        a, c = rng.choice(cell); cid = f"b:{sample}:{new_root}:{family}:{effort}:{index}"
                        boot += [replace(a, root=new_root, content_id=cid), replace(c, root=new_root, content_id=cid)]
        draws.append(compute_primary_rcg(boot).aggregate_rcg)
    return BootstrapResult(estimate, _pct(draws, .025), _pct(draws, .975), samples, seed)


def _valid(rows):
    rows = _primary(rows)
    if not rows or {r.root for r in rows} != set(ROOTS) or {r.arm_id for r in rows} != set(EXP301_ARMS) or {r.family for r in rows} != set(FAMILIES):
        return False
    if any(r.resident_parameters != RESIDENT_PARAMETERS or r.compute_match_status != "VALID_COMPUTE_MATCH" or r.accounted_flops <= 0 for r in rows):
        return False
    for root in ROOTS:
        for family in FAMILIES:
            for effort in PRIMARY_TRAINED_EFFORTS:
                sets = []
                for arm in EXP301_ARMS:
                    ids = [r.content_id for r in rows if r.root == root and r.family == family and r.effort_multiplier == effort and r.arm_id == arm]
                    if not ids or len(ids) != len(set(ids)):
                        return False
                    sets.append(set(ids))
                if not (sets[0] == sets[1] == sets[2]):
                    return False
    return True


def _family_gains(rows):
    rows = _primary(rows); total = 0
    for family in REASONING:
        gains = []
        for effort in PRIMARY_TRAINED_EFFORTS:
            a = [r for r in rows if r.arm_id == "A_FIXED" and r.family == family and r.effort_multiplier == effort]
            c = [r for r in rows if r.arm_id == "C_NRS_CORE" and r.family == family and r.effort_multiplier == effort]
            gains.append(_mean(c) - _mean(a))
        total += max(gains) >= .05
    return total


def _language_regression(rows):
    rows = _primary(rows); a = [r for r in rows if r.arm_id == "A_FIXED" and r.family == LANGUAGE]; c = [r for r in rows if r.arm_id == "C_NRS_CORE" and r.family == LANGUAGE]
    return _mean(a) - _mean(c)


def decision_from_summary(s: Exp301AnalysisSummary) -> str:
    if not s.infrastructure_valid:
        return "INVALID_COURT"
    if not s.valid_primary_efforts:
        return "RECURRENCE_STABILITY_DIAGNOSTIC_ONLY"
    passed = s.protected_floors_all_clear and s.aggregate_rcg >= .05 and s.reasoning_families_ge_5pp >= 2 and s.language_regression <= .02 and s.positive_roots >= 3 and s.bootstrap_ci_low > 0
    return "PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY" if passed else "KILL_H_RD_01"


def reduce_exp301(rows: Iterable[Exp301EvaluationRow], *, bootstrap_samples: int, bootstrap_seed: int = BOOTSTRAP_SEED) -> Exp301AnalysisSummary:
    rows = list(rows)
    if not _valid(rows):
        return Exp301AnalysisSummary("INVALID_COURT", 0, 0, 0, 0, 0, math.inf, False, False, (), bootstrap_seed, False, False)
    rcg = compute_primary_rcg(rows); boot = paired_hierarchical_bootstrap(rows, samples=bootstrap_samples, seed=bootstrap_seed); floors = evaluate_protected_floors(_primary(rows))
    positive = sum(compute_primary_rcg([r for r in rows if r.root == root]).aggregate_rcg > 0 for root in ROOTS)
    summary = Exp301AnalysisSummary("UNDECIDED", rcg.aggregate_rcg, boot.ci_low, boot.ci_high, positive, _family_gains(rows), _language_regression(rows), bool(floors["all_clear"]), True, rcg.valid_efforts, bootstrap_seed, False, False)
    return replace(summary, decision=decision_from_summary(summary))
