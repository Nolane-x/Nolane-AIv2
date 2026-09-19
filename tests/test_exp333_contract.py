import hashlib, json
from pathlib import Path

from nolane_ai.experiments.exp333_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    MODES,
    PAIR_IDS,
    PAIR_MEMBERS,
    PairArmResult,
    reduce_complete_pair_lattice,
    training_schedule,
)

def _arm(pair, mode, passed=True, *, projected=0, model="a", optimizer="b", rng="c"):
    token = (1.0, 1.0) if passed else (0.5, 1.0)
    exact = (1.0, 1.0) if passed else (0.0, 1.0)
    negative = max(projected, 4 if mode != "CONTROL" else 0)
    return PairArmResult(pair, mode, 64, token, exact, model*64, optimizer*64, rng*64, 0, negative, projected)

def _lattice(*, baseline_failures=("P02",), rescued=None, regressions=(), no_trigger=()):
    rescued = set(baseline_failures if rescued is None else rescued)
    out = {}
    for pair in PAIR_IDS:
        control_pass = pair not in baseline_failures
        out[f"{pair}:CONTROL"] = _arm(pair, "CONTROL", control_pass)
        out[f"{pair}:SHAM"] = _arm(pair, "SHAM", control_pass)
        project_pass = (pair in rescued) if pair in baseline_failures else pair not in regressions
        projected = 0 if pair in no_trigger else (5 if pair in baseline_failures else 1)
        out[f"{pair}:PROJECT"] = _arm(pair, "PROJECT", project_pass, projected=projected, model="d", optimizer="e")
    return out

def test_complete_pair_geometry_is_all_28_pairs_over_worlds_0_to_7():
    assert len(PAIR_IDS) == 28
    assert set(PAIR_MEMBERS.values()) == {(i, j) for i in range(8) for j in range(i+1, 8)}
    assert PAIR_IDS[0] == "P01" and PAIR_IDS[-1] == "P67"

def test_schedule_is_64_updates_and_ascending_within_round():
    schedule = training_schedule("P47")
    assert len(schedule) == 64
    assert schedule[:4] == ((4,0,1),(7,0,1),(4,1,2),(7,1,2))

def test_strongest_positive_requires_full_rescue_and_no_regression():
    d, _, failures, rescued, unresolved, regressions, no_trigger = reduce_complete_pair_lattice(
        _lattice(), parent_anchor_reproduced=True
    )
    assert d == "COMPLETE_8WORLD_PAIR_LATTICE_RESCUE_NO_REGRESSION"
    assert failures == ("P02",) and rescued == ("P02",)
    assert unresolved == regressions == no_trigger == ()

def test_new_baseline_failures_are_handled_without_posthoc_selection():
    d, _, failures, rescued, unresolved, _, _ = reduce_complete_pair_lattice(
        _lattice(baseline_failures=("P02","P04"), rescued=("P02",)),
        parent_anchor_reproduced=True,
    )
    assert d == "BASELINE_PAIR_FAILURES_PARTIALLY_RESCUED"
    assert failures == ("P02","P04")
    assert rescued == ("P02",) and unresolved == ("P04",)

def test_regression_has_precedence_over_positive_claim():
    d, _, _, _, _, regressions, _ = reduce_complete_pair_lattice(
        _lattice(regressions=("P13",)), parent_anchor_reproduced=True
    )
    assert d == "PROJECT_PAIR_REGRESSION_PRESENT"
    assert regressions == ("P13",)

def test_baseline_failure_without_projection_trigger_fails_closed():
    d = reduce_complete_pair_lattice(
        _lattice(baseline_failures=("P02","P04"), no_trigger=("P04",)),
        parent_anchor_reproduced=True,
    )[0]
    assert d == "BASELINE_FAILURE_WITHOUT_PROJECTION_TRIGGER"

def test_no_rescue_is_distinct():
    d = reduce_complete_pair_lattice(
        _lattice(rescued=()), parent_anchor_reproduced=True
    )[0]
    assert d == "BASELINE_PAIR_FAILURES_NOT_RESCUED"

def test_parent_anchor_gate_fails_closed():
    assert reduce_complete_pair_lattice(
        _lattice(), parent_anchor_reproduced=False
    )[0] == "PARENT_PAIR_ANCHOR_REPRODUCTION_MISMATCH"

def test_preregistration_digest_and_raw_checksum_are_locked():
    root = Path(__file__).resolve().parents[1]
    path = root / "protocols/v017/exp333_preregistration_v1.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == APPROVED_PREREGISTRATION_DIGEST
    expected = (root / "protocols/v017/exp333_preregistration_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest() == expected
