import hashlib, json
from pathlib import Path

from nolane_ai.experiments.exp331_contract import MODES, PAIR_IDS, PairArmResult
from nolane_ai.experiments.exp332_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    metric_vector_equal,
    reduce_portable_pair_lattice,
)

def _arm(pair: str, mode: str, passed: bool = True, *, model: str = "a", optimizer: str = "b", rng: str = "c", projected: int = 0) -> PairArmResult:
    token = (1.0, 1.0) if passed else (0.5, 1.0)
    exact = (1.0, 1.0) if passed else (0.0, 1.0)
    negative = max(projected, 4 if mode != "CONTROL" else 0)
    return PairArmResult(pair, mode, 64, token, exact, model * 64, optimizer * 64, rng * 64, 0, negative, projected)

def _lattice(*, p02_project: bool = True, regressions: tuple[str, ...] = ()) -> dict[str, PairArmResult]:
    out = {}
    for pair in PAIR_IDS:
        control_pass = pair != "P02"
        out[f"{pair}:CONTROL"] = _arm(pair, "CONTROL", control_pass)
        out[f"{pair}:SHAM"] = _arm(pair, "SHAM", control_pass)
        project_pass = p02_project if pair == "P02" else pair not in regressions
        out[f"{pair}:PROJECT"] = _arm(pair, "PROJECT", project_pass, model="d", optimizer="e", projected=5 if pair == "P02" else 0)
    return out

def test_metric_vector_equal_normalizes_json_lists_and_runtime_tuples():
    assert metric_vector_equal((1.0, 0.5), [1.0, 0.5])
    assert not metric_vector_equal((1.0, 0.5), [1.0, 0.6])

def test_positive_requires_parent_behavior_witness_and_no_regression():
    decision, _, regressions = reduce_portable_pair_lattice(
        _lattice(),
        parent_behavior_reproduced=True,
        witness_divergence_confirmed=True,
    )
    assert decision == "PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION"
    assert regressions == ()

def test_parent_behavior_gate_and_witness_gate_fail_closed():
    assert reduce_portable_pair_lattice(
        _lattice(),
        parent_behavior_reproduced=False,
        witness_divergence_confirmed=True,
    )[0] == "PARENT_BEHAVIORAL_REPRODUCTION_MISMATCH"
    assert reduce_portable_pair_lattice(
        _lattice(),
        parent_behavior_reproduced=True,
        witness_divergence_confirmed=False,
    )[0] == "INVALID_PORTABLE_PAIR_LATTICE_COURT"

def test_regression_and_no_rescue_are_distinct():
    assert reduce_portable_pair_lattice(
        _lattice(regressions=("P13",)),
        parent_behavior_reproduced=True,
        witness_divergence_confirmed=True,
    )[0] == "P02_RESCUE_WITH_PAIR_REGRESSION"
    assert reduce_portable_pair_lattice(
        _lattice(p02_project=False),
        parent_behavior_reproduced=True,
        witness_divergence_confirmed=True,
    )[0] == "P02_PROJECTION_NO_RESCUE"

def test_preregistration_digest_is_locked():
    root = Path(__file__).resolve().parents[1]
    path = root / "protocols/v017/exp332_preregistration_v1.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == APPROVED_PREREGISTRATION_DIGEST
    expected = (root / "protocols/v017/exp332_preregistration_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest() == expected
