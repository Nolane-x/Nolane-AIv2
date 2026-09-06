# EXP-282 Confirmatory Analysis Court Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic EXP-282 confirmatory-open analysis court that can emit bounded EV-E3 Stage-A decisions only from lineage-closed raw neural evidence and frozen protected endpoints.

**Architecture:** Add one focused analysis module that validates all upstream artifacts, computes the paired bootstrap statistic, reconstructs the existing analytical compute ledger from frozen execution geometry, applies the frozen Brier/compute guards, and emits a self-hashed result artifact. Add one fail-closed CLI wrapper and CI/model-smoke coverage. The raw confirmatory artifact remains immutable.

**Tech Stack:** Python 3.11/3.13, PyTorch for matched-arm reconstruction, stdlib `hashlib`, `random`, `statistics`, JSON, pytest, existing Nolane protocol/evidence helpers.

**Spec:** `docs/superpowers/specs/2026-09-06-exp282-confirmatory-analysis-court-design.md`

## Global Constraints

- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- Canonical frozen Stage-A V1 SHA-256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- EXP-282 primary endpoint: `grounded_decision_accuracy`, higher is better.
- EXP-282 absolute MESI: `0.03`.
- Familywise alpha: `0.05`; inference tail: one-sided lower bound for promotion.
- Brier guard: `explicit_belief <= recurrent_hidden + 0.02`.
- Relative accounted-FLOP guard: `<= 0.05`.
- Bootstrap resamples: exactly `10_000`.
- Highest result evidence level: `EV-E3`; never EV-E4/EV-E5.
- No actual confirmatory execution or challenge materialization in this PR.
- Keep analytical FLOP semantics explicitly separate from hardware-profiler FLOPs.

---

### Task 1: RED analysis-court contract tests

**Files:**
- Create: `tests/test_exp282_confirmatory_analysis.py`

**Interfaces:**
- Consumes: existing validators for prep, paired development, reconstruction, and raw confirmatory artifacts.
- Produces expected future interface: `build_exp282_confirmatory_analysis(...)`, `validate_exp282_confirmatory_analysis(...)`, `bootstrap_paired_effect(...)` in `nolane_ai.experiments.exp282_confirmatory_analysis`.

- [ ] **Step 1: Write failing import and contract tests**

Create fixtures by copying valid upstream artifact structures through existing builders where practical, then mutate only the confirmatory raw per-replicate metrics for deterministic decision-boundary tests. Add tests asserting:

```python
from nolane_ai.experiments.exp282_confirmatory_analysis import (
    BOOTSTRAP_SAMPLES,
    build_exp282_confirmatory_analysis,
    bootstrap_paired_effect,
    validate_exp282_confirmatory_analysis,
)

assert BOOTSTRAP_SAMPLES == 10_000
```

Cover:

```python
assert result["evidence_level"] == "EV-E3"
assert result["decision"] in {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}
assert result["raw_per_replicate_metrics"] == raw["per_replicate"]
assert result["protected_endpoints"]["brier"]["pass"] is True
assert result["protected_endpoints"]["compute"]["pass"] is True
```

Add explicit synthetic cases:

- every paired accuracy delta `0.05` -> lower bound >= `0.03` -> promote when guards pass;
- mixed deltas around `0.03` -> lower < `0.03` and upper >= `0.03` -> hold;
- every delta `0.00` -> upper < `0.03` -> kill;
- Brier delta `>0.02` with strong accuracy -> hold, not promote;
- invalid/tampered artifact digest -> raises before a result is returned;
- forged protocol digest -> raises with canonical frozen protocol identity error;
- result mutated after build -> validator returns analysis-digest mismatch.

- [ ] **Step 2: Run RED test**

Run:

```bash
pytest -q tests/test_exp282_confirmatory_analysis.py
```

Expected: collection/import failure because `exp282_confirmatory_analysis.py` does not exist.

- [ ] **Step 3: Preserve RED evidence in CI/PR notes**

Do not weaken the test to make it collect. The missing-module failure is the expected TDD baseline.

---

### Task 2: GREEN deterministic statistics and lineage court

**Files:**
- Create: `src/nolane_ai/experiments/exp282_confirmatory_analysis.py`
- Test: `tests/test_exp282_confirmatory_analysis.py`

**Interfaces:**
- Consumes:
  - `require_canonical_stage_a_v1_digest(protocol_digest: str)` from `nolane_ai.protocol.identity`;
  - `validate_exp282_confirmatory_prep(payload)`;
  - `validate_exp282_paired_development(payload)`;
  - `validate_exp282_confirmatory_reconstruction(payload)`;
  - `validate_exp282_confirmatory_open_raw(payload)`;
  - `account_matched_belief_arm_pair(...)` and `build_matched_belief_arm_pair(...)`.
- Produces:

```python
BOOTSTRAP_SAMPLES: int = 10_000
SCHEMA: str = "NLM-EXP-282-CONFIRMATORY-ANALYSIS-V1"

def bootstrap_paired_effect(
    values: list[float],
    *,
    seed_material: str,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
) -> dict[str, object]: ...

def build_exp282_confirmatory_analysis(
    *,
    protocol: dict[str, object],
    protocol_digest: str,
    prep_artifact: dict[str, object],
    paired_execution_artifact: dict[str, object],
    reconstruction_authorization: dict[str, object],
    raw_artifact: dict[str, object],
    analysis_code_digest: str,
) -> dict[str, object]: ...

def validate_exp282_confirmatory_analysis(payload: dict[str, object]) -> list[str]: ...
```

- [ ] **Step 1: Implement frozen-contract constants and helpers**

Add constants for experiment ID, MESI `0.03`, alpha `0.05`, Brier margin `0.02`, FLOP margin `0.05`, bootstrap sample count `10_000`, and the exact schema.

Implement finite-number validation and canonical self-hash:

```python
def _analysis_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("analysis_digest", None)
    return canonical_sha256(clean)
```

- [ ] **Step 2: Implement deterministic paired bootstrap**

Derive the PRNG integer from:

```python
seed_bytes = hashlib.sha256(seed_material.encode("utf-8")).digest()
rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
```

For each of 10,000 resamples, sample `n` paired effects with replacement and store the mean. Sort means. Use deterministic percentile indices:

```python
lower_index = max(0, math.ceil(alpha * samples) - 1)
upper_index = min(samples - 1, math.ceil((1.0 - alpha) * samples) - 1)
```

Return observed mean, median, lower/upper one-sided bounds, sign counts, sample count, alpha, and `seed_digest = sha256(seed_material).hexdigest()`.

- [ ] **Step 3: Implement upstream artifact and lineage validation**

Validate each artifact with its existing validator. Then require exact equality for protocol/prep/paired/reconstruction/raw digests, checkpoint SHA, execution-contract digest, confirmatory `n`, and reserved replicate IDs.

Require:

```python
require_canonical_stage_a_v1_digest(protocol_digest)
raw_artifact["challenge_materialized"] is False
raw_artifact["confirmatory_data_consumed"] is True
raw_artifact["per_replicate"] replicate order == reserved IDs
```

- [ ] **Step 4: Implement Brier sufficient-statistic guard**

Aggregate `brier_sum` and `brier_count` across all rows for each arm and compute:

```python
recurrent_brier = recurrent_sum / recurrent_count
explicit_brier = explicit_sum / explicit_count
delta = explicit_brier - recurrent_brier
passed = delta <= 0.02
```

Record all sums/counts and the exact margin.

- [ ] **Step 5: Reconstruct and apply analytical compute guard**

From `reconstruction_authorization["execution_contract"]`, instantiate matched arms using `arm_geometry`, then call:

```python
ledger = account_matched_belief_arm_pair(
    recurrent,
    explicit,
    timesteps=int(world_geometry["timesteps"]),
    variables=int(world_geometry["variables"]),
)
```

Pass only if:

```python
ledger["primitive_operation_match"] is True
and float(ledger["relative_accounted_flop_difference"]) <= 0.05
and ledger["hardware_profiler_flops_claimed"] is False
```

- [ ] **Step 6: Implement decision logic**

Use this exact order:

```python
if lower_bound >= 0.03 and brier_pass and compute_pass:
    decision = "PROMOTE_TO_NEXT_STAGE"
elif upper_bound < 0.03:
    decision = "KILL_SUBSYSTEM"
else:
    decision = "HOLD_UNSTABLE"
```

Always emit `EV-E3` for a valid analyzed confirmatory-open result. Do not emit practical equivalence because no equivalence margin is frozen.

- [ ] **Step 7: Build the self-hashed artifact**

Include raw rows exactly, frozen-contract summary, statistic summary, protected endpoints, full compute ledger, complete lineage, blockers for EV-E4/EV-E5, and `analysis_digest`.

- [ ] **Step 8: Implement result validator**

Recompute digest, primary effect from raw rows, Brier sufficient statistics, decision rule, and protected guard booleans. Reject EV-E4/EV-E5, missing raw rows, non-finite values, inconsistent decisions, or digest/lineage fields.

- [ ] **Step 9: Run GREEN tests**

Run:

```bash
pytest -q tests/test_exp282_confirmatory_analysis.py
```

Expected: all analysis-court tests pass.

---

### Task 3: RED/GREEN fail-closed analysis CLI

**Files:**
- Create: `scripts/analyze_exp282_confirmatory_open.py`
- Create: `tests/test_exp282_confirmatory_analysis_cli.py`

**Interfaces:**
- Consumes: `build_exp282_confirmatory_analysis`, upstream validators, `file_sha256`, `require_canonical_stage_a_v1_digest`, `source_tree_digest`.
- Produces CLI JSON file and compact stdout summary.

- [ ] **Step 1: Write failing CLI tests**

Exercise a valid generated fixture chain and assert the command:

```bash
python scripts/analyze_exp282_confirmatory_open.py \
  --protocol protocols/stage_a_v1.json \
  --protocol-digest-file protocols/stage_a_v1.sha256 \
  --prep <prep.json> \
  --execution <paired.json> \
  --reconstruction <reconstruction.json> \
  --raw <raw.json> \
  --output <analysis.json>
```

returns zero, writes a self-validating result, refuses an existing output, and rejects a forged protocol plus matching forged digest file with a canonical frozen protocol identity error.

- [ ] **Step 2: Run RED CLI tests**

Run:

```bash
pytest -q tests/test_exp282_confirmatory_analysis_cli.py
```

Expected: failure because the CLI script does not exist.

- [ ] **Step 3: Implement CLI**

Parse explicit input paths. Before loading scientific artifacts:

```python
load_and_validate_protocol(args.protocol)
actual = file_sha256(args.protocol)
expected = args.protocol_digest_file.read_text(encoding="utf-8").strip()
if actual != expected:
    raise RuntimeError(...)
require_canonical_stage_a_v1_digest(actual)
```

Refuse overwrite before analysis. Load JSON inputs, validate, build result with `source_tree_digest(ROOT)`, write immutable JSON, and print schema/evidence/decision/lower/upper/Brier guard/compute guard/digest.

- [ ] **Step 4: Run GREEN CLI tests**

Run:

```bash
pytest -q tests/test_exp282_confirmatory_analysis_cli.py tests/test_exp282_confirmatory_analysis.py
```

Expected: pass.

---

### Task 4: CI integration and version alignment

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`

**Interfaces:**
- Consumes: new analysis tests and CLI.
- Produces: package version `0.9.0` and model-smoke coverage.

- [ ] **Step 1: Add analysis tests to model-smoke pytest command**

Append:

```text
tests/test_exp282_confirmatory_analysis.py
tests/test_exp282_confirmatory_analysis_cli.py
```

- [ ] **Step 2: Add a CI smoke command that does not consume real confirmatory data**

Do not execute a real confirmatory-open lane in CI. CI only runs the unit/CLI synthetic fixture tests. Preserve existing paired/prep commands unchanged.

- [ ] **Step 3: Bump version consistently**

Set package/project version to `0.9.0` in both `pyproject.toml` and `src/nolane_ai/__init__.py`.

- [ ] **Step 4: Run full verification**

Run:

```bash
pytest -q
python -m compileall -q src scripts
python scripts/verify_protocol.py
```

Expected: all pass; protocol digest remains canonical.

---

### Task 5: PR scientific review, squash, exact-head CI, merge

**Files:**
- Review all branch changes; no new production files required.

**Interfaces:**
- Consumes: completed implementation and CI evidence.
- Produces: one clean PR commit merged only after exact-head green CI.

- [ ] **Step 1: Open PR with explicit scientific boundary**

Document:

- no protocol drift;
- no actual confirmatory data consumed by the PR itself;
- EV-E3 is the maximum result level;
- analytical FLOPs are not hardware-profiler FLOPs;
- challenge/independent replication remain open;
- RED/GREEN evidence.

- [ ] **Step 2: Review changed files for leakage and overclaiming**

Specifically inspect protocol identity, bootstrap seed derivation, result-decision recomputation, Brier sufficient-statistic aggregation, compute ledger reconstruction, raw evidence preservation, CLI overwrite ordering, and artifact self-hash.

- [ ] **Step 3: Verify branch scope**

Compare against current `main`; require ahead 1 after squash, behind 0, total commits 1, and zero changed files under `protocols/`.

- [ ] **Step 4: Run exact-final-head GitHub Actions**

Require `core (3.11)`, `core (3.13)`, and `model-smoke` all success on the exact squash head SHA.

- [ ] **Step 5: Merge only with expected head SHA**

Merge PR only after all gates pass. Record final `main` merge SHA and leave raw scientific outcomes untouched.
