# EXP-289 Confirmatory Gate-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed EXP-289 Stage-A confirmatory path from DEVELOPMENT RDER evidence through immutable pre-beacon authority, while preserving the frozen protocol and EV-E2/UNVERIFIED boundary until a valid real Gate B.

**Architecture:** Keep EXP-289-specific statistics separate from EXP-286 log-cost machinery. Gate A derives paired relative RDER reductions only from reconstruction-valid DEVELOPMENT rows, freezes sample size before challenge randomness, and refuses zero/invalid baseline denominators. Later authority layers bind the exact trained state, code tree, analysis contract, challenge contract, and post-freeze beacon without allowing CI rehearsal to promote evidence.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, GitHub Actions, canonical SHA-256 evidence contracts.

**Spec:** `docs/superpowers/specs/2026-09-10-exp289-confirmatory-gate-a-design.md`

## Global Constraints

- Frozen protocol file and digest must not change: `protocols/stage_a_v1.json` / canonical digest `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- EXP-289 primary endpoint remains `repeat_dead_end_rate`, lower is better.
- MESI remains 0.25 relative reduction.
- Confirmatory paired n remains 32..128; power target remains 0.90; familywise alpha remains 0.05.
- Protected over-prune ceiling remains 0.005; protected solution floor remains `local_nogood >= no_nogood - 0.01`.
- Zero-opportunity episodes remain in raw evidence/safety accounting but are excluded from the primary RDER denominator.
- No epsilon denominator rescue.
- DEVELOPMENT observations are never reclassified as confirmatory observations.
- No real beacon/challenge materialization before immutable Gate-A authority is complete.
- TEST-ONLY ceremony paths remain EV-E2 / UNVERIFIED.

## Progress ledger — 2026-09-10

- [x] Task 1 RED observed at `762b6893b2c52e8f4110da260f1428d14a2c1f64`; RDER Gate-A prep implemented and canonical protected-endpoint parser defect fixed without changing the 0.005 guard.
- [x] Task 2 transactional CLI + focused workflow implemented; tiny 32-pair rehearsal remains EV-E2 / UNVERIFIED and pre-beacon.
- [x] Task 3 authoritative non-tiny DEVELOPMENT geometry frozen and exact-head verified at `d0c4fd0a0e8ca4cac5dc328418ab1fc128593446`. Authoritative DEVELOPMENT planning: n=32, no-nogood mean RDER=1.0, local-nogood mean RDER=0.0, paired mean relative reduction=1.0, paired SD=0.0, local over-prune=0.0, both verified-solution rates=1.0, Gate A `CONFIRMATORY_GATE_A_PREPARED`, confirmatory_n=32. This is DEVELOPMENT planning evidence only, not confirmatory inference.
- [x] Task 4 RED observed at `442430eb9522ef3a6071756059a2338778059646`: 3 failures, all from the deliberately missing `exp289_checkpoint` module; 55 other focused tests passed.
- [ ] Task 4 GREEN candidate `76beb3c977ff70331482b67e61dc6d0cbf1219ba`: exact trained-state functional-only checkpoint/replay court implemented. Normal CI core 3.11, core 3.13, and model-smoke are GREEN; exact-head EXP-289 focused CI remains the outstanding verification gate before Task 5.
- [ ] Task 5 execution authorization + immutable pre-beacon seal. Must additionally bind checkpoint receipt digest, scientific identity digest, checkpoint file SHA-256, and checkpoint execution-contract digest; no seed/beacon/challenge observations may be present.
- [ ] Task 6 TEST-ONLY post-freeze challenge + Gate-B firewall.
- [ ] Task 7 exact-head closure / real-ceremony eligibility audit.

---

### Task 1: RDER-specific Gate-A preparation

**Files:**
- Create: `src/nolane_ai/experiments/exp289_confirmatory_prep.py`
- Create: `tests/test_exp289_confirmatory_prep.py`

**Interfaces:**
- Consumes: `validate_exp289_paired_artifact(payload)` or the existing EXP-289 DEVELOPMENT validator, neural arm registry, canonical frozen Stage-A protocol.
- Produces: `frozen_exp289_gate_a_contract() -> dict`, `prepare_exp289_confirmatory_gate_a(...) -> dict`, `validate_exp289_confirmatory_prep(payload) -> list[str]`.

- [x] **Step 1: Write the failing contract tests**
- [x] **Step 2: Run the focused test and confirm RED**
- [x] **Step 3: Implement minimal Gate-A statistics**
- [x] **Step 4: Run Gate-A tests GREEN**
- [x] **Step 5: Commit**

---

### Task 2: Transactional Gate-A CLI and focused CI

**Files:**
- Create: `scripts/prepare_exp289_confirmatory_gate_a.py`
- Create: `tests/test_exp289_confirmatory_prep_cli.py`
- Create: `.github/workflows/exp289-confirmatory-ci.yml`

- [x] **Step 1: Write RED tests**
- [x] **Step 2: Observe RED**
- [x] **Step 3: Implement atomic publication and workflow**
- [x] **Step 4: Verify GREEN**
- [x] **Step 5: Commit**

---

### Task 3: Authoritative DEVELOPMENT geometry and Gate-A evidence

**Files:**
- Create: `protocols/exp289_authoritative_development_v1.json`
- Create: `tests/test_exp289_authoritative_development.py`
- Modify: `.github/workflows/exp289-confirmatory-ci.yml`

- [x] **Step 1: RED-test exact geometry binding**
- [x] **Step 2: Observe RED, then add a fixed non-tiny DEVELOPMENT geometry**
- [x] **Step 3: Execute exactly the frozen geometry and bind its canonical SHA-256 into execution/prep lineage**
- [x] **Step 4: Verify exact-head focused + normal CI at `d0c4fd0a0e8ca4cac5dc328418ab1fc128593446`**

Authoritative DEVELOPMENT result at that exact head remains EV-E2 / UNVERIFIED: geometry digest `948347a9f06313cc1a86ccc3efc0aed4298a3c6b3cd7b35803e1710778ea3499`; scientific execution digest `017f3bbcba5f45fe82809ebb958cbf98d03d10900e08731ef08d0066df2eaf23`; enveloped execution digest `44e451e1557e56fe3a30fb2ce3f92b03cef548143b899c79d97830978f928283`; Gate A prepared n=32 with no challenge materialization.

---

### Task 4: Exact trained-state checkpoint court

**Files:**
- Modify if required by RED evidence: `src/nolane_ai/experiments/exp289_paired_runner.py`
- Create: `src/nolane_ai/experiments/exp289_checkpoint.py`
- Create: `tests/test_exp289_checkpoint.py`

- [x] **Step 1: RED tests require optimizer/loss/final-state identity.**
- [x] **Step 2: Observe RED at `442430eb9522ef3a6071756059a2338778059646` (3 missing-module failures; 55 other focused tests passed).**
- [x] **Step 3: Add exact replay + canonical checkpoint/file/receipt/scientific-identity digests at `76beb3c977ff70331482b67e61dc6d0cbf1219ba`.**
- [ ] **Step 4: Verify checkpoint + existing EXP-289 suites GREEN on exact `76beb3c...`.** Normal CI is GREEN; focused run is pending.
- [x] **Step 5: Commit candidate.**

---

### Task 5: Execution court and immutable pre-beacon seal

**Files:**
- Create: `src/nolane_ai/experiments/exp289_challenge_worlds.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_authorization.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_ceremony.py`
- Create: `tests/test_exp289_confirmatory_authorization.py`
- Create: `tests/test_exp289_confirmatory_ceremony.py`

**Interfaces:**
- Authorization binds protocol digest, code-tree digest, authoritative geometry digest, scientific + enveloped DEVELOPMENT artifact identities, prep digest, exact trained-state checkpoint receipt/file/scientific identity, challenge-contract digest and frozen confirmatory n.
- Seal binds freeze commit SHA/timestamp and explicitly contains no beacon entropy, challenge seed or confirmatory observations.

- [ ] **Step 1: RED tamper tests** for each binding, including checkpoint identity/file/receipt tampering.
- [ ] **Step 2: Observe RED.**
- [ ] **Step 3: Implement canonical authorization + seal digests.**
- [ ] **Step 4: Verify GREEN and commit.**

---

### Task 6: TEST-ONLY post-freeze challenge and Gate-B firewall

**Files:**
- Create: `src/nolane_ai/experiments/exp289_beacon.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_executor.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_analysis.py`
- Create: tests for beacon, executor, analysis, and TEST-ONLY non-promotion.

- [ ] **Step 1: RED tests prove pre-freeze/replayed/tampered beacons fail and TEST-ONLY cannot promote.**
- [ ] **Step 2: Observe RED.**
- [ ] **Step 3: Implement generator/executor/analysis.**
- [ ] **Step 4: Verify focused TEST-ONLY ceremony GREEN and commit.**

---

### Task 7: Exact-head closure and real-ceremony eligibility audit

- [ ] **Step 1:** Lock PR HEAD SHA and run/fetch focused EXP-289 CI, normal CI core 3.11/3.13/model-smoke, and frozen-protocol verification.
- [ ] **Step 2:** Audit diff from `main@8c0b5add4387cbf244d5df61b2ef1a6ebac79a66`; assert no changes to `protocols/stage_a_v1.json` or `.sha256`.
- [ ] **Step 3:** If Gate A is NOT_READY, persist that EV-E2 disposition and do not build a real beacon ceremony. If READY, freeze exact source/config/evaluator/analysis identity before any future public beacon.
- [ ] **Step 4:** Invoke `superpowers:verification-before-completion` and `superpowers:requesting-code-review` before any merge-ready claim.
