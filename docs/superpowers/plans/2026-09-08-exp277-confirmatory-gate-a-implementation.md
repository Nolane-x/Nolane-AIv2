# EXP-277 Confirmatory Gate A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a preregistered, future-beacon, first-valid-attempt EXP-277 confirmatory program that can return `PROMOTE_TO_NEXT_STAGE`, `HOLD_UNSTABLE`, `KILL_SUBSYSTEM`, `INVALID_RUN`, or a valid pre-inference `NOT_READY_*` outcome without modifying the canonical Stage-A V1 protocol.

**Architecture:** Add dedicated EXP-277 confirmatory modules rather than generalizing EXP-282/297. Gate A freezes DEVELOPMENT-derived sample size, a deterministic pre-beacon trained checkpoint, analysis/challenge/reconstruction code and ceremony lineage. Gate B consumes only a post-freeze public beacon, derives challenge seeds, performs zero parameter updates, emits raw evidence, reconstructs it independently, then applies the frozen paired-bootstrap decision rule.

**Tech Stack:** Python 3.11/3.13, PyTorch CPU, pytest, GitHub Actions, canonical SHA-256 JSON/tensor identities, existing `nolane_ai.protocol` identity/evidence utilities.

**Spec:** `docs/superpowers/specs/2026-09-08-exp277-confirmatory-gate-a-design.md`

## Global Constraints

- Canonical protocol `protocols/stage_a_v1.json` and its digest file MUST NOT change.
- EXP-277 arms remain `arcs_branch` and `oracle_cbrf`; only the oracle arm may receive `oracle_incidence`.
- Primary endpoint is `verified_utility_per_accounted_flop`; MESI is relative gain `+0.10`.
- Protected solution endpoint is `oracle_cbrf - arcs_branch >= -0.005`.
- Confirmatory sample-size bounds are 32..128 with power 0.90 and endpoint alpha 0.025.
- DEVELOPMENT pilot observations may plan `n` but can never be reclassified as confirmatory observations.
- Scientific challenge randomness is derived only after the Gate A freeze from a future public beacon.
- No optimizer step, gradient update or parameter write is permitted after the Gate A checkpoint seal.
- Scientific primary ratio uses no epsilon denominator.
- Deterministic challenge seed material is exactly `protocol_digest | beacon_receipt_digest | EXP-277 | stream | replicate`.
- Integrity/provenance/resource-court failure is `INVALID_RUN`, not scientific `KILL_SUBSYSTEM`.
- Real ceremony uses first-valid-attempt discipline; persistence may not rerun science.

---

### Task 1: Confirmatory prep and sample-size freeze

**Files:**
- Create: `src/nolane_ai/experiments/exp277_confirmatory_prep.py`
- Create: `tests/test_exp277_confirmatory_prep.py`

**Interfaces:**
- Consumes: frozen EXP-277 protocol entry, `NLM-EXP-277-PAIRED-DEV-EVAL-V1`, neural arm registry, source-tree digest.
- Produces: `build_exp277_confirmatory_prep(...) -> dict`, `validate_exp277_confirmatory_prep(payload) -> list[str]`.

- [ ] **Step 1: Write RED tests** proving: EV-E2/UNVERIFIED only; >=32 pilot reps; nonpositive baseline mean -> `NOT_READY_PRIMARY_BASELINE_NONPOSITIVE`; pilot paired effect `d_i=(U_o-U_a)/mean(U_a)`; one-sided alpha 0.025; power 0.90; n clamp 32..128; >128 -> `NOT_READY_VARIANCE_EXCEEDS_MAX_N`; reserved IDs disjoint from training/pilot; no confirmatory seed materialization.
- [ ] **Step 2: Run** `pytest -q tests/test_exp277_confirmatory_prep.py` and require RED from missing module/functions only.
- [ ] **Step 3: Implement** strict protocol constants and normal-approximation planner using `NormalDist`, with finite-value checks and canonical `prep_digest`.
- [ ] **Step 4: Run** focused test and `python -m compileall -q src` until GREEN.
- [ ] **Step 5: Commit** `feat(exp277): freeze confirmatory sample size`.

### Task 2: Deterministic pre-beacon checkpoint

**Files:**
- Create: `src/nolane_ai/experiments/exp277_checkpoint.py`
- Create: `tests/test_exp277_checkpoint.py`

**Interfaces:**
- Consumes: DEVELOPMENT execution artifact + fixed geometry/training lineage.
- Produces: `build_exp277_checkpoint_receipt(...)`, `validate_exp277_checkpoint_receipt(...)`, checkpoint save/load helpers that identify tensors by canonical tensor-byte digests rather than pickle bytes.

- [ ] **Step 1: RED tests** require replay of exactly 16 DEVELOPMENT augmentation replicates; batch digests and final functional digests must match DEVELOPMENT; ARCS oracle separation remains closed; mutated tensor/geometry/training lineage is rejected.
- [ ] **Step 2: Run focused RED.**
- [ ] **Step 3: Implement** deterministic matched-pair reconstruction using existing arm builder/optimizer, canonical ordered tensor digests, functional-only checkpoint receipt and zero dependence on optimizer state for Gate B.
- [ ] **Step 4: GREEN focused tests + compileall.**
- [ ] **Step 5: Commit** `feat(exp277): freeze trained checkpoint identity`.

### Task 3: Post-freeze challenge generator

**Files:**
- Create: `src/nolane_ai/experiments/exp277_challenge_worlds.py`
- Create: `tests/test_exp277_challenge_worlds.py`

**Interfaces:**
- Produces: immutable challenge batch schema generated from a beacon-derived integer seed; no DEVELOPMENT root-seed or `augmentation/evaluation` operator API.

- [ ] **Step 1: RED tests** prove deterministic regeneration; challenge digest changes with seed/replicate; fixed outer geometry; both arms receive same surface/variable tensors; evaluator-only targets/incidence; no arm-visible leakage; no arbitrary stream/root-seed override.
- [ ] **Step 2: Run focused RED.**
- [ ] **Step 3: Implement** structure-dense challenge generation with seed-controlled membership, anchors, latent codes, event order and noise, while preserving EXP-277 semantics.
- [ ] **Step 4: GREEN + compileall.**
- [ ] **Step 5: Commit** `feat(exp277): add post-freeze challenge worlds`.

### Task 4: Public beacon receipt and seed derivation

**Files:**
- Create: `src/nolane_ai/experiments/exp277_beacon.py`
- Create: `tests/test_exp277_beacon.py`

**Interfaces:**
- Produces: `validate_exp277_beacon_receipt`, TEST-ONLY builder, `derive_exp277_challenge_seed`.

- [ ] **Step 1: RED tests** cover ISO-8601 UTC, >=256-bit entropy, digest tamper, TEST-ONLY scientific ineligibility, real evidence classification, strictly-after-freeze timestamp, and exact seed material `protocol_digest|receipt_digest|EXP-277|stream|replicate`.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** dedicated receipt schema; keep external authenticity status separate from cryptographic verification.
- [ ] **Step 4: GREEN + compileall.**
- [ ] **Step 5: Commit** `feat(exp277): bind public beacon challenge seeds`.

### Task 5: Gate A authorization and seal

**Files:**
- Create: `src/nolane_ai/experiments/exp277_confirmatory_authorization.py`
- Create: `tests/test_exp277_confirmatory_authorization.py`

**Interfaces:**
- Produces authorization + Gate A seal binding protocol/source tree, DEVELOPMENT artifact/registry, checkpoint receipt, prep/frozen-analysis/challenge/beacon/executor/reconstruction digests and reserved IDs.

- [ ] **Step 1: RED tamper tests** for every lineage digest, frozen geometry, endpoint alpha, bootstrap count, resource court, oracle separation and any embedded beacon/challenge data before seal.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** EV-E2/UNVERIFIED seal with `confirmatory_data_consumed=false`, `challenge_materialized=false`, `seed_materialization_status=NOT_EXECUTED`, `decision_rule_executed=false`.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** `feat(exp277): seal confirmatory Gate A`.

### Task 6: Reconstruction court

**Files:**
- Create: `src/nolane_ai/experiments/exp277_reconstruction_court.py`
- Create: `tests/test_exp277_reconstruction_court.py`

**Interfaces:**
- Produces reconstruction authorization and raw-row reconstruction/validation from seal + beacon lineage.

- [ ] **Step 1: RED tests** re-hash forged raw rows and require rejection for replicate ID, seed, challenge digest, predictions, solution rate, FLOPs, utility, paired differences, checkpoint identity and oracle leakage.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** independent deterministic regeneration from lineage; never trust aggregates when raw data can be recomputed.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** `feat(exp277): add confirmatory reconstruction court`.

### Task 7: Frozen Gate B raw executor

**Files:**
- Create: `src/nolane_ai/experiments/exp277_confirmatory_executor.py`
- Create: `tests/test_exp277_confirmatory_executor.py`

**Interfaces:**
- Produces `NLM-EXP-277-CONFIRMATORY-CHALLENGE-RAW-V1` without applying the scientific decision rule.

- [ ] **Step 1: RED tests** require exact seal/current source identity, valid post-freeze beacon, checkpoint digest match, zero parameter writes, paired worlds, oracle incidence only to oracle arm, complete analytical cost receipts, raw metrics and scientific/test classification.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** inference-only executor under `torch.no_grad()`, snapshotting functional digests before/after and failing `INVALID_RUN` on drift.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** `feat(exp277): execute frozen confirmatory challenge`.

### Task 8: Frozen analysis and decision rule

**Files:**
- Create: `src/nolane_ai/experiments/exp277_confirmatory_analysis.py`
- Create: `tests/test_exp277_confirmatory_analysis.py`

**Interfaces:**
- Produces deterministic paired-bootstrap primary/protected summaries and final scientific decision.

- [ ] **Step 1: RED tests** cover PROMOTE, KILL-by-primary, KILL-by-protected, HOLD threshold overlap, nonpositive observed denominator, materially unstable bootstrap denominator, no epsilon, 10,000 deterministic samples, sign consistency/divergence reporting, TEST-ONLY non-promotion, and integrity failure -> `INVALID_RUN`.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** ratio-of-means paired bootstrap and paired solution-rate bootstrap with alpha 0.025. Define denominator instability deterministically: any non-finite bootstrap statistic or any bootstrap ARCS mean <=0 marks the endpoint unstable; scientific result is HOLD unless an independently valid frozen kill condition applies.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** `feat(exp277): add frozen confirmatory analysis`.

### Task 9: Gate A/Gate B CLIs and ceremony API

**Files:**
- Create: `src/nolane_ai/experiments/exp277_confirmatory_ceremony.py`
- Create: `scripts/prepare_exp277_confirmatory_gate_a.py`
- Create: `scripts/run_exp277_confirmatory_gate_b.py`
- Create: `tests/test_exp277_confirmatory_ceremony.py`
- Create: `tests/test_exp277_confirmatory_cli.py`

**Interfaces:**
- Gate A CLI transactionally writes prep/checkpoint/authorization/reconstruction/seal and refuses beacon input.
- Gate B CLI requires seal + beacon JSON and exposes no direct challenge seed.

- [ ] **Step 1: RED CLI/ceremony tests** for canonical protocol, no overwrite, invalid input rejection, no beacon in Gate A, no seed in Gate B, transactional publication, TEST-ONLY classification and raw-before-analysis sequencing.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** ceremony orchestration with atomic staged files and explicit wall-energy evidence object.
- [ ] **Step 4: GREEN focused suite.**
- [ ] **Step 5: Commit** `feat(exp277): add confirmatory ceremony CLIs`.

### Task 10: CI TEST-ONLY end-to-end smoke

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: package/version metadata only if the repository's existing versioning convention requires it.

- [ ] **Step 1: Add TEST-ONLY smoke** using tiny geometry: DEVELOPMENT >=32 pilot rows -> Gate A -> synthetic future beacon -> Gate B raw -> reconstruction -> analysis -> assert `scientific_evidence_eligible=false` and no scientific promotion.
- [ ] **Step 2: Run exact focused tests via PR CI.**
- [ ] **Step 3: Fix only deterministic infrastructure failures; never tune scientific thresholds from smoke outcomes.**
- [ ] **Step 4: Require Python 3.11, Python 3.13 and model-smoke GREEN.**
- [ ] **Step 5: Commit** `ci(exp277): exercise test-only confirmatory ceremony`.

### Task 11: Real dormant/armed first-valid ceremony

**Files:**
- Create: `tests/test_exp277_real_ceremony_workflow.py`
- Create: `.github/workflows/exp277-real-gate-b-ceremony.yml`
- Later create only: `.github/ceremony/exp277-real-gate-b-arm.json`

- [ ] **Step 1: RED static-contract tests** require dormant behavior without arm file, exact-head CI waiter, `src/scripts/protocol` drift guard, fixed non-tiny geometry, Gate A before beacon, inference-start artifact before Gate B, no rerun after inference marker and outcome-agnostic artifact upload.
- [ ] **Step 2: Add dormant workflow and obtain exact-head normal CI 3/3 GREEN.**
- [ ] **Step 3: Arm with a commit whose only change is the preregistered arm JSON.**
- [ ] **Step 4: Execute first valid real ceremony. Preserve `NOT_READY`, `PROMOTE`, `HOLD`, `KILL` or `INVALID_RUN` exactly as produced.**
- [ ] **Step 5: Record run id, exact SHA, artifact digest, source-tree digest, checkpoint/seal/beacon lineage and outcome.**

### Task 12: Persistence, integration and closure

**Files:**
- Create: `.github/workflows/exp277-persist-real-gate-b-evidence.yml`
- Create: `evidence/exp277/<date>-real-gate-b/*`
- Modify docs only as needed to record the narrow result and non-claims.

- [ ] **Step 1: Persistence workflow pins the authoritative ceremony run/artifact and verifies every file digest without invoking Gate B.**
- [ ] **Step 2: Persist `SHA256SUMS`, run/artifact metadata, provenance, Gate A artifacts, beacon evidence, raw evidence, analysis and summary.**
- [ ] **Step 3: Audit final PR scope and require no unauthorized frozen-protocol changes.**
- [ ] **Step 4: Require exact-head CI 3/3 GREEN, review-thread audit and mergeability; squash-merge with expected-head guard.**
- [ ] **Step 5: Verify `main` exact merge SHA and post-merge push CI 3/3 GREEN; add closure comment with scientific scope and outcome.**

## Plan Self-Review

- Spec coverage: prep, checkpoint, challenge, beacon, authorization/seal, reconstruction, executor, analysis, CLI, TEST-ONLY CI, real ceremony and persistence are all assigned to explicit tasks.
- Placeholder scan: no implementation step depends on an unspecified threshold. Denominator instability is frozen as any non-finite bootstrap effect or bootstrap ARCS mean <=0.
- Type/identity consistency: all modules consume canonical dict artifacts and return canonical dict artifacts with validators; scientific lineage always binds protocol/source/checkpoint/seal/beacon identities but challenge seed material remains exactly the frozen protocol rule.
- Scope: this plan closes only EXP-277 H-CBRF-01. EXP-279/286/289 remain separate follow-on closures.
