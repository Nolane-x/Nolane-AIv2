# EXP-297 Confirmatory Gate A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and verify the complete pre-beacon EXP-297 confirmatory machinery, including sample-size prep, hidden challenge generation, authorization/reconstruction, Gate A sealing, synthetic-beacon Gate B execution/analysis code, CLI publication, CI coverage, and package 0.16.0.

**Architecture:** Build dedicated EXP-297 confirmatory modules patterned after EXP-282 but specialized for semantic-fidelity balanced accuracy, two safety rates, exact witness reconstruction, and future-beacon seed derivation. Gate A produces only EV-E2/UNVERIFIED freeze artifacts and never consumes a real beacon; Gate B code is frozen and tested only with explicit TEST-ONLY synthetic beacons.

**Tech Stack:** Python 3.11/3.13, stdlib `statistics`, `math`, `hashlib`, JSON, PyTorch for matched neural-arm reconstruction, pytest, existing Nolane protocol/evidence helpers.

**Spec:** `docs/superpowers/specs/2026-09-08-exp297-confirmatory-gate-a-design.md`

## Global Constraints

- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- EXP-297 primary endpoint is `semantic_fidelity_balanced_accuracy`, higher is better.
- MESI is absolute gain `0.10`.
- Protected floors are `wrong_formalization_authority_rate <= 0.05` and `faithful_formalization_rejection_rate <= 0.10`.
- Confirmatory sample size is paired, power target `0.90`, min `32`, max `128`.
- Familywise alpha is `0.05`; Gate A freezes `alpha_per_endpoint = 0.05 / 3`.
- Gate A must not materialize a public beacon, challenge seed, or challenge candidate.
- Every Gate A artifact remains `EV-E2 / UNVERIFIED`; semantic authority is never promoted.
- Candidate truth and trap family remain evaluator-only.
- Package target after integration is `0.16.0`.

---

### Task 1: Confirmatory prep and sample-size freeze

**Files:**
- Create: `src/nolane_ai/experiments/exp297_confirmatory_prep.py`
- Create: `tests/test_exp297_confirmatory_prep.py`

**Interfaces:**
- Consumes: `validate_exp297_execution(payload)`, frozen EXP-297 protocol item, extended registry.
- Produces: `build_exp297_confirmatory_prep(...) -> dict[str, Any]`, `validate_exp297_confirmatory_prep(payload) -> list[str]`.

- [ ] Write tests that build a 32-replicate DEVELOPMENT pilot and assert: per-replicate paired BA effects reconstruct from raw rows; pilot reuse is forbidden; required n is frozen into `[32,128]`; reserved IDs are contiguous and disjoint; EV-E2/UNVERIFIED flags stay false for consumption/materialization/promotion.
- [ ] Add re-hash tamper tests for paired SD, MESI, alpha allocation, required n, confirmatory n, reserved IDs, and overlapping development IDs.
- [ ] Run `pytest -q tests/test_exp297_confirmatory_prep.py` and verify RED because the module does not exist.
- [ ] Implement protocol-contract validation, replicate metric reconstruction, paired SD, one-sided normal-approximation n formula, deterministic reservation, digest, and semantic validator.
- [ ] Run focused tests and verify GREEN.
- [ ] Commit `feat: add EXP-297 confirmatory prep freeze`.

### Task 2: Hidden challenge world generator and leakage court

**Files:**
- Create: `src/nolane_ai/experiments/exp297_challenge_worlds.py`
- Create: `tests/test_exp297_challenge_worlds.py`

**Interfaces:**
- Consumes: `FidelityCourt`, CPS primitives, opaque-ID conventions from `exp297_fidelity_worlds.py`.
- Produces: `generate_exp297_challenge_world(seed: int) -> FidelityWorldBatch` and `CHALLENGE_CONTRACT_DIGEST` helper/payload.

- [ ] Write tests requiring deterministic generation, all eight frozen strata, one faithful + one wrong per stratum, 16 compile-valid candidates, exact semantic provenance across many seeds, opaque IDs, neutral labels/world IDs, and position/polarity non-correlation across seeds.
- [ ] Add tests proving challenge instances differ materially from DEVELOPMENT worlds for the same integer seed while preserving the same semantic contract.
- [ ] Run focused test and observe RED.
- [ ] Implement seed-varying domain size, relation offset/orientation, allowed-row ordering, neutral variable permutation, transformation parameters, neutral serialization, and deterministic shuffle.
- [ ] Run focused tests and verify GREEN.
- [ ] Commit `feat: add hidden EXP-297 challenge worlds`.

### Task 3: Public-beacon receipt and deterministic seed contract

**Files:**
- Create: `src/nolane_ai/experiments/exp297_beacon.py`
- Create: `tests/test_exp297_beacon.py`

**Interfaces:**
- Produces: `build_test_beacon_receipt(...)`, `validate_exp297_beacon_receipt(...)`, `derive_exp297_challenge_seed(...)`.

- [ ] Write RED tests for canonical receipt digest, UTC timestamp parsing, entropy hex validation, strict `published_at > freeze_timestamp`, TEST-ONLY marker behavior, and deterministic seed derivation `SHA256(protocol_digest|freeze_commit_sha|beacon_receipt_digest|EXP-297|stream|replicate)`.
- [ ] Assert no API accepts a raw operator challenge seed.
- [ ] Implement the minimal receipt and derivation contract using stdlib only.
- [ ] Verify focused tests GREEN and commit `feat: freeze EXP-297 beacon seed contract`.

### Task 4: Execution authorization and reconstruction authorization

**Files:**
- Create: `src/nolane_ai/experiments/exp297_confirmatory_execution_court.py`
- Create: `src/nolane_ai/experiments/exp297_reconstruction_court.py`
- Create: `tests/test_exp297_confirmatory_execution_court.py`
- Create: `tests/test_exp297_reconstruction_court.py`

**Interfaces:**
- Produces `authorize_exp297_confirmatory_execution(...)`, `validate_exp297_confirmatory_execution_authorization(...)`, `build_exp297_confirmatory_reconstruction(...)`, `validate_exp297_confirmatory_reconstruction(...)`.

- [ ] Write RED tests requiring `AUTHORIZED_NOT_EXECUTED`, exact reserved IDs, frozen analysis/sample-size digests, model geometry, court ceiling, pair audit digest, challenge contract digest, code digests, and no seeds/beacon/challenge candidates.
- [ ] Write tamper tests for code lineage, pair audit, candidate count, geometry, reserved IDs, and forbidden materialization flags.
- [ ] Implement minimal authorizers/validators with canonical digests and cross-artifact lineage checks.
- [ ] Verify GREEN and commit `feat: add EXP-297 confirmatory authorization courts`.

### Task 5: Freeze Gate B raw executor and reconstruction validator

**Files:**
- Create: `src/nolane_ai/experiments/exp297_confirmatory_executor.py`
- Create: `tests/test_exp297_confirmatory_executor.py`

**Interfaces:**
- Consumes Gate A reconstruction authorization + beacon receipt.
- Produces `execute_exp297_confirmatory_challenge(...) -> dict[str, Any]`, `validate_exp297_confirmatory_raw(...) -> list[str]`.

- [ ] Write RED tests using only TEST-ONLY synthetic beacons. Assert beacon must postdate freeze timestamp; seed lineage is reconstructible; reserved IDs/order are exact; each raw row reproduces challenge candidate, compile result, court receipt, authority, neural/accounted cost, and evaluator truth.
- [ ] Add re-hash tamper tests for beacon receipt, seed lineage, candidate order/drop, candidate digest, truth/stratum, witness, authority, neural cost, semantic cost, `court_inconclusive -> court_accept`, and scientific flags.
- [ ] Implement execution by rebuilding matched arms from frozen geometry/model seed, deriving only beacon-based seeds, generating challenge worlds, and recording reconstructible raw rows.
- [ ] Implement validator that independently re-derives seeds/worlds/court/matched-pair/costs and rejects TEST-ONLY artifacts as scientific evidence.
- [ ] Verify GREEN and commit `feat: freeze EXP-297 confirmatory executor`.

### Task 6: Frozen confirmatory analysis

**Files:**
- Create: `src/nolane_ai/experiments/exp297_confirmatory_analysis.py`
- Create: `tests/test_exp297_confirmatory_analysis.py`

**Interfaces:**
- Produces `bootstrap_exp297_paired_gain`, `wilson_upper_bound`, `build_exp297_confirmatory_analysis(...)`, `validate_exp297_confirmatory_analysis(...)`.

- [ ] Write RED tests for 10,000-sample deterministic paired bootstrap, one-sided Wilson upper bounds, aggregate count reconstruction, three-way decision rule, frozen analysis digest, and rejection of TEST-ONLY beacon evidence for scientific promotion.
- [ ] Test PROMOTE/HOLD/KILL fixtures, including safety failure overriding primary gain.
- [ ] Implement deterministic bootstrap seeded from frozen lineage, Wilson interval, safety summaries, decision rule, and validator.
- [ ] Verify GREEN and commit `feat: freeze EXP-297 confirmatory analysis`.

### Task 7: Gate A ceremony seal

**Files:**
- Create: `src/nolane_ai/experiments/exp297_confirmatory_ceremony.py`
- Create: `tests/test_exp297_confirmatory_ceremony.py`

**Interfaces:**
- Produces `seal_exp297_confirmatory_gate_a(...)`, `validate_exp297_confirmatory_gate_a_seal(...)`.

- [ ] Write RED tests for schema `NLM-EXP-297-CONFIRMATORY-GATE-A-SEAL-V1`, EV-E2/UNVERIFIED, narrow `confirmatory_ready=true`, no consumption/materialization/decision/promotion, exact freeze SHA/timestamp, embedded/bound prep + authorization + reconstruction, and code-tree identity closure.
- [ ] Add tamper tests for freeze commit SHA/timestamp, frozen-analysis digest, challenge contract, code digests, reserved IDs, and forbidden flags.
- [ ] Implement seal and validator; require code digests to agree and reject any preexisting beacon material.
- [ ] Verify GREEN and commit `feat: seal EXP-297 confirmatory Gate A`.

### Task 8: Hardened Gate A and Gate B CLIs

**Files:**
- Create: `scripts/prepare_exp297_confirmatory_gate_a.py`
- Create: `scripts/run_exp297_confirmatory_gate_b.py`
- Create: `tests/test_exp297_confirmatory_cli.py`

**Interfaces:**
- Gate A CLI consumes DEVELOPMENT execution + registry and publishes prep/auth/reconstruction/seal transactionally.
- Gate B CLI consumes Gate A seal + beacon receipt; it exposes no raw seed argument.

- [ ] Write RED CLI tests for canonical protocol verification, no-overwrite, minimum 32 pilot replicates, atomic rollback, Gate A beacon rejection, Gate B missing/early beacon rejection, no `--seed` option, TEST-ONLY execution mode, and output validation.
- [ ] Implement shared staging/atomic publication pattern from EXP-297 DEVELOPMENT CLI.
- [ ] Gate A must accept explicit `--freeze-commit-sha` and `--freeze-commit-timestamp-utc` only for the branch/CI test lane; production release instructions bind them to the exact merge commit during sealing.
- [ ] Gate B must refuse TEST-ONLY beacon promotion and write raw/analysis artifacts only after full validation.
- [ ] Verify GREEN and commit `feat: add EXP-297 confirmatory ceremony CLIs`.

### Task 9: CI, docs, version, exact-head review and merge

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`

**Interfaces:** package/release integration only.

- [ ] Add all new EXP-297 confirmatory tests to model-smoke; core lane must remain torch-independent.
- [ ] Add a Gate A CI smoke that first generates a 32-replicate tiny DEVELOPMENT pilot, then runs Gate A prep/seal with synthetic freeze metadata. Do not invoke real-beacon Gate B in normal CI.
- [ ] Add a focused TEST-ONLY Gate B unit/CLI smoke whose output is explicitly non-scientific and removed after execution.
- [ ] Update README with Gate A status, artifact schemas, commands, future-beacon boundary, and non-claims.
- [ ] Bump package/version module to `0.16.0`.
- [ ] Run exact-head CI; inspect all core 3.11/core 3.13/model-smoke jobs and every EXP-297 confirmatory step.
- [ ] Review changed-file set and confirm frozen Stage-A protocol files are absent.
- [ ] Update PR body with exact-head evidence, mark ready, and squash-merge only with `expected_head_sha`.
- [ ] Verify `main` points to the squash SHA and post-merge push CI is 3/3 success before declaring Gate A engineering-closed.
