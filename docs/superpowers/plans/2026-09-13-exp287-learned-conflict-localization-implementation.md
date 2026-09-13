# EXP-287 Learned Conflict Localization Court Implementation Plan

> **For agent:** Execute this plan task-by-task with strict TDD. Do not run held-out EXP-287 DEVELOPMENT data until Tasks 1–5 are exact-head GREEN and the release marker is the only new change.

**Goal:** Build and execute the preregistered EXP-287 DEVELOPMENT court that measures whether a supervised learned conflict localizer can recurrently recover at least half of same-weights oracle conflict-core cost value across four independent roots without violating the solution-rate floor.

**Architecture:** Add an EXP-287-specific synthetic conflict generator and one matched canonical neural model with a localizer head. Train one model per root on augmentation-only data, freeze it, then evaluate identical weights in `NULL_CORE_CONTROL`, `LEARNED_CORE_TOP2`, and `ORACLE_CORE_UPPER_BOUND` modes. Root/cross receipts recompute all classifications and preserve EV-E2-only boundaries.

**Tech stack:** Python 3.11/3.13, PyTorch CPU, pytest, repository protocol/evidence/seed helpers, GitHub Actions.

---

## Task 1 — Freeze EXP-287 geometry and matched tri-mode primitives

**Files**
- Create: `protocols/exp287_development_v1.json`
- Create: `protocols/exp287_development_v1.sha256`
- Create: `src/nolane_ai/experiments/exp287_development_geometry.py`
- Create: `src/nolane_ai/experiments/exp287_conflict_worlds.py`
- Create: `src/nolane_ai/experiments/matched_conflict_localizer_arms.py`
- Create: `tests/test_exp287_learned_conflict_localization.py`

**RED first**
1. Add tests that import the missing EXP-287 modules and require:
   - exact frozen geometry: four canonical roots, train64/eval32, batch8, d64/h48/500k, timesteps4, variables8, decoys3, max-search16, noise .05, lr .002, wd0;
   - augmentation/evaluation disjoint lineage;
   - deterministic EXP-287 world digest and exact two-variable core;
   - no evaluator/core truth inside model-visible tensors beyond synthetic state signal;
   - one canonical model exposing null/learned/oracle forward modes;
   - identical parameter inventory and analytical FLOPs across modes;
   - localizer logits depend only on event/variable/recurrent state;
   - pre-contradiction core token is null in every mode;
   - learned top-2 tie-break is ascending variable index;
   - oracle mode marked non-deployable.
2. Commit test-only RED and verify focused failure is missing EXP-287 modules/contracts only.

**GREEN**
3. Implement frozen geometry loader and SHA validation.
4. Implement EXP-287 deterministic generator with EXP-287 seed namespace; do not mutate EXP-286 generator.
5. Implement canonical model with event projection, variable projection, deliberation GRU, localizer head, core adapter, rollback head, verifier head, matched reserve, and exact compute ledger.
6. Implement the three information modes on the same model/state.
7. Run focused tests + frozen Stage-A verifier + compile; commit GREEN.

## Task 2 — Canonical training, same-weights evaluation, and root receipt

**Files**
- Create: `src/nolane_ai/experiments/exp287_learned_conflict_localization.py`
- Extend: `tests/test_exp287_learned_conflict_localization.py`

**RED first**
1. Add tests requiring:
   - one model-init seed per canonical root;
   - train only on augmentation replicates;
   - fixed loss `rollback CE + verifier BCE + localizer BCE` with coefficients 1/1/1;
   - no class weights/calibration/early stopping/sweep hooks;
   - frozen model digest unchanged before/after all three evaluation modes;
   - learned evaluation never consumes core masks for action;
   - oracle evaluation may consume core masks only post-contradiction;
   - control and oracle still execute/charge localizer computation;
   - episode search receipts retain censored outcomes and exact accounted FLOPs.
2. Commit RED and verify failure is absent runner implementation.

**GREEN**
3. Implement canonical train loop, freeze digest, three-mode evaluator, search receipts, held-out localization diagnostics, and primitive root payload.
4. Compute `C0`, `CL`, `CO`, `H_oracle`, `H_learned`, unclipped capture, solution rates, top-2 precision/recall/exact recovery/off-core/AP.
5. Implement root classifier exactly from design, including `ORACLE_HEADROOM_NOT_REPLICATED` fail-closed state.
6. Run focused tests + protocol verifier + compile; commit GREEN.

## Task 3 — Fail-closed receipts and cross-root reducer

**Files**
- Create: `src/nolane_ai/experiments/exp287_receipts.py`
- Create: `tests/test_exp287_receipts.py`

**RED first**
1. Require validators to reject:
   - forged root classification;
   - wrong geometry/root index/root prefix;
   - training/evaluation overlap;
   - duplicate roots;
   - model digest drift across modes;
   - evaluation core leakage into learned mode;
   - pre-contradiction core delivery;
   - boundary overclaims;
   - integer-key/canonical JSON instability;
   - cross decision inconsistent with four recomputed root decisions.
2. Require serialize→parse→serialize byte identity and SHA sidecar equality.
3. Commit RED; verify only receipts module is missing.

**GREEN**
4. Implement canonical JSON bytes/SHA helpers, root validator/builder, and four-root cross reducer.
5. Cross classifier outputs only:
   - `LOCALIZATION_VALUE_RECURRENT`
   - `LOCALIZATION_VALUE_INTERMITTENT`
   - `LOCALIZATION_VALUE_NOT_ESTABLISHED`
   - `ORACLE_REPLICATION_INCOMPLETE`.
6. Only recurrent sets `successor_design_authorized=true` and `authorization_scope=DESIGN_EXP288_BACKJUMP_COURT_ONLY`; all scientific/confirmatory/challenge/promotion flags remain false.
7. Run focused receipt tests + complete EXP-287 suite + compile; commit GREEN.

## Task 4 — Frozen CLI surface

**Files**
- Create: `scripts/run_exp287_learned_conflict_localization_dev.py`
- Create: `scripts/reduce_exp287_learned_conflict_localization_dev.py`
- Create: `tests/test_exp287_cli.py`

**RED first**
1. Tests require `root` and `cross` CLI operations and prove `--help` exposes no knobs for geometry, roots, sample counts, top-k, loss weights, capture threshold, precision threshold, or model dimensions.
2. Root CLI accepts only canonical index plus output path; cross CLI accepts exactly four validated root receipts plus output path.
3. Commit RED; verify failures are missing scripts.

**GREEN**
4. Implement fixed-surface CLIs, atomic/no-overwrite output, receipt + `.sha256` sidecar, and fail-closed validation before publication.
5. Run CLI tests, complete EXP-287 suite, protocol verifier, compile; commit GREEN.

## Task 5 — Exact-head CI, one-shot DEVELOPMENT workflow, and freeze guard

**Files**
- Create: `.github/workflows/exp287-learned-conflict-localization-ci.yml`
- Create: `.github/workflows/exp287-learned-conflict-localization-development.yml`
- Create: `.github/workflows/exp287-learned-conflict-localization-freeze-guard.yml`
- Create: `tests/test_exp287_workflow_contract.py`

**RED first**
1. Tests require:
   - dedicated contract CI on relevant paths;
   - DEVELOPMENT workflow triggered only by exact release-marker path on PR/push context, no `workflow_dispatch`;
   - preflight verifies exact marker bytes/current head/frozen geometry/protocol;
   - four root jobs, then one cross reducer;
   - every artifact publishes receipt + SHA and uses finite retention;
   - freeze guard permits exactly one authoritative development run and identifies duplicates;
   - no secrets, confirmatory beacon, challenge seed, or scientific promotion path.
2. Commit RED and verify only workflow files are missing.

**GREEN**
3. Implement the workflows and marker contract template, but do **not** create the marker yet.
4. Run dedicated/focused/generic CI at exact pre-marker head.
5. Freeze guard must report zero EXP-287 DEVELOPMENT runs at pre-marker head.
6. Commit GREEN only after all workflow contract tests pass.

## Task 6 — Pre-data freeze, one-shot DEVELOPMENT execution, audit, and program update

**Files**
- Create only after all pre-data gates GREEN: `protocols/exp287-learned-conflict-localization-release.lock`
- Update after valid sealed result: `docs/superpowers/specs/2026-09-13-v0161-program-closure-matrix-design.md`
- Update PR body with immutable ledger; do not rewrite scientific code post-result.

**Release**
1. Verify exact pre-marker head: dedicated EXP-287 suite, focused relevant Stage-A/EXP-286 regression, generic core/model smoke, protocol digest, compile, and freeze guard with zero prior runs.
2. Commit a marker-only release commit. Audit that the commit changes exactly one file and exact frozen bytes.
3. Freeze branch semantics after marker. Do not patch model/threshold/root/sample geometry in place after held-out visibility.

**Execution**
4. Confirm freeze guard selects exactly one authoritative DEVELOPMENT run.
5. Require preflight GREEN before any root job.
6. Require 4/4 root artifacts SUCCESS with receipt/sidecar verification.
7. Require cross reducer SUCCESS and sealed final artifact.

**Independent audit**
8. Download final artifacts; independently recompute ZIP SHA, receipt SHA sidecars, canonical roundtrip, primitive classifications, cross decision, model-digest equality, and evidence boundaries.
9. Record exact root metrics and final decision without favorable reinterpretation.

**Closure**
10. Update closure matrix:
    - recurrent → EXP-287 `DEVELOPMENT_OPEN`/positive scoped result and EXP-288 `AUTHORIZED_NEXT_STAGE` for design only;
    - any non-recurrent valid result → EXP-287 `DEVELOPMENT_CLOSED_NEGATIVE`, EXP-288 remains `BLOCKED_ON_PARENT`;
    - oracle replication incomplete → record invalid-for-learned-conclusion dependency state without calling localizer negative.
11. Close the DEVELOPMENT PR unmerged if the result is negative/non-authorizing; merge only reusable non-falsified infrastructure if evidence/scope and final branch review justify it. Never merge a falsified learned mechanism merely to preserve code.
12. Re-read `main` after closure and record that frozen Stage-A bytes were unchanged.

## Verification before any completion claim

Run/recheck fresh evidence on exact final head:

- complete EXP-287 pytest suite;
- relevant EXP-286 regression suite;
- frozen Stage-A protocol verifier;
- generic core Python 3.11 and 3.13;
- model-smoke;
- compileall;
- workflow/freeze-guard status;
- artifact and receipt hashes if data was executed;
- PR merge/closed state and final `main` SHA.

Engineering GREEN, DEVELOPMENT evidence, and EV-E3 scientific evidence must be reported as separate categories.